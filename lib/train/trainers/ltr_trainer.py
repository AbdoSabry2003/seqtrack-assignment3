# lib/train/trainers/ltr_trainer.py
import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from datetime import datetime, timedelta
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc
import threading  # 🔥 OPTIMIZATION: Added for async upload


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False, cfg=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
            cfg - Configuration object (🔥 NEW for Assignment 3)
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # 🔥 NEW: Assignment 3 additions
        self.cfg = cfg
        self.training_start_time = None
        self.epoch_start_time = None
        self.last_50_start_time = None

        # 🔥 Setup HuggingFace (optional)
        self.hf_token = getattr(cfg.TRAIN, 'HF_TOKEN', None) if cfg else None
        self.hf_repo = getattr(cfg.TRAIN, 'HF_REPO', None) if cfg else None

        # 🔥 Enhanced logging file
        self.assignment_log_file = os.path.join(settings.save_dir, 'logs', 'assignment_3_training.log')
        os.makedirs(os.path.dirname(self.assignment_log_file), exist_ok=True)

        if settings.local_rank in [-1, 0]:
            self._log_assignment(f"\n{'=' * 80}")
            self._log_assignment(f"ASSIGNMENT 3 - SeqTrack Training Started")
            self._log_assignment(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if cfg:
                seed = getattr(cfg.TRAIN, 'SEED', 'Not set')
                self._log_assignment(f"Seed: {seed}")
                selected_classes = getattr(cfg.DATA.TRAIN, 'SELECTED_CLASSES', 'All classes')
                self._log_assignment(f"Selected Classes: {selected_classes}")
            self._log_assignment(f"{'=' * 80}\n")

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def _log_assignment(self, message):
        """Log to assignment log file and print"""
        print(message)
        with open(self.assignment_log_file, 'a') as f:
            f.write(message + '\n')

    def _format_time(self, seconds):
        """Convert seconds to H:MM:SS format"""
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"

    def _capture_rng_states(self):
        """Capture all RNG states for perfect reproducibility"""
        import random, numpy as np, torch
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        return state

    def _restore_rng_states(self, state):
        """Restore all RNG states"""
        import random, numpy as np, torch
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda_all") is not None and torch.cuda.is_available():
            cuda_states = state["torch_cuda_all"]
            n = min(len(cuda_states), torch.cuda.device_count())
            curr = torch.cuda.get_rng_state_all()
            curr[:n] = cuda_states[:n]
            torch.cuda.set_rng_state_all(curr)
        torch.backends.cudnn.deterministic = state.get("cudnn_deterministic", True)
        torch.backends.cudnn.benchmark = state.get("cudnn_benchmark", False)

    def load_checkpoint(self, path):
        """Load checkpoint and restore complete training state"""
        ckpt = torch.load(path, map_location='cpu')

        # Load model state
        state_dict = ckpt['state_dict']
        if hasattr(self.actor.net, 'module'):
            self.actor.net.module.load_state_dict(state_dict, strict=True)
        else:
            self.actor.net.load_state_dict(state_dict, strict=True)

        # Load optimizer state
        if self.optimizer and ckpt.get('optimizer'):
            self.optimizer.load_state_dict(ckpt['optimizer'])

        # Load scheduler state
        if self.lr_scheduler and ckpt.get('lr_scheduler'):
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

            # Sync the epoch number with scheduler
            try:
                self.lr_scheduler.last_epoch = int(ckpt['epoch'])
            except Exception:
                pass

        # Load AMP scaler state
        if hasattr(self, 'scaler') and self.use_amp and ckpt.get('scaler'):
            self.scaler.load_state_dict(ckpt['scaler'])

        # Restore RNG states
        if ckpt.get('rng_state'):
            self._restore_rng_states(ckpt['rng_state'])

        # Set starting epoch so that next loop starts at epoch+1
        last_epoch = int(ckpt['epoch'])
        self.epoch = last_epoch  # BaseTrainer loop will start at self.epoch + 1

        self._log_assignment(f"🔁 Resumed from {path} -> next epoch will be {self.epoch + 1}")
        return self.epoch + 1

    def _reseed_epoch(self):
        """🔥 NEW: Reset seed at the start of each epoch (Assignment 3 requirement)"""
        if not self.cfg or getattr(self.cfg.TRAIN, 'SEED', None) is None:
            return

        seed = self.cfg.TRAIN.SEED  # Should be 13 for team 13

        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # [A3-SEED] also reseed dataloader generators
        for loader in self.loaders:
            gen = getattr(loader, 'generator', None)
            if gen is not None:
                gen.manual_seed(seed)

        if self.settings.local_rank in [-1, 0]:
            self._log_assignment(f"🌱 Reset seed to {seed} at start of epoch {self.epoch}")

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        # 🔥 MODIFIED: Track samples, not batches
        self.epoch_start_time = time.time()
        self.last_50_start_time = time.time()
        self.samples_processed = 0
        # 🔥 Get sample logging interval from config
        sample_log_interval = getattr(self.cfg.TRAIN, 'PRINT_INTERVAL_SAMPLES', 50) if self.cfg else 50
        self.next_log_at = sample_log_interval
        self._sample_log_interval = sample_log_interval

        # Use configured samples per epoch
        # 🔥 MODIFIED: Calculate total samples based on loader type
        if loader.training:
            total_samples = self.cfg.DATA.TRAIN.SAMPLE_PER_EPOCH if self.cfg else len(loader) * loader.batch_size
        else:
            # For validation, calculate dynamically based on loader length
            total_samples = len(loader) * loader.batch_size

        if self.training_start_time is None:
            self.training_start_time = time.time()

        for i, data in enumerate(loader, 1):
            # 🔥 OPTIMIZATION: Use non_blocking transfer
            if self.move_data_to_gpu:
                try:
                    data = data.to(self.device, non_blocking=True)
                except TypeError:
                    # Fallback for older PyTorch or custom data structures
                    data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # 🔥 OPTIMIZATION: REMOVED per-batch synchronization for speed
            # torch.cuda.synchronize() was here - removed for performance

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self.samples_processed += batch_size
            self._update_stats(stats, batch_size, loader)

            # 🔥 MODIFIED: Log every 50 SAMPLES, not iterations
            while self.samples_processed >= self.next_log_at and self.settings.local_rank in [-1, 0]:
                self._log_progress_samples(self.next_log_at, total_samples, stats, loader)
                self.next_log_at += self._sample_log_interval
                self.last_50_start_time = time.time()

            # print statistics (original behavior)
            self._print_stats(i, loader, batch_size)

    def _log_progress_samples(self, current_samples, total_samples, stats, loader):
        """🔥 NEW: Log progress every 50 samples with exact format required"""

        # 🔥 OPTIMIZATION: Synchronize only before timing measurements
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        now = time.time()

        # Calculate timings
        time_for_50 = now - self.last_50_start_time
        time_since_start = now - self.training_start_time

        # Estimate time left
        if current_samples > 0:
            avg_time_per_sample = (now - self.epoch_start_time) / current_samples
            samples_left = total_samples - current_samples
            time_left = avg_time_per_sample * samples_left
        else:
            time_left = 0

        # Calculate samples per second
        samples_per_sec = current_samples / max((now - self.epoch_start_time), 1e-6)

        # Get loss value (look for 'loss' or 'Loss' in stats)
        loss_val = "N/A"
        for key in stats.keys():
            if 'loss' in key.lower():
                loss_val = f"{stats[key]:.4f}"
                break

        # Get IoU if available
        iou_val = "N/A"
        for key in stats.keys():
            if 'iou' in key.lower():
                iou_val = f"{stats[key]:.4f}"
                break

        # [A3-LOG] Add loader name prefix for clarity
        prefix = f"[{loader.name}] "

        # Format message EXACTLY as required
        message = prefix + (
            f"Epoch {self.epoch} : {current_samples} / {total_samples} samples , "
            f"time for last {self._sample_log_interval} samples : {self._format_time(time_for_50)} hours , "
            f"time since beginning : {self._format_time(time_since_start)} hours , "
            f"time left to finish the epoch : {self._format_time(time_left)} hours , "
            f"Loss: {loss_val}"
        )

        # Add IoU if available
        if iou_val != "N/A":
            message += f" , IoU: {iou_val}"

        # Add samples/sec for performance metrics
        message += f" , samples/sec: {samples_per_sec:.2f}"

        self._log_assignment(message)

    def _save_checkpoint_with_upload(self):
        """🔥 NEW: Save checkpoint and upload to HuggingFace"""
        if self.settings.local_rank not in [-1, 0]:
            return

        # 🔥 Create checkpoint directory (unified with default path)
        checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints', self.settings.project_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 🔥 Save checkpoint (uppercase name to match default)
        checkpoint_name = f'{self.settings.script_name.upper()}_ep{self.epoch:04d}.pth.tar'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        # Prepare checkpoint data
        if hasattr(self.actor.net, 'module'):
            state_dict = self.actor.net.module.state_dict()
        else:
            state_dict = self.actor.net.state_dict()

        checkpoint_data = {
            'epoch': self.epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'settings': self.settings,
            'rng_state': self._capture_rng_states(),  # 🔥 NEW
        }

        if self.lr_scheduler is not None:
            checkpoint_data['lr_scheduler'] = self.lr_scheduler.state_dict()

        # 🔥 NEW: Save AMP scaler if used
        if hasattr(self, 'scaler') and self.use_amp:
            checkpoint_data['scaler'] = self.scaler.state_dict()

        # Save locally
        torch.save(checkpoint_data, checkpoint_path)
        self._log_assignment(f"💾 Checkpoint saved: {checkpoint_path}")

        # 🔥 Upload to HuggingFace (if configured)
        if self.hf_token and self.hf_repo:
            self._upload_to_huggingface(checkpoint_path, checkpoint_name)

    def _upload_to_huggingface(self, checkpoint_path, checkpoint_name):
        """🔥 OPTIMIZATION: Async upload to HuggingFace Hub with output redirected to log file"""
        try:
            import os
            import contextlib
            import threading
            from huggingface_hub import HfApi, create_repo

            # 🔥 Disable progress bars
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

            # 🔥 Optional: Enable faster transfer if hf_transfer is installed
            # Uncomment the next line if you install: pip install hf_transfer
            # os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

            # 🔥 OPTIMIZATION: Run upload in background thread with redirected output
            def _do_upload():
                try:
                    api = HfApi()

                    # Create repo if it doesn't exist
                    try:
                        create_repo(
                            repo_id=self.hf_repo,
                            token=self.hf_token,
                            private=False,
                            exist_ok=True,
                            repo_type="model"
                        )
                    except Exception:
                        pass  # Repo might already exist

                    # 🔥 Create log file path for HF upload output
                    log_dir = os.path.join(self.settings.save_dir, 'logs')
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, 'hf_upload.log')

                    # 🔥 Redirect stdout and stderr to log file during upload
                    with open(log_path, 'a') as logfile:
                        # Write timestamp and checkpoint info
                        from datetime import datetime
                        logfile.write(f"\n{'=' * 60}\n")
                        logfile.write(f"Upload started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        logfile.write(f"Checkpoint: {checkpoint_name}\n")
                        logfile.write(f"Size: {os.path.getsize(checkpoint_path) / (1024 ** 3):.2f} GB\n")
                        logfile.write(f"{'=' * 60}\n")
                        logfile.flush()

                        # Redirect all output to log file
                        with contextlib.redirect_stdout(logfile), \
                                contextlib.redirect_stderr(logfile):
                            # Upload file
                            api.upload_file(
                                path_or_fileobj=checkpoint_path,
                                path_in_repo=f"checkpoints/{checkpoint_name}",
                                repo_id=self.hf_repo,
                                token=self.hf_token,
                                commit_message=f"Add checkpoint {checkpoint_name}"
                            )

                        # Write completion message
                        logfile.write(f"Upload completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        logfile.write(f"{'=' * 60}\n\n")

                    # Log success to main assignment log (visible in console)
                    self._log_assignment(f"☁️  Checkpoint uploaded to HuggingFace: {self.hf_repo}")
                    self._log_assignment(f"   📝 Upload details logged to: {log_path}")

                except Exception as e:
                    # Log error to main assignment log
                    self._log_assignment(f"❌ HuggingFace upload failed: {e}")

                    # Also write error to HF log file
                    try:
                        with open(log_path, 'a') as logfile:
                            logfile.write(f"ERROR: {e}\n")
                            logfile.write(f"{'=' * 60}\n\n")
                    except:
                        pass

            # Start upload in background thread
            upload_thread = threading.Thread(target=_do_upload, daemon=True)
            upload_thread.start()
            self._log_assignment(f"📤 Background upload started for {checkpoint_name}")

        except ImportError:
            self._log_assignment("⚠️  huggingface_hub not installed. Skipping upload.")
        except Exception as e:
            self._log_assignment(f"❌ HuggingFace upload thread failed to start: {e}")

    def train_epoch(self):
        """Do one epoch for each loader."""
        # 🔥 NEW: Reset seed at the start of each epoch
        self._reseed_epoch()

        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    # 🔥 MODIFIED: Use fixed seed for strict reproducibility
                    seed = getattr(self.cfg.TRAIN, 'SEED', self.epoch) if self.cfg else self.epoch
                    loader.sampler.set_epoch(seed)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

            # 🔥 NEW: Save checkpoint after each epoch
            self._save_checkpoint_with_upload()

            # 🔥 Log epoch completion
            self._log_assignment(f"\n{'=' * 80}")
            self._log_assignment(f"✅ Epoch {self.epoch} completed")
            self._log_assignment(f"{'=' * 80}\n")

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            if misc.is_main_process():
                with open(self.settings.log_file, 'a') as f:
                    f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)