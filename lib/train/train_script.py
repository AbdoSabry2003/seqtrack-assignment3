# lib/train/train_script.py
import os
import random
import numpy as np
import torch

# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.seqtrack import build_seqtrack
# forward propagation related
from lib.train.actors import SeqTrackActor
# for import modules
import importlib


def init_seeds(seed, local_rank=-1):
    """Initialize random seeds for reproducibility"""
    # 🔥 للـ distributed training: كل GPU ياخد seed مختلف شوية
    if local_rank != -1:
        actual_seed = seed + local_rank
    else:
        actual_seed = seed

    # Set environment variables for determinism
    import os
    os.environ["PYTHONHASHSEED"] = str(actual_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    print(f"🌱 Random seed set to: {actual_seed} (base seed: {seed}, local_rank: {local_rank})")


def run(settings):
    settings.description = 'Training script for SeqTrack'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg  # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file)  # update cfg from experiments

    # 🔥 SEED INITIALIZATION - المكان الوحيد!
    seed = getattr(cfg.TRAIN, 'SEED', None)
    if seed is not None:
        init_seeds(seed, settings.local_rank)
    else:
        print("⚠️  No seed specified in config. Training will be non-deterministic.")

    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

        # Print selected classes if available
        selected_classes = getattr(cfg.DATA.TRAIN, 'SELECTED_CLASSES', None)
        if selected_classes is not None:
            print(f"📌 Selected Classes: {selected_classes}")
            print(f"📊 Using {len(selected_classes)} classes for training\n")

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)

        # 🔥 NEW: Build validation loader
        selected_classes = getattr(cfg.DATA.TRAIN, 'SELECTED_CLASSES', None)
        loader_val = build_val_loader(cfg, settings, selected_classes)

        if settings.local_rank in [-1, 0]:
            print(f"✅ Train loader: {cfg.DATA.TRAIN.SAMPLE_PER_EPOCH} samples/epoch")
            val_samples = len(loader_val) * cfg.TRAIN.BATCH_SIZE
            print(f"✅ Val loader: {val_samples} samples/epoch")
            print(f"📊 Classes: {selected_classes if selected_classes else 'All'}\n")
    else:
        raise ValueError("illegal DATA LOADER")

    # Create network
    if settings.script_name == "seqtrack":
        net = build_seqtrack(cfg)  # pix2seq method with multi-frames and encoder mask
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # Loss functions and Actors
    if settings.script_name == "seqtrack":
        bins = cfg.MODEL.BINS
        weight = torch.ones(bins + 2)
        weight[bins] = 0.01
        weight[bins + 1] = 0.01
        objective = {'ce': CrossEntropyLoss(weight=weight)}
        loss_weight = {'ce': cfg.TRAIN.CE_WEIGHT}
        actor = SeqTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)

    # 🔥 MODIFIED: Pass both train and validation loaders to trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp, cfg=cfg)

    # 🔥 NEW: Resume from checkpoint if specified
    resume_path = getattr(cfg.TRAIN, 'RESUME_PATH', None)
    if resume_path and os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)
        print(f"✅ Resumed training from: {resume_path}")
    elif resume_path:
        print(f"⚠️  Resume path specified but file not found: {resume_path}")

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True)  # 🔥 Changed load_latest to False