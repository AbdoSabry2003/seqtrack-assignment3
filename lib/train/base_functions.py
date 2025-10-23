import torch
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, Imagenet1k
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
from lib.train.data import jpeg4py_loader  # 🔥 OPTIMIZATION: Import jpeg4py_loader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def seed_worker(worker_id):
    """Set seed for DataLoader workers for reproducibility"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': getattr(cfg.DATA.TEMPLATE, "FACTOR", None),
                                   'search': getattr(cfg.DATA.SEARCH, "FACTOR", None)}
    settings.output_sz = {'template': getattr(cfg.DATA.TEMPLATE, "SIZE", 128),
                          'search': getattr(cfg.DATA.SEARCH, "SIZE", 256)}
    settings.center_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "CENTER_JITTER", None),
                                     'search': getattr(cfg.DATA.SEARCH, "CENTER_JITTER", None)}
    settings.scale_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "SCALE_JITTER", None),
                                    'search': getattr(cfg.DATA.SEARCH, "SCALE_JITTER", None)}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader, selected_classes=None, split='train'):
    """
    🔥 MODIFIED: Added split parameter for train/test separation
    """
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID",
                        "TRACKINGNET", "IMAGENET1K"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split=split, image_loader=image_loader,
                                           selected_classes=selected_classes))
            else:
                # 🔥 Pass split parameter to Lasot
                datasets.append(Lasot(settings.env.lasot_dir, split=split, image_loader=image_loader,
                                      selected_classes=selected_classes))
                if selected_classes:
                    print(f"📌 LaSOT ({split}) filtered to classes: {selected_classes}")
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(
                    Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "IMAGENET1K":
            datasets.append(Imagenet1k(settings.env.imagenet1k_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)

    # 🔥 NEW: Extract selected classes from config
    selected_classes = getattr(cfg.DATA.TRAIN, "SELECTED_CLASSES", None)
    if selected_classes:
        print(f"\n{'=' * 60}")
        print(f"📌 ASSIGNMENT 3: Using selected classes only")
        print(f"   Classes: {selected_classes}")
        print(f"{'=' * 60}\n")

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SeqTrackProcessing(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          settings=settings)

    # Train sampler and loader
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    print("sampler_mode", sampler_mode)

    # 🔥 OPTIMIZATION: Use jpeg4py_loader instead of opencv_loader for faster JPEG loading
    dataset_train = sampler.TrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, jpeg4py_loader,  # 🔥 CHANGED
                                selected_classes=selected_classes, split='train'),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_train,
        frame_sample_mode=sampler_mode
    )

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    # 🔥 NEW: Add seed generator for reproducibility
    seed = getattr(cfg.TRAIN, 'SEED', 0)
    g = torch.Generator()
    g.manual_seed(seed)

    # 🔥 OPTIMIZATION: Enhanced DataLoader configuration
    loader_train = LTRLoader(
        'train', dataset_train, training=True,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,
        stack_dim=1,
        sampler=train_sampler,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,  # 🔥 OPTIMIZATION: Faster GPU transfer
        persistent_workers=(cfg.TRAIN.NUM_WORKER > 0),  # 🔥 OPTIMIZATION: Avoid worker restart
        prefetch_factor=4 if cfg.TRAIN.NUM_WORKER > 0 else None  # 🔥 OPTIMIZATION: More prefetching
    )

    return loader_train


def build_val_loader(cfg, settings, selected_classes=None):
    """Build validation loader with same classes but no augmentation"""

    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)

    # 🔥 Validation transforms - no augmentation
    transform_val = tfm.Transform(
        tfm.ToTensorAndJitter(0.0),  # no jitter
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
    )

    # The tracking pairs processing module for validation
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_val = processing.SeqTrackProcessing(
        search_area_factor=search_area_factor,
        output_sz=output_sz,
        center_jitter_factor={'template': 0.0, 'search': 0.0},  # no jitter
        scale_jitter_factor={'template': 0.0, 'search': 0.0},  # no jitter
        mode='sequence',
        transform=transform_val,
        joint_transform=None,  # no joint transform
        settings=settings
    )

    # Validation sampler and loader
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")

    # 🔥 OPTIMIZATION: Reduced validation samples and use jpeg4py_loader
    dataset_val = sampler.TrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, jpeg4py_loader,  # 🔥 CHANGED
                                selected_classes=selected_classes, split='test'),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=50,  # 🔥 OPTIMIZATION: Reduced from 200 to 50
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_val,
        frame_sample_mode=sampler_mode
    )

    val_sampler = DistributedSampler(dataset_val, shuffle=False) if settings.local_rank != -1 else None

    val_every = getattr(cfg.TRAIN, 'VAL_EVERY', 1)

    # 🔥 NEW: Add seed generator for reproducibility
    seed = getattr(cfg.TRAIN, 'SEED', 0)
    g = torch.Generator()
    g.manual_seed(seed)

    # 🔥 OPTIMIZATION: Enhanced validation DataLoader with epoch_interval
    loader_val = LTRLoader(
        'val', dataset_val, training=False,  # training=False for validation
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,  # no shuffle for validation
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=False,
        stack_dim=1,
        sampler=val_sampler,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,  # 🔥 OPTIMIZATION: Faster GPU transfer
        persistent_workers=(max(1, cfg.TRAIN.NUM_WORKER // 2) > 0),  # 🔥 OPTIMIZATION
        prefetch_factor=2 if cfg.TRAIN.NUM_WORKER > 1 else None,  # 🔥 OPTIMIZATION
        epoch_interval=val_every  # 🔥 Validate every epoch
    )

    print(f"📊 Validation loader created with {50} samples per run, every {val_every} epoch(s)")

    return loader_val


def get_optimizer_scheduler(net, cfg):
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "encoder" in n and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
        },
    ]
    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print(n)

    # 🔥 OPTIMIZATION: Try to use foreach for faster optimizer updates
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        # استخدم AdamW القياسي بدون foreach لضمان الـ determinism
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        if is_main_process():
            print("✅ Using standard AdamW for maximum determinism")
    else:
        raise ValueError("Unsupported Optimizer")

    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler