# SeqTrack — Assignment 3 (Team 13)

This repo contains my Assignment 3 work on SeqTrack (CVPR'23) with a reproducible setup on LaSOT using two classes.

- Paper: SeqTrack: Sequence to Sequence Learning for Visual Object Tracking
- Environment: WSL Ubuntu + PyTorch 1.11.0 + CUDA 11.3

## Highlights
- Deterministic training (seed=13) re-applied at the start of each epoch
- Two LaSOT classes: mouse, electricfan (16 train + 4 test per class)
- Full checkpoints saved per epoch: model + optimizer + LR scheduler + RNG states (+ AMP scaler if enabled)
- Detailed logging every 50 samples with time stats (console + file)
- Resume support from any checkpoint (verified: Phase 2 matches Phase 1 for epochs 3–10)

## Reproducibility
- Seed is reset each epoch (Python/NumPy/Torch/torch.cuda + cuDNN deterministic flags)
- DataLoader generator and worker_init_fn are seeded
- Checkpoints include RNG states; resume loads optimizer/scheduler/scaler/RNG and continues seamlessly

## Datasets
LaSOT only. Folder set via:
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

## Quickstart
```
# Phase 1 (from scratch)
python tracking/train.py --script seqtrack --config seqtrack_b256 --save_dir ./output --mode single

# Phase 2 (resume from epoch 2)
# Set in YAML: TRAIN.RESUME_PATH: ./output/checkpoints/train/seqtrack/seqtrack_b256/SEQTRACK_ep0002.pth.tar
python tracking/train.py --script seqtrack --config seqtrack_b256 --save_dir ./output --mode single
```

## Checkpoints (Hugging Face)
All 10 checkpoints are hosted on the Hub:
- https://huggingface.co/AbdoSabry2003/seqtrack-assignment3
- Files: checkpoints/SEQTRACK_ep0001.pth.tar ... SEQTRACK_ep0010.pth.tar

Download example:
```python
from huggingface_hub import hf_hub_download
pth = hf_hub_download(
    repo_id="AbdoSabry2003/seqtrack-assignment3",
    filename="checkpoints/SEQTRACK_ep0005.pth.tar"
)
print("downloaded:", pth)
```

## Scripts
- scripts/count_dataset.py — counts frames/sequences and computes samples/epoch
- scripts/compare_phases.py — verifies that Phase 2 matches Phase 1 (epochs 3–10)
- scripts/plot_training_curves.py — generates Loss/IoU curves
- scripts/split_log.py — splits combined log into phase1/phase2

## Notes
- For the attached runs we used SAMPLE_PER_EPOCH=100 for time constraints
- The correct value for two classes is 21,960 samples/epoch (65,880 frames / 3)

## Links
- GitHub: https://github.com/AbdoSabry2003/seqtrack-assignment3
- Hugging Face: https://huggingface.co/AbdoSabry2003/seqtrack-assignment3

---
library_name: pytorch
pipeline_tag: object-tracking
tags:
  - seqtrack
  - lasot
  - tracking
  - reproducibility
datasets:
  - lasot
license: mit
---

# SeqTrack — Assignment 3 (Team 13)

This model card hosts 10 training checkpoints from my Assignment 3 on SeqTrack trained on a two-class subset of LaSOT (mouse, electricfan). Training is deterministic with full-state checkpoints that enable perfect resume.

## Contents
- checkpoints/SEQTRACK_ep0001.pth.tar
- ...
- checkpoints/SEQTRACK_ep0010.pth.tar

## Training Setup
- Classes: mouse, electricfan (16 train + 4 test per class)
- Seed: 13 (reset at the start of every epoch)
- Checkpoints contain:
  - model state_dict
  - optimizer state_dict
  - LR scheduler state_dict
  - RNG states (Python/NumPy/Torch/torch.cuda)
  - (optional) AMP scaler

## How to use
```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("AbdoSabry2003/seqtrack-assignment3", "checkpoints/SEQTRACK_ep0008.pth.tar")
```

## Notes
- For time constraints, training runs used SAMPLE_PER_EPOCH=100 (the correct count is 21,960 for these two classes).
- Verified reproducibility: resuming from epoch 2 yields identical Loss/IoU from epoch 3–10 compared to a straight-through run.
- Repository: https://github.com/AbdoSabry2003/seqtrack-assignment3

Data (LaSOT) — not included in this repo
We do NOT ship datasets in this repository to keep it lightweight. Please prepare LaSOT locally before training.

Option A) Manual download and placement
- Download LaSOT (train/test) and extract under:
  - WSL path example: /home/<user>/datasets/lasot
  - Windows drive via WSL: /mnt/d/datasets/lasot
- Expected folder structure:
  SeqTrack/
    data/
      lasot/
        mouse/
          mouse-1/
            img/00000001.jpg
            groundtruth.txt
            full_occlusion.txt
            out_of_view.txt
          ...
        electricfan/
          electricfan-1/
          ...
- Note: Each class has 20 sequences total (16 train + 4 test by the official split).

Option B) Download via Hugging Face Hub
If you have a dataset repo on the Hub (recommended for teams):
- Install: pip install huggingface-hub
- Example code:
  python
  from huggingface_hub import snapshot_download
  # Replace with your dataset repo_id
  repo_id = "YOUR_USERNAME/lasot-2classes"  # repo_type="dataset"
  snapshot_download(
      repo_id=repo_id,
      repo_type="dataset",
      local_dir="data/lasot",
      local_dir_use_symlinks=False,  # safer on Windows/WSL
  )
- After this, you should have data/lasot/<class>/<class-id>/...

Set project paths (one-time)
- Generate local path config:
  bash
  python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
- Then adjust the LaSOT root if needed:
  - Edit lib/train/admin/local.py and set:
    python
    self.lasot_dir = "/home/<user>/datasets/lasot"  # or /mnt/d/datasets/lasot
- Alternatively, keep the dataset inside ./data/lasot and you’re done.

Verify the split and counts
- We provide a helper script:
  bash
  python scripts/count_dataset.py
- You should see (for our config):
  - mouse: 16 train + 4 test sequences
  - electricfan: 16 train + 4 test sequences
  - Total train frames: 65,880 → samples_per_epoch = 65,880 / 3 = 21,960
- Note: In our test runs we used SAMPLE_PER_EPOCH=100 for time constraints; the correct value for full two-class training is 21,960.
