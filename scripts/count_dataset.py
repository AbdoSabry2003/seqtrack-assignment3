# count_dataset.py
import os
import sys

sys.path.insert(0, '.')

from lib.train.dataset.lasot import Lasot
from lib.train.admin import env_settings
from lib.train.data import opencv_loader

selected_classes = ['mouse', 'electricfan']

# Train dataset
dataset_train = Lasot(
    split='train',
    selected_classes=selected_classes,
    image_loader=opencv_loader
)

# Test dataset
dataset_test = Lasot(
    split='test',
    selected_classes=selected_classes,
    image_loader=opencv_loader
)

print("=" * 80)
print("DATASET STATISTICS FOR ASSIGNMENT 3")
print("=" * 80)
print(f"\nSelected Classes: {selected_classes}")
print(f"\n{'=' * 80}")
print("TRAIN SPLIT")
print("=" * 80)

total_train_frames = 0
for cls in selected_classes:
    sequences = dataset_train.seq_per_class.get(cls, [])
    print(f"\n{cls}:")
    print(f"  Number of sequences: {len(sequences)}")

    # Count total frames
    frames = 0
    seq_names = []
    for seq_id in sequences:
        seq_info = dataset_train.get_sequence_info(seq_id)
        frames += len(seq_info['bbox'])
        seq_names.append(dataset_train.sequence_list[seq_id])

    total_train_frames += frames
    print(f"  Total frames: {frames}")
    print(f"  Sequences: {', '.join(seq_names)}")

print(f"\nTotal train sequences: {dataset_train.get_num_sequences()}")
print(f"Total train frames: {total_train_frames}")

print(f"\n{'=' * 80}")
print("TEST SPLIT")
print("=" * 80)

total_test_frames = 0
for cls in selected_classes:
    sequences = dataset_test.seq_per_class.get(cls, [])
    print(f"\n{cls}:")
    print(f"  Number of sequences: {len(sequences)}")

    # Count total frames
    frames = 0
    seq_names = []
    for seq_id in sequences:
        seq_info = dataset_test.get_sequence_info(seq_id)
        frames += len(seq_info['bbox'])
        seq_names.append(dataset_test.sequence_list[seq_id])

    total_test_frames += frames
    print(f"  Total frames: {frames}")
    print(f"  Sequences: {', '.join(seq_names)}")

print(f"\nTotal test sequences: {dataset_test.get_num_sequences()}")
print(f"Total test frames: {total_test_frames}")

print(f"\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)
print(f"Samples per epoch (train): {total_train_frames} frames / 3 = {total_train_frames // 3} samples")
print(f"Expected SAMPLE_PER_EPOCH in YAML: {total_train_frames // 3}")
print("=" * 80)