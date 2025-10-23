#!/usr/bin/env python3
"""
Plot training curves (Loss and IoU) from assignment_3_training.log
"""

import re
import sys
import matplotlib.pyplot as plt
import os


def parse_log(path):
    """Extract epoch-level metrics from log"""
    pattern = re.compile(
        r'\[(train|val)\]\s+Epoch\s+(\d+)\s*:.*?'
        r'Loss:\s+([\d.]+)(?:.*?IoU:\s+([\d.]+))?',
        re.DOTALL
    )

    train_data = {}
    val_data = {}

    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                loader = m.group(1)
                epoch = int(m.group(2))
                loss = float(m.group(3))
                iou = float(m.group(4)) if m.group(4) else 0.0

                if loader == 'train':
                    train_data[epoch] = {'loss': loss, 'iou': iou}
                else:
                    val_data[epoch] = {'loss': loss, 'iou': iou}

    return train_data, val_data


def plot_curves(train_data, val_data, output_prefix, title_suffix=""):
    """Plot Loss and IoU curves"""

    # Sort by epoch
    train_epochs = sorted(train_data.keys())
    val_epochs = sorted(val_data.keys())

    train_losses = [train_data[e]['loss'] for e in train_epochs]
    train_ious = [train_data[e]['iou'] for e in train_epochs]

    val_losses = [val_data[e]['loss'] for e in val_epochs] if val_data else []
    val_ious = [val_data[e]['iou'] for e in val_epochs] if val_data else []

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    if val_losses:
        plt.plot(val_epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_loss.png', dpi=150)
    print(f"✅ Saved: {output_prefix}_loss.png")
    plt.close()

    # Plot IoU
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_ious, 'b-o', label='Train IoU', linewidth=2)
    if val_ious:
        plt.plot(val_epochs, val_ious, 'r-s', label='Val IoU', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.title(f'Training IoU{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_iou.png', dpi=150)
    print(f"✅ Saved: {output_prefix}_iou.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_curves.py <log_file> [output_prefix] [title_suffix]")
        sys.exit(1)

    log_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "training_curves"
    title_suffix = sys.argv[3] if len(sys.argv) > 3 else ""

    if title_suffix:
        title_suffix = f" - {title_suffix}"

    print("=" * 80)
    print("PLOTTING TRAINING CURVES")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print(f"Output prefix: {output_prefix}")
    print()

    train_data, val_data = parse_log(log_file)

    print(f"Parsed {len(train_data)} train epochs")
    print(f"Parsed {len(val_data)} val epochs")
    print()

    plot_curves(train_data, val_data, output_prefix, title_suffix)

    print("=" * 80)
    print("✅ DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()