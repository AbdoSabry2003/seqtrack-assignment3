#!/usr/bin/env python3
"""
Compare Phase 1 and Phase 2 training logs to verify reproducibility
from Epoch 3 onwards.
"""

import re
import sys


def parse_log(path):
    """Extract epoch, samples, loss, and IoU from assignment_3_training.log"""
    pattern = re.compile(
        r'\[(train|val)\]\s+Epoch\s+(\d+)\s*:\s*(\d+)\s*/\s*(\d+)\s+samples.*?'
        r'Loss:\s+([\d.]+)(?:.*?IoU:\s+([\d.]+))?',
        re.DOTALL
    )

    results = []
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                loader = m.group(1)
                epoch = int(m.group(2))
                samples = int(m.group(3))
                total = int(m.group(4))
                loss = float(m.group(5))
                iou = float(m.group(6)) if m.group(6) else 0.0
                results.append((loader, epoch, samples, total, loss, iou))

    return results


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_phases.py <phase1_log> <phase2_log>")
        sys.exit(1)

    phase1_log = sys.argv[1]
    phase2_log = sys.argv[2]

    print("=" * 80)
    print("PHASE COMPARISON - Assignment 3")
    print("=" * 80)

    phase1 = parse_log(phase1_log)
    phase2 = parse_log(phase2_log)

    # Filter to Epoch 3-10 only
    phase1_filtered = [x for x in phase1 if x[1] >= 3 and x[1] <= 10]
    phase2_filtered = [x for x in phase2 if x[1] >= 3 and x[1] <= 10]

    print(f"\nPhase 1: {len(phase1_filtered)} entries from Epoch 3-10")
    print(f"Phase 2: {len(phase2_filtered)} entries from Epoch 3-10")

    if len(phase1_filtered) != len(phase2_filtered):
        print("\n⚠️  WARNING: Different number of entries!")
        print("Phases may not be directly comparable.")

    print("\n" + "=" * 80)
    print("DETAILED COMPARISON (Epoch 3-10)")
    print("=" * 80)
    print(f"{'#':<4} {'Loader':<6} {'Epoch':<6} {'Loss Δ':<12} {'IoU Δ':<12} {'Status':<10}")
    print("-" * 80)

    all_match = True
    tolerance = 1e-4  # Tolerance for floating point comparison

    for i, (p1, p2) in enumerate(zip(phase1_filtered, phase2_filtered), 1):
        loader1, epoch1, samples1, total1, loss1, iou1 = p1
        loader2, epoch2, samples2, total2, loss2, iou2 = p2

        # Check if entries match
        loss_diff = abs(loss1 - loss2)
        iou_diff = abs(iou1 - iou2)

        match = (
                loader1 == loader2 and
                epoch1 == epoch2 and
                samples1 == samples2 and
                loss_diff < tolerance and
                iou_diff < tolerance
        )

        status = "✅ OK" if match else "❌ DIFF"
        if not match:
            all_match = False

        print(f"{i:<4} {loader1:<6} {epoch1:<6} {loss_diff:<12.8f} {iou_diff:<12.8f} {status:<10}")

    print("=" * 80)
    if all_match:
        print("✅ SUCCESS: All epochs from 3-10 match perfectly!")
    else:
        print("❌ FAILURE: Some differences detected. Check logs above.")
    print("=" * 80)


if __name__ == "__main__":
    main()