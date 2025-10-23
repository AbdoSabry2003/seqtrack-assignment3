#!/usr/bin/env python3
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "output/logs/assignment_3_training.log"

with open(log_file, 'r') as f:
    content = f.read()

# Split by "ASSIGNMENT 3 - SeqTrack Training Started"
parts = content.split("=" * 80 + "\nASSIGNMENT 3 - SeqTrack Training Started")

if len(parts) < 3:
    print("Could not split log properly!")
    sys.exit(1)

# Phase 1: first session (index 1)
phase1 = "=" * 80 + "\nASSIGNMENT 3 - SeqTrack Training Started" + parts[1]

# Phase 2: second session (index 2)
phase2 = "=" * 80 + "\nASSIGNMENT 3 - SeqTrack Training Started" + parts[2]

with open("output/logs/assignment_3_training_phase1.log", 'w') as f:
    f.write(phase1)

with open("output/logs/assignment_3_training_phase2.log", 'w') as f:
    f.write(phase2)

print("✅ Split complete:")
print("   - output/logs/assignment_3_training_phase1.log")
print("   - output/logs/assignment_3_training_phase2.log")
