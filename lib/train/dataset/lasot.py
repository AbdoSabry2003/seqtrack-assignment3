import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Lasot(BaseVideoDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None,
                 selected_classes=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            selected_classes - List of class names to use (e.g., ['mouse', 'electricfan']). If None, all classes are used.
        """
        root = env_settings().lasot_dir if root is None else root
        super().__init__('LaSOT', root, image_loader)

        # 🔥 MODIFIED: Filter class_list BEFORE building sequence_list
        all_classes = [f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]

        if selected_classes is not None:
            self.class_list = [c for c in all_classes if c in selected_classes]
            print(f"📌 Using selected classes: {self.class_list}")
        else:
            self.class_list = all_classes

        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_per_class = self._build_class_list()

        # 🔥 NEW: Print statistics for selected classes
        if selected_classes is not None:
            print(f"📊 Dataset statistics:")
            for class_name in self.class_list:
                num_videos = len(self.seq_per_class.get(class_name, []))
                print(f"   - {class_name}: {num_videos} videos")
            print(f"   Total sequences: {len(self.sequence_list)}")

    def _read_ids_from_file(self, file_path):
        """Read video IDs from training_set.txt or testing_set.txt"""
        ids = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.isdigit():
                        ids.append(int(ln))
                    else:
                        try:
                            ids.append(int(ln.split('-')[-1]))
                        except:
                            pass
        return ids

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split not in ['train', 'test']:
            raise ValueError('split must be "train" or "test".')

        # Store split for later use
        self.split = split

        # Case 1: Use class-specific files when selected_classes is provided
        if hasattr(self, 'class_list') and len(self.class_list) > 0:
            sequence_list = []
            for cls in self.class_list:
                cls_dir = os.path.join(self.root, cls)

                # Try different naming conventions
                if split == 'train':
                    candidates = ['training_set.txt', 'trainingset.txt']
                else:
                    candidates = ['testing_set.txt', 'testingset.txt']

                split_file = None
                for fn in candidates:
                    fp = os.path.join(cls_dir, fn)
                    if os.path.exists(fp):
                        split_file = fp
                        break

                if split_file is None:
                    # Fallback: If no split files exist, use old method
                    print(f"⚠️  No {split} split file found for {cls}, using fallback method")
                    ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
                    if split == 'train':
                        file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
                    else:
                        file_path = os.path.join(ltr_path, 'data_specs', 'lasot_test_split.txt')

                    if os.path.exists(file_path):
                        all_sequences = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
                        cls_sequences = [seq for seq in all_sequences if seq.startswith(cls + '-')]
                        sequence_list.extend(cls_sequences)
                else:
                    # Read IDs from split file
                    ids = self._read_ids_from_file(split_file)
                    for vid in ids:
                        sequence_list.append(f"{cls}-{vid}")

            print(f"📌 LaSOT {split}: {len(sequence_list)} sequences from {len(self.class_list)} classes")
            return sequence_list

        # Case 2: Fallback to old method (no selected_classes)
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        if split == 'train':
            file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
        else:
            # Create test split file path (even if it doesn't exist in original)
            file_path = os.path.join(ltr_path, 'data_specs', 'lasot_test_split.txt')

        if os.path.exists(file_path):
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        else:
            sequence_list = []

        # Filter by class_list if exists
        if hasattr(self, 'class_list'):
            sequence_list = [seq for seq in sequence_list if seq.split('-')[0] in self.class_list]

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'lasot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        """🔥 OPTIMIZATION: More robust reading of occlusion/out_of_view files"""

        def _read_flags_file(file_path):
            """Helper function to read flag files more robustly"""
            flags = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        # Remove whitespace and commas
                        line = line.strip().replace(',', ' ')
                        if not line:
                            continue
                        # Split by any whitespace and process each token
                        for token in line.split():
                            try:
                                flags.append(int(token))
                            except ValueError:
                                # Skip non-integer tokens
                                continue
            return torch.ByteTensor(flags if flags else [0])

        # Read full occlusion and out_of_view files
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        try:
            # Try the more robust method first
            occlusion = _read_flags_file(occlusion_file)
            out_of_view = _read_flags_file(out_of_view_file)
        except Exception as e:
            # Fallback to original method if something goes wrong
            print(f"⚠️ Warning: Using fallback method for reading visibility files: {e}")
            try:
                with open(occlusion_file, 'r', newline='') as f:
                    occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
                with open(out_of_view_file, 'r') as f:
                    out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
            except Exception as e2:
                print(f"❌ Error reading visibility files, using defaults: {e2}")
                # Return all visible as fallback
                bbox = self._read_bb_anno(seq_path)
                return torch.ones(len(bbox), dtype=torch.uint8)

        target_visible = ~occlusion & ~out_of_view
        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        # 🔥 IMPROVED: Use os.path for cross-platform compatibility
        return os.path.basename(os.path.dirname(seq_path))

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames