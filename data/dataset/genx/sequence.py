import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict

class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', sequence_length: int = 1,
                 guarantee_label: bool = False, padding: str = 'pad', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.sequence_length = sequence_length
        self.guarantee_label = guarantee_label
        self.padding = padding
        self.transform = transform

        # 全ての .npz ファイルのパスを取得し、タイムスタンプでソート
        self.data_files = self._get_all_data_files()
        self.data_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

        # シーケンスの開始インデックスを決定
        self.start_indices = self._get_start_indices()

    def _get_all_data_files(self) -> List[str]:
        data_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.npz'):
                    data_files.append(os.path.join(root, file))
        return data_files

    def _has_label(self, idx: int) -> bool:
        data_file = self.data_files[idx]
        with np.load(data_file, allow_pickle=True) as data:
            if 'labels' not in data:
                return False
            labels = data['labels']
        return len(labels) > 0

    def _get_start_indices(self) -> List[int]:
        indices = []
        total_files = len(self.data_files)
        for idx in range(0, total_files, self.sequence_length):
            end_idx = idx + self.sequence_length
            if end_idx > total_files:
                if self.padding == 'truncate':
                    continue
                elif self.padding == 'pad' or self.padding == 'ignore':
                    end_idx = total_files
            if self.guarantee_label:
                has_label = any(self._has_label(i) for i in range(idx, end_idx))
                if has_label:
                    indices.append(idx)
            else:
                indices.append(idx)
        return indices

    def _extract_timestamp(self, file_path: str) -> int:
        """ファイル名から開始のタイムスタンプを抽出"""
        file_name = os.path.basename(file_path)
        start_time_str = file_name.split('_')[0]
        return int(start_time_str)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.sequence_length

        frames = []
        labels_sequence = []
        mask = []
        timestamps = []

        for i in range(start_idx, min(end_idx, len(self.data_files))):
            data_file = self.data_files[i]
            with np.load(data_file, allow_pickle=True) as data:
                event_frame = data['event']
                labels = data['labels']
                
                frames.append(torch.from_numpy(event_frame).permute(2, 0, 1))
                labels_sequence.append(labels)
                mask.append(1)
                timestamps.append(self._extract_timestamp(data_file))  # タイムスタンプを取得してリストに追加

        if len(frames) < self.sequence_length:
            if self.padding == 'pad':
                padding_length = self.sequence_length - len(frames)
                frames.extend([torch.zeros_like(frames[0])] * padding_length)
                labels_sequence.extend([[]] * padding_length)
                mask.extend([0] * padding_length)
                timestamps.extend([0] * padding_length)  # パディング部分に0のタイムスタンプを追加
            elif self.padding == 'ignore':
                pass
            elif self.padding == 'truncate':
                return None

        outputs = {
            'events': torch.stack(frames),
            'labels': labels_sequence,
            'file_paths': self.data_files[start_idx:end_idx],
            'timestamps': torch.tensor(timestamps, dtype=torch.int64),  # タイムスタンプを追加
            'is_start_sequence': True if self.mode == 'train' else (idx == 0),
            'mask': torch.tensor(mask, dtype=torch.int64)
        }

        if self.transform is not None:
            outputs = self.transform(outputs)

        return outputs
