import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict
import json

class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', sequence_length: int = 1,
                 guarantee_label: bool = False, padding: str = 'pad', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.sequence_length = sequence_length
        self.guarantee_label = guarantee_label
        self.padding = padding
        self.transform = transform

        # インデックスファイルからデータを読み込む
        self.index_entries = self._load_index_files()

        # ラベルが保証されている場合、ラベルが存在するエントリのみをフィルタリング
        if self.guarantee_label:
            self.index_entries = [entry for entry in self.index_entries if entry['label_file'] is not None]

        # タイムスタンプでソート
        self.index_entries.sort(key=lambda x: x['timestamp'][0])

        # シーケンスの開始インデックスを決定
        self.start_indices = self._get_start_indices()

    def _load_index_files(self):
        index_entries = []
        for root, _, files in os.walk(self.data_dir):
            if 'index.json' in files:
                index_path = os.path.join(root, 'index.json')
                with open(index_path, 'r') as f:
                    entries = json.load(f)
                    index_entries.extend(entries)
        return index_entries

    def _get_start_indices(self) -> List[int]:
        indices = []
        total_entries = len(self.index_entries)
        for idx in range(0, total_entries, self.sequence_length):
            end_idx = idx + self.sequence_length
            if end_idx > total_entries:
                if self.padding == 'truncate':
                    continue
                elif self.padding == 'pad' or self.padding == 'ignore':
                    end_idx = total_entries
            indices.append(idx)
        return indices

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.sequence_length

        frames = []
        labels_sequence = []
        mask = []
        timestamps = []

        for i in range(start_idx, min(end_idx, len(self.index_entries))):
            entry = self.index_entries[i]

            # 圧縮されたイベントフレームを読み込む
            with np.load(entry['event_file'], allow_pickle=True) as data:
                event_frame = data['events']
            frames.append(torch.from_numpy(event_frame).permute(2, 0, 1))

            # ラベルを読み込む（存在しない場合は空のリスト）
            if entry['label_file'] and os.path.exists(entry['label_file']):
                with np.load(entry['label_file'], allow_pickle=True) as data:
                    labels = data['labels']
            else:
                labels = []

            labels_sequence.append(labels)
            mask.append(1)
            timestamps.append(entry['timestamp'][0])

        # パディング処理
        if len(frames) < self.sequence_length:
            if self.padding == 'pad':
                padding_length = self.sequence_length - len(frames)
                frames.extend([torch.zeros_like(frames[0])] * padding_length)
                labels_sequence.extend([[]] * padding_length)
                mask.extend([0] * padding_length)
                timestamps.extend([0] * padding_length)
            elif self.padding == 'ignore':
                pass
            elif self.padding == 'truncate':
                return None

        outputs = {
            'events': torch.stack(frames),
            'labels': labels_sequence,
            'timestamps': torch.tensor(timestamps, dtype=torch.int64),
            'is_start_sequence': True if self.mode == 'train' else (idx == 0),
            'mask': torch.tensor(mask, dtype=torch.int64)
        }

        if self.transform is not None:
            outputs = self.transform(outputs)

        return outputs
