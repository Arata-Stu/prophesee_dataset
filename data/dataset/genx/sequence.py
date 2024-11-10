import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict

class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', sequence_length: int = 1,
                 guarantee_label: bool = False, padding: str = 'pad', transform=None):
        """
        Args:
            data_dir (str): データが保存されているディレクトリのパス。
            mode (str): 'train', 'val', 'test' のいずれか。
            sequence_length (int): シーケンスの長さ。
            guarantee_label (bool): True の場合、ラベルが存在するシーケンスのみを含める。
            padding (str): シーケンスの長さが足りない場合の対処法。'ignore', 'pad', 'truncate' のいずれか。
            transform (Optional[callable]): データに適用する変換。
        """
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
        """ディレクトリ内の全ての .npz ファイルのパスを取得"""
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
        """シーケンスの開始インデックスを取得"""
        indices = []
        total_files = len(self.data_files)

        for idx in range(0, total_files, self.sequence_length):
            end_idx = idx + self.sequence_length
            if end_idx > total_files:
                if self.padding == 'truncate':
                    continue
                elif self.padding == 'pad' or self.padding == 'ignore':
                    end_idx = total_files

            # デバッグ出力：ラベルの存在を確認
            if self.guarantee_label:
                has_label = any(self._has_label(i) for i in range(idx, end_idx))
                if has_label:
                    indices.append(idx)
            else:
                indices.append(idx)

        # ここでインデックスリストを返す
        return indices



    def __len__(self):
        return len(self.start_indices)  # start_indices の長さを返す

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.sequence_length

        frames = []
        labels_sequence = []
        mask = []

        for i in range(start_idx, min(end_idx, len(self.data_files))):
            data_file = self.data_files[i]
            with np.load(data_file, allow_pickle=True) as data:
                event_frame = data['event']
                labels = data['labels']
                
                frames.append(torch.from_numpy(event_frame).permute(2, 0, 1))
                labels_sequence.append(labels)
                mask.append(1)  # 有効なフレームにはマスク値 1 を設定

        # シーケンスが指定の長さに満たない場合の処理
        if len(frames) < self.sequence_length:
            if self.padding == 'pad':
                padding_length = self.sequence_length - len(frames)
                frames.extend([torch.zeros_like(frames[0])] * padding_length)
                labels_sequence.extend([[]] * padding_length)
                mask.extend([0] * padding_length)  # パディング部分にマスク値 0 を設定
            elif self.padding == 'ignore':
                pass  # データの最後でシーケンスが短くなるのを許容
            elif self.padding == 'truncate':
                return None  # この場合はシーケンスから省く

        outputs = {
            'event': torch.stack(frames),
            'labels': labels_sequence,
            'file_paths': self.data_files[start_idx:end_idx],
            'is_start_sequence': True if self.mode == 'train' else (idx == 0),
            'mask': torch.tensor(mask, dtype=torch.int64)  # マスクを追加
        }

        if self.transform is not None:
            outputs = self.transform(outputs)
        
        return outputs
