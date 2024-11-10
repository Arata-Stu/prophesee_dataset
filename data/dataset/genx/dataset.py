import os
from torch.utils.data import ConcatDataset
from .sequence import SequenceDataset  # SequenceDataset クラスをインポート

class PropheseeConcatDataset(ConcatDataset):
    def __init__(self, base_data_dir: str, mode: str, tau: int, delta_t: int, 
                 sequence_length: int = 1, guarantee_label: bool = False, transform=None):
        """
        Args:
            base_data_dir (str): ベースのデータディレクトリのパス。
            mode (str): 'train', 'val', 'test' のいずれか。
            tau (int): タウの値（例: 50）。
            delta_t (int): デルタtの値（例: 10 または 50）。
            sequence_length (int): シーケンスの長さ。
            guarantee_label (bool): True の場合、ラベルが存在するシーケンスのみを含める。
            transform (callable, optional): データに適用する変換関数。
        """
        self.base_data_dir = base_data_dir
        self.sequence_length = sequence_length
        self.guarantee_label = guarantee_label
        self.mode = mode
        self.tau = tau
        self.delta_t = delta_t

        # 指定されたモードのディレクトリから全シーケンスディレクトリを取得
        mode_dir = os.path.join(self.base_data_dir, self.mode)
        if not os.path.isdir(mode_dir):
            raise ValueError(f"The directory for mode '{self.mode}' does not exist in {self.base_data_dir}")
        
        # tau と delta_t に対応するサブディレクトリを含む SequenceDataset を作成し、リストに追加
        datasets = []
        for sequence in os.listdir(mode_dir):
            sequence_path = os.path.join(mode_dir, sequence)
            if os.path.isdir(sequence_path):
                # tau と delta_t に一致するディレクトリパスを生成
                tau_delta_dir = f"tau={self.tau}_dt={self.delta_t}"
                full_path = os.path.join(sequence_path, tau_delta_dir)
                
                if os.path.isdir(full_path):
                    datasets.append(
                        SequenceDataset(
                            data_dir=full_path,
                            mode=self.mode,
                            sequence_length=self.sequence_length,
                            guarantee_label=self.guarantee_label,
                            transform=transform,
                        )
                    )
        
        # ConcatDataset の初期化を利用して複数のデータセットを結合
        super().__init__(datasets)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        """
        結合されたデータセットから指定されたインデックスのアイテムを取得します。
        ConcatDataset では index は結合後のインデックスであるため、
        個々のデータセットにインデックスが属しているかを確認し、該当するデータセットからアイテムを取得します。
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError("index out of range")
