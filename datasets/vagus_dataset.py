import numpy as np
import torch


class VagusDataset(torch.utils.data.Dataset):
    def __init__(self, train) -> None:
        self.train : bool = train
        self.data = self._load_data()


    def __getitem__(self, idx: int):
        return self.data[idx]

    
    def __len__(self) -> int:
        return len(self.data)

    
    def _load_data(self):
        data_file = f"vagus_{'train' if self.train else 'test'}.npy"
        data = np.load(f"./data/Metcalfe-2014/{data_file}")

        return data
