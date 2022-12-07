import numpy as np
import torch


class VagusDataset(torch.utils.data.Dataset):
    def __init__(self, train) -> None:
        self.train : bool = train
        self.data = self._load_data()


    def __getitem__(self, idx: int):
        # data_torch = torch.from_numpy(self.data[idx])
        # return data_torch.unsqueeze(0)
        return torch.from_numpy(self.data[idx])

    
    def __len__(self) -> int:
        return len(self.data)

    
    def _load_data(self):
        data_file = f"vagus_{'train' if self.train else 'test'}_bp_narrow_1024.npy"
        data = np.load(f"./data/Metcalfe-2014/{data_file}")

        return data


class VagusDatasetN2N(torch.utils.data.Dataset):
    def __init__(self, train) -> None:
        self.train : bool = train
        self.data = self._load_data()
        self.targets = self._load_targets()


    def __getitem__(self, idx: int):
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.targets[idx])

    
    def __len__(self) -> int:
        return len(self.data)

    
    def _load_data(self):
        data_file = f"vagus_{'train' if self.train else 'test'}_raw_1024.npy"
        data = np.load(f"./data/Metcalfe-2014/{data_file}")

        return data


    def _load_targets(self):
        targets_file = f"vagus_{'train' if self.train else 'test'}_bp_narrow_1024.npy"
        targets = np.load(f"./data/Metcalfe-2014/{targets_file}")
        
        return targets
