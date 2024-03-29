import numpy as np
import torch


# self.data is supposed to be [window, num_channels, samples, type of recording], so 
# last index is 0 if ENG recording or 1 if blood pressure recording

class VagusDataset(torch.utils.data.Dataset):
    def __init__(self, train) -> None:
        self.train : bool = train
        self.data = self._load_data()


    def __getitem__(self, idx: int):
        # data_torch = torch.from_numpy(self.data[idx])
        # return data_torch.unsqueeze(0)
        return torch.from_numpy(self.data[idx, :, :, 0])

    
    def __len__(self) -> int:
        return len(self.data)

    
    def _load_data(self):
        data_file = f"vagus_{'train' if self.train else 'test'}_filt_narrow_1024.npy"
        data = np.load(f"./data/Metcalfe-2014/{data_file}")

        return data


    def load_bp_data(self, idx_start, idx_end):
        return self.data[idx_start:idx_end, :, :, 1]



class VagusDatasetN2N(torch.utils.data.Dataset):
    def __init__(self, stage) -> None:
        self.stage : str = stage
        self.data = self._load_data()
        self.targets = self._load_targets()


    def __getitem__(self, idx: int):
        return torch.from_numpy(self.data[idx, :, :, 0]), torch.from_numpy(self.targets[idx, :, :, 0]), self.data[idx, :, :, 1]

    
    def __len__(self) -> int:
        return len(self.data)

    
    def _load_data(self):
        data_file = f"n2n_X_{self.stage}.npy"
        data = np.load(f"./data/Metcalfe-2014/{data_file}")

        return data


    def _load_targets(self):
        targets_file = f"n2n_y_{self.stage}.npy"
        targets = np.load(f"./data/Metcalfe-2014/{targets_file}")
        
        return targets

    def load_bp_data(self, idx_start, idx_end):
        return self.data[idx_start:idx_end, :, :, 1]
