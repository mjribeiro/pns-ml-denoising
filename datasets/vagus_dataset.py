import torch


class VagusDataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
