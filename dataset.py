from torch.utils.data import Dataset


class RibFracDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError
