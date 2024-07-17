from torch.utils.data import DataLoader, TensorDataset, Dataset

# Create a custom Dataset class
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # print(f'{idx=}')
        # return idx
        # print(f'{idx=}')
        # print(f'{self.sequences[idx].shape=}')
        # print(f'{self.sequences[self.lengths[idx].item()+3, :]=}')
        return idx, self.sequences[idx], self.labels[idx], self.lengths[idx]