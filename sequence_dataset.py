from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np

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

class SequenceDatasetNPY(Dataset):
    def __init__(self, sequences, labels, lengths, max_len = 150):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # print(f'{idx=}')
        # return idx
        # print(f'{idx=}')
        # print(f'{self.sequences[idx].shape=}')
        # print(f'{self.sequences[self.lengths[idx].item()+3, :]=}')
        return self.pad_sequence(self.sequences[idx]), self.labels[idx], self.lengths[idx]

    def pad_sequence(self, sequence_name):
        sequence = np.load(sequence_name)
        if sequence.shape[0] < self.max_len:
            zeros = np.zeros((self.max_len, sequence.shape[1]), dtype=np.float32)
            zeros[:sequence.shape[0], :] = sequence
            return zeros
        else:
            return sequence