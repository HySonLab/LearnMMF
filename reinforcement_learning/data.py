import torch
from torch.utils.data import IterableDataset, Dataset
from utils import generate_random_weighted_graph_laplacian


class MMFTrainsSet(IterableDataset):
    """
    Generate Laplacian instances and node features on the fly
    """

    def __init__(self, n_batch_per_epoch, batch_size, matrix_size, device='cpu'):
        super().__init__()
        self.n_batch_per_epoch = n_batch_per_epoch
        self.batch_size = batch_size
        self.matrix_size = matrix_size
        self.device = device

    def __generator(self):
        for _ in range(self.n_batch_per_epoch):
            yield generate_random_weighted_graph_laplacian(self.batch_size, self.matrix_size, torch.rand(1).item(), self.device)

    def __iter__(self):
        return iter(self.__generator())

    def __len__(self):
        return self.n_batch_per_epoch
    

class MMFTestSet(Dataset):
    def __init__(self, batch_size, matrix_size, device):
        super().__init__()
        self.data = generate_random_weighted_graph_laplacian(batch_size, matrix_size, 0.6, device)

    def __len__(self):
        return self.data['x'].size()[0]

    def __getitem__(self, idx):
        return {
            'x': self.data['x'][idx],
            'A': self.data['A'][idx],
        }
