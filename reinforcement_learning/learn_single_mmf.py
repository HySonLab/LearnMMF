import time
import torch
import torch.nn as nn
from torch.optim import Adagrad


class LearnableSingleMMF(nn.Module):
    def __init__(self, A, L, K, wavelet_indices, rest_indices, device = 'cpu'):
        super(LearnableSingleMMF, self).__init__()
        
        # Matrix
        self.A = A

        # Size of the matrix
        self.N = A.size(0)
        assert A.dim() == 2
        assert A.size(1) == self.N

        # Number of resolutions
        self.L = L

        # Size of the Jacobian rotation matrix
        self.K = K

        # Device
        self.device = device

        # Given indices
        self.wavelet_indices = wavelet_indices
        self.rest_indices = rest_indices

        active_index = torch.ones(self.N)
        self.selected_indices = []

        # Initialization of the Jacobi rotation matrix
        self.all_O = torch.nn.ParameterList()
        
        # The current matrix
        A = torch.Tensor(self.A.data)

        for l in range(self.L):
            # Set the indices for this rotation
            indices = self.wavelet_indices[l] + self.rest_indices[l]
            indices.sort()
            assert len(indices) == self.K
            index = torch.zeros(self.N)
            for k in range(self.K):
                index[indices[k]] = 1
            self.selected_indices.append(index)

            # Outer product map
            outer = torch.outer(index, index)

            # Eigen-decomposition
            A_part = torch.matmul(A[index == 1], torch.transpose(A[index == 1], 0, 1))
            values, vectors = torch.linalg.eig(torch.reshape(A_part, (self.K, self.K)))

            # Rotation matrix
            O = torch.nn.Parameter(vectors.real.transpose(0, 1).data, requires_grad = True)
            self.all_O.append(O)

            # Full Jacobian rotation matrix
            U = torch.eye(self.N).to(device = self.device)
            U[outer == 1] = O.flatten()

            if l == 0:
                right = U
            else:
                right = torch.matmul(U, right)

            # New A
            A = torch.matmul(torch.matmul(U, A), U.transpose(0, 1))

            # Drop the wavelet
            active_index[self.wavelet_indices[l]] = 0

        self.final_active_index = active_index

        # Block diagonal left
        left_index = torch.outer(active_index, active_index).to(device = self.device)
        left_index = torch.eye(self.N).to(device = self.device) - torch.diag(torch.diag(left_index)) + left_index
        D = A * left_index

        # Reconstruction
        A_rec = torch.matmul(torch.matmul(torch.transpose(right, 0, 1), D), right)

        print('Initialization loss:', torch.norm(self.A.data - A_rec, p = 'fro'))

    def forward(self):
        # The current matrix
        A = self.A

        # For each resolution
        for l in range(self.L):
            # Randomization of the indices
            index = self.selected_indices[l]
            
            # Outer product map
            outer = torch.outer(index, index)

            # Jacobi rotation matrix
            O = self.all_O[l].to(device = self.device)

            # Full Jacobian rotation matrix
            U = torch.eye(self.N).to(device = self.device)
            U[outer == 1] = O.flatten()

            if l == 0:
                right = U
            else:
                right = torch.matmul(U, right)

            # New A
            A = torch.matmul(torch.matmul(U, A), U.transpose(0, 1))

        # Diagonal left
        active_index = self.final_active_index.to(device = self.device)
        left_index = torch.outer(active_index, active_index)
        left_index = torch.eye(self.N).to(device = self.device) - torch.diag(torch.diag(left_index)) + left_index
        D = A * left_index

        # Reconstruction
        A_rec = torch.matmul(torch.matmul(torch.transpose(right, 0, 1), D), right)

        # Result
        return A_rec, right, D


def train_single_mmf(A, L, K, wavelet_indices, rest_indices, epochs = 10000, learning_rate = 1e-4, early_stop = True):
    model = LearnableSingleMMF(A, L, K, wavelet_indices, rest_indices)

    optimizer = Adagrad(model.parameters(), lr = learning_rate)
    # Training
    best = 1e9
    for epoch in range(epochs):
        t = time.time()
        optimizer.zero_grad()

        A_rec, right, D = model()

        loss = torch.norm(A - A_rec, p = 'fro')
        loss.backward()

        if epoch % 100 == 0 and epoch > 0:
            print('---- Epoch', epoch, '----')
            print('Loss =', loss.item())
            print('Time =', time.time() - t)

        if loss.item() < best:
            best = loss.item()
        else:
            if early_stop:
                print('Early stop at epoch', epoch)
                break

        optimizer.step()

    # Return the result
    A_rec, right, D = model()

    loss = torch.norm(A - A_rec, p = 'fro')
    print('---- Final loss ----')
    print('Loss =', loss.item())

    return A_rec, right, D, loss