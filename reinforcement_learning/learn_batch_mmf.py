import time
import torch
import torch.nn as nn
from torch.optim import Adagrad


class LearnBatchMMF(nn.Module):
    def __init__(self, A, L, K, wavelet_indices, rest_indices, device='cpu'):
        # A (batch_size, N, N)
        # wavelet_indices (batch_size, L)
        # rest_indices (batch_size, L, K - 1)
        super(LearnBatchMMF, self).__init__()
        
        self.A = A
        self.batch_size, self.N, _ = A.size()
        self.L = L
        self.K = K
        self.device = device
        self.wavelet_indices = wavelet_indices
        self.rest_indices = rest_indices
        self.selected_indices = []

        # Initialization of the Jacobi rotation matrix
        self.all_O = torch.nn.ParameterList()
        A = self.A.detach()
        all_levels_indices = torch.cat([self.wavelet_indices.unsqueeze(-1), self.rest_indices], dim=-1)
        assert all_levels_indices.size() == (self.batch_size, self.L, self.K)

        for l in range(self.L):
            # Set the indices for this rotation
            indices, _ = torch.sort(all_levels_indices[:, l, :]) # (batch_size, k)
            index = torch.zeros((self.batch_size, self.N), device=device)
            index.scatter_(dim=-1, index=indices, src=torch.ones((self.batch_size, self.N), device=device))
            self.selected_indices.append(index)

            # Outer product map
            outer = torch.bmm(index.unsqueeze(2), index.unsqueeze(1)) # (batch_size, N, N)

            # Eigen-decomposition
            A_rows = torch.gather(input=A, dim=1, index=indices.unsqueeze(-1).expand((-1, -1, self.N)))
            A_part = torch.bmm(A_rows, torch.transpose(A_rows, 1, 2)) # (batch_size, k, k)
            _, vectors = torch.linalg.eig(A_part)

            # Rotation matrix
            O = torch.nn.Parameter(vectors.real.transpose(1, 2).data, requires_grad=True)
            self.all_O.append(O)

            # Full Jacobian rotation matrix
            U = torch.eye(self.N, device=self.device).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
            U[outer == 1] = O.flatten()
            right = U if l == 0 else torch.bmm(U, right)

            # New A
            A = torch.bmm(torch.bmm(U, A), U.transpose(1, 2))

        # Block diagonal left
        active_index = torch.ones((self.batch_size, self.N), device=device)
        active_index.scatter_(dim=-1, index=self.wavelet_indices, src=torch.zeros_like(active_index))
        left_index = torch.bmm(active_index.unsqueeze(2), active_index.unsqueeze(1)) # (batch_size, N, N)
        left_index[:, torch.arange(self.N), torch.arange(self.N)] = 1
        D = A * left_index


        # Save the remaining active indices
        self.final_active_index = active_index

        # Reconstruction
        A_rec = torch.bmm(torch.bmm(torch.transpose(right, 1, 2), D), right)

        print('Mean initialization loss:', torch.mean(torch.linalg.matrix_norm(self.A.data - A_rec)).item())

    def forward(self):
        # The current matrix
        A = self.A

        # For each resolution
        for l in range(self.L):
            # Randomization of the indices
            index = self.selected_indices[l]
            
            # Outer product map
            outer = torch.bmm(index.unsqueeze(2), index.unsqueeze(1)) # (batch_size, N, N)

            # Jacobi rotation matrix
            O = self.all_O[l]

            # Full Jacobian rotation matrix
            U = torch.eye(self.N, device=self.device).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
            U[outer == 1] = O.flatten()
            right = U if l == 0 else torch.bmm(U, right)

            # New A
            A = torch.bmm(torch.bmm(U, A), U.transpose(1, 2))

        # Diagonal left
        active_index = self.final_active_index
        left_index = torch.bmm(active_index.unsqueeze(2), active_index.unsqueeze(1)) # (batch_size, N, N)
        left_index[:, torch.arange(self.N), torch.arange(self.N)] = 1
        D = A * left_index

        # Reconstruction
        A_rec = torch.bmm(torch.bmm(torch.transpose(right, 1, 2), D), right)

        # Result
        return A_rec, right, D
    

def train_learn_batch_mmf(A, L, K, wavelet_indices, rest_indices, epochs=1000, learning_rate=1e-4, early_stop=True, logging=True):
    model = LearnBatchMMF(A, L, K, wavelet_indices, rest_indices)
    
    optimizer = Adagrad(model.parameters(), lr = learning_rate)

    # Training
    best = 1e9
    for epoch in range(epochs):
        t = time.time()
        optimizer.zero_grad()

        A_rec, right, D = model()

        loss = torch.mean(torch.linalg.matrix_norm(A - A_rec))
        loss.backward()

        if epoch % 100 == 0 and epoch > 0 and logging:
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

    per_matrix_loss = torch.linalg.matrix_norm(A - A_rec).detach()
    loss = torch.mean(per_matrix_loss).item()
    if logging:
        print('---- Final loss ----')
        print('Loss =', loss)

    return A_rec, right, D, per_matrix_loss