import torch
import random
import unittest
from model import GNN
from utils import get_cost
from learn_single_mmf import train_single_mmf
from learn_batch_mmf import train_learn_batch_mmf
from utils import generate_random_weighted_graph_laplacian


class TestReinforcementLearning(unittest.TestCase):
    def test_generate_random_weighted_graph_laplacian(self):
        # Define dummy input parameters
        batch_size = 2
        matrix_size = 5
        edge_probability = 0.7
        device = 'cpu'

        # Call the function
        output = generate_random_weighted_graph_laplacian(batch_size, matrix_size, edge_probability, device)

        # Assertions
        self.assertIn('x', output)             # Check if 'x' key is present
        self.assertIn('A', output)             # Check if 'A' key is present
        self.assertIsInstance(output['x'], torch.Tensor)  # Check type of node features
        self.assertIsInstance(output['A'], torch.Tensor)  # Check type of Laplacian matrices
        self.assertEqual(output['x'].shape, (batch_size, matrix_size, 2))  # Check shape of node features
        self.assertEqual(output['A'].shape, (batch_size, matrix_size, matrix_size))  # Check shape of Laplacian matrices

        x, A = output['x'], output['A']
        print(f'x = {x}')
        print(f'A = {A}')

    def test_train_learn_batch_mmf(self):
        # Define dummy input parameters
        batch_size = 10
        matrix_size = 10
        edge_probability = 0.8
        device = 'cpu'

        # Call the function
        A = generate_random_weighted_graph_laplacian(batch_size, matrix_size, edge_probability, device)['A']
        L = 4
        K = 3

        wavelet_indices = torch.zeros(batch_size, L, dtype=torch.int64)
        for i in range(batch_size):
            wavelet_indices[i] = torch.tensor(random.sample(range(matrix_size), L))

        rest_indices = torch.zeros(batch_size, L, K - 1, dtype=torch.int64)
        for i in range(batch_size):
            for j in range(L):
                values = list(range(matrix_size))
                values.remove(wavelet_indices[i][j].item())
                rest_indices[i][j] = torch.tensor(random.sample(values, K - 1))

        epochs = 100
        learning_rate = 1e-3
        early_stop = False

        # Calculate the cost for the matrices one at a time
        single_loss = []
        for i in range(batch_size):
            loss = train_single_mmf(A[i, :, :], L, K, wavelet_indices[i, :].unsqueeze(-1).tolist(), rest_indices[i, :, :].tolist(), epochs, learning_rate, early_stop)[3]
            single_loss.append(loss.item())

        # Calculate the cost for the whole batch
        A_rec, right, D, batch_loss = train_learn_batch_mmf(A, L, K, wavelet_indices, rest_indices, epochs, learning_rate, early_stop)
        
        print(f'The per matrix loss of single MMF is {single_loss}')
        print(f'The per matrix loss of batch MMF is {batch_loss.tolist()}')


class TestGNN(unittest.TestCase):
    def setUp(self):
        self.input_dim = 2
        self.hidden_dim = 16
        self.n_layers = 3
        self.L = 5
        self.k = 4
        self.device = torch.device('cpu')  # or specify 'cuda' if you have GPU support

    def test_gnn_forward(self):
        model = GNN(self.input_dim, self.hidden_dim, self.n_layers, self.L, self.k).to(self.device)

        # Dummy input data
        batch_size = 2
        matrix_size = 10
        edge_probability = 0.7
        output = generate_random_weighted_graph_laplacian(batch_size, matrix_size, edge_probability, self.device)
        x = output['x']
        A = output['A']

        # Forward pass
        agg_wavelet_indices, agg_rest_indices, sum_log_prob = model(x, A)

        # Assertions
        self.assertEqual(agg_wavelet_indices.shape, (batch_size, self.L))  # Check shape of agg_wavelet_indices
        self.assertEqual(agg_rest_indices.shape, (batch_size, self.L, self.k - 1))  # Check shape of agg_rest_indices
        print(f'Wavelets are {agg_wavelet_indices.size()} \n {agg_wavelet_indices}')
        print(f'Rests are {agg_rest_indices.size()} \n {agg_rest_indices}')
        print(f'Log probability are {sum_log_prob.size()} \n {sum_log_prob}')

        cost = get_cost(A, self.L, self.k, agg_wavelet_indices, agg_rest_indices)
        print(f'The cost is {cost}')

if __name__ == '__main__':
    unittest.main()