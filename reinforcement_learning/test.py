import unittest
import torch
from learn_batch_mmf import train_learn_batch_mmf
from learn_single_mmf import train_single_mmf
from utils import generate_random_weighted_graph_laplacian


class TestReinforcementLearning(unittest.TestCase):
    def test_generate_random_weighted_graph_laplacian(self):
        # Define dummy input parameters
        batch_size = 2
        matrix_size = 5
        edge_probability = 0.7
        weight_range = (1, 10)
        device = 'cpu'

        # Call the function
        output = generate_random_weighted_graph_laplacian(batch_size, matrix_size, edge_probability, weight_range, device)

        # Assertions
        self.assertIn('x', output)             # Check if 'x' key is present
        self.assertIn('A', output)             # Check if 'A' key is present
        self.assertIsInstance(output['x'], torch.Tensor)  # Check type of node features
        self.assertIsInstance(output['A'], torch.Tensor)  # Check type of Laplacian matrices
        self.assertEqual(output['x'].shape, (batch_size, matrix_size, 2))  # Check shape of node features
        self.assertEqual(output['A'].shape, (batch_size, matrix_size, matrix_size))  # Check shape of Laplacian matrices

    def test_train_learn_batch_mmf(self):
        # Define dummy input parameters
        batch_size = 2
        matrix_size = 10
        edge_probability = 0.2
        weight_range = (1, 10)
        device = 'cpu'

        # Call the function
        A = generate_random_weighted_graph_laplacian(batch_size, matrix_size, edge_probability, weight_range, device)['A']
        L = 4
        K = 3
        wavelet_indices = torch.tensor([[0, 1, 2, 3], [2, 1, 5, 9]], dtype=torch.int64, device=device)
        rest_indices = torch.tensor([
            [[1, 2], [2, 0], [7, 8], [5, 9]],
            [[1, 6], [2, 0], [7, 8], [5, 3]]
            ], dtype=torch.int64, device=device)
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
        

if __name__ == '__main__':
    unittest.main()