import torch
import random
import unittest
from metaheuristics import get_cost, evolutionary_algorithm
from utils import generate_single_random_weighted_graph_laplacian

class TestGenerateSingleRandomWeightedGraphLaplacian(unittest.TestCase):
    def test_output_shape(self):
        matrix_size = 5
        edge_probability = 0.5
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        self.assertEqual(laplacian_matrix.shape, (matrix_size, matrix_size))

    def test_symmetric(self):
        matrix_size = 5
        edge_probability = 0.5
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        self.assertTrue(torch.allclose(laplacian_matrix, laplacian_matrix.t()))

    def test_matrix_content(self):
        matrix_size = 5
        edge_probability = 0.6
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        print(f'The matrix is {laplacian_matrix}, size {laplacian_matrix.size()}')


class TestMetaheuristics(unittest.TestCase):
    def test_get_cost(self):
        # Define dummy input parameters
        matrix_size = 10
        edge_probability = 0.8
        device = 'cpu'

        # Call the function
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        L = 4
        K = 3

        wavelet_indices = torch.tensor(random.sample(range(matrix_size), L))
        rest_indices = torch.zeros(L, K - 1, dtype=torch.int64)
        for j in range(L):
            values = list(range(matrix_size))
            values.remove(wavelet_indices[j].item())
            rest_indices[j] = torch.tensor(random.sample(values, K - 1))

        cost = get_cost(laplacian_matrix, wavelet_indices, rest_indices, L, K)
        print(f'The cost is {cost}')

    def test_evolutionary_algorithm(self):
        # Define dummy input parameters
        matrix_size = 10
        edge_probability = 0.8
        device = 'cpu'

        # Call the function
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        L = 4
        K = 3

        wavelet_indices = torch.tensor(random.sample(range(matrix_size), L))
        rest_indices = torch.zeros(L, K - 1, dtype=torch.int64)
        for j in range(L):
            values = list(range(matrix_size))
            values.remove(wavelet_indices[j].item())
            rest_indices[j] = torch.tensor(random.sample(values, K - 1))

        cost = get_cost(laplacian_matrix, wavelet_indices, rest_indices, L, K)
        print(f'The random cost is {cost}')

        wavelet_indices, rest_indices, ea_cost = evolutionary_algorithm(get_cost, laplacian_matrix, L, K)
        print(f'The best EA cost is {ea_cost}')
        print(f'The wavelets is {wavelet_indices}, rest_indices {rest_indices}')


if __name__ == '__main__':
    unittest.main()