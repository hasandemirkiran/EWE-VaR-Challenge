import unittest
import numpy as np
import pandas as pd
import challenge
from scipy.stats import norm
from unittest.mock import patch, mock_open


class TestVaRCalculation(unittest.TestCase):

    def setUp(self):
        self.params = {
            'Bremen': {'A': 2.8, 'B': -37.0, 'C': 6.25, 'D': 0.06, 'K': 25000},
            'Oldenburg': {'A': 3.5, 'B': -38.2, 'C': 4.00, 'D': 0.17, 'K': 18000}
        }
        self.temperature_expectation = {
            'Bremen': [8.3, 12.9, 15.9],
            'Oldenburg': [8.1, 12.5, 15.6]
        }

    def test_offtake(self):
        self.assertAlmostEqual(
            challenge.offtake(10, 25000, 2.8, -37.0, 6.25, 0.06), 16365.256977069748, places=1)
        self.assertAlmostEqual(
            challenge.offtake(0, 25000, 2.8, -37.0, 6.25, 0.06), 44862.2462031523, places=1)

    @patch('pandas.read_csv')
    def test_read_data(self, mock_read_csv):
        mock_data = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=28),
            'Bremen_Apr': np.random.rand(28),
            'Bremen_Mai': np.random.rand(28),
            'Bremen_Jun': np.random.rand(28),
            'Oldenburg_Apr': np.random.rand(28),
            'Oldenburg_Mai': np.random.rand(28),
            'Oldenburg_Jun': np.random.rand(28),
            'TTF_Apr': np.random.rand(28),
            'TTF_Mai': np.random.rand(28),
            'TTF_Jun': np.random.rand(28),
            'GPL_Apr': np.random.rand(28),
            'GPL_Mai': np.random.rand(28),
            'GPL_Jun': np.random.rand(28)
        })
        mock_read_csv.return_value = mock_data

        data = challenge.read_data('dummy_file.csv')
        self.assertEqual(data.shape, (28, 13))
        self.assertTrue(isinstance(data['datetime'].iloc[0], pd.Timestamp))

    def test_extract_temperatures(self):
        data = pd.DataFrame({
            'Bremen_Apr': np.random.rand(28),
            'Bremen_Mai': np.random.rand(28),
            'Bremen_Jun': np.random.rand(28),
        })
        bremen_temps = challenge.extract_temperatures(data, 'Bremen')
        self.assertEqual(bremen_temps.shape, (28, 3))
        self.assertTrue(np.array_equal(
            bremen_temps[:, 0], data['Bremen_Apr'].values))

    def test_extract_prices(self):
        data = pd.DataFrame({
            'TTF_Apr': np.random.rand(28),
            'TTF_Mai': np.random.rand(28),
            'TTF_Jun': np.random.rand(28),
        })
        ttf_prices = challenge.extract_prices(data, 'TTF')
        self.assertEqual(ttf_prices.shape, (28, 3))
        self.assertTrue(np.array_equal(
            ttf_prices[:, 0], data['TTF_Apr'].values))

    def test_calculate_initial_volumes(self):
        initial_volumes = challenge.calculate_initial_volumes(
            self.temperature_expectation, self.params)
        self.assertEqual(len(initial_volumes['Bremen']), 3)
        self.assertEqual(len(initial_volumes['Oldenburg']), 3)
        self.assertGreater(initial_volumes['Bremen'][0], 0)
        self.assertGreater(initial_volumes['Oldenburg'][0], 0)

    def test_calculate_forecasted_volumes(self):
        forecasted_temps = {
            'Bremen': np.array([[8, 12, 16], [9, 13, 17]]),
            'Oldenburg': np.array([[7, 11, 15], [8, 12, 16]])
        }
        forecasted_volumes = challenge.calculate_forecasted_volumes(
            forecasted_temps, self.params)
        self.assertEqual(forecasted_volumes['Bremen'].shape, (2, 3))
        self.assertEqual(forecasted_volumes['Oldenburg'].shape, (2, 3))
        self.assertGreater(np.min(forecasted_volumes['Bremen']), 0)
        self.assertGreater(np.min(forecasted_volumes['Oldenburg']), 0)

    def test_calculate_delta_volumes(self):
        initial_volumes = {
            'Bremen': [1000000, 1100000, 1200000],
            'Oldenburg': [800000, 850000, 900000]
        }
        forecasted_volumes = {
            'Bremen': np.array([[1010000, 1110000, 1210000], [1020000, 1120000, 1220000]]),
            'Oldenburg': np.array([[810000, 860000, 910000], [820000, 870000, 920000]])
        }
        delta_volumes = challenge.calculate_delta_volumes(
            forecasted_volumes, initial_volumes)

        # Check the shapes
        self.assertEqual(delta_volumes['Bremen'].shape, (2, 3))
        self.assertEqual(delta_volumes['Oldenburg'].shape, (2, 3))

        # Check the values
        expected_delta_bremen = np.array(
            [[10000, 10000, 10000], [20000, 20000, 20000]])
        expected_delta_oldenburg = np.array(
            [[10000, 10000, 10000], [20000, 20000, 20000]])

        self.assertTrue(
            np.all(delta_volumes['Bremen'] == expected_delta_bremen))
        self.assertTrue(
            np.all(delta_volumes['Oldenburg'] == expected_delta_oldenburg))

    def test_calculate_portfolio_value_changes(self):
        delta_volumes = {
            'Bremen': np.array([[10000, 10000, 10000], [20000, 20000, 20000]]),
            'Oldenburg': np.array([[5000, 5000, 5000], [6000, 6000, 6000]])
        }
        spot_prices = {
            'TTF': np.array([[20, 21, 22], [23, 24, 25]]),
            'GPL': np.array([[15, 16, 17], [18, 19, 20]])
        }
        initial_prices = {
            'TTF': [20, 21, 22],
            'GPL': [15, 16, 17]
        }
        portfolio_value_changes = challenge.calculate_portfolio_value_changes(
            delta_volumes, spot_prices, initial_prices)
        self.assertEqual(portfolio_value_changes['Bremen'].shape, (2, 3))
        self.assertEqual(portfolio_value_changes['Oldenburg'].shape, (2, 3))
        self.assertTrue(np.all(portfolio_value_changes['Bremen'][0] == 0))
        self.assertTrue(np.all(portfolio_value_changes['Oldenburg'][0] == 0))

    def test_calculate_cov_matrix(self):
        temperature_changes = {
            'Bremen': np.array([[1, 1, 1], [2, 2, 2]]),
            'Oldenburg': np.array([[1, 1, 1], [2, 2, 2]])
        }
        cov_matrix = challenge.calculate_cov_matrix(temperature_changes)

        # Check the shape of the covariance matrix
        self.assertEqual(cov_matrix.shape, (6, 6))

        # Check that the covariance matrix is symmetric
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))

        # Check that the diagonal elements are non-negative
        self.assertTrue(np.all(np.diag(cov_matrix) >= 0))

        # Check that the covariance matrix is positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_matrix)

        # Allow for small numerical errors
        self.assertTrue(np.all(eigenvalues >= -1e-10))

        # Check that covariances between identical columns are equal to their variances
        for i in range(3):
            self.assertAlmostEqual(cov_matrix[i, i], cov_matrix[i+3, i+3])
            self.assertAlmostEqual(cov_matrix[i, i+3], cov_matrix[i, i])

    def test_calculate_VaR(self):
        portfolio_delta = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]])
        cov_matrix = np.eye(6)
        confidence_level = 0.95
        holding_period = 3

        VaR = challenge.calculate_VaR(
            portfolio_delta, cov_matrix, confidence_level, holding_period)

        portfolio_variance = np.sum([
            np.dot(delta.T, np.dot(cov_matrix, delta)) for delta in portfolio_delta
        ])
        portfolio_std_dev = np.sqrt(
            portfolio_variance / portfolio_delta.shape[0])
        expected_VaR = norm.ppf(confidence_level) * \
            portfolio_std_dev * np.sqrt(holding_period)

        self.assertAlmostEqual(VaR, expected_VaR, places=2)

    def test_calculate_VaR_edge_cases(self):
        # Test with zero portfolio delta
        portfolio_delta_zero = np.zeros((2, 6))
        cov_matrix = np.eye(6)
        self.assertEqual(challenge.calculate_VaR(
            portfolio_delta_zero, cov_matrix, 0.95, 3), 0)

        # Test with non-zero portfolio delta
        portfolio_delta_nonzero = np.array(
            [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]])

        # Test that VaR increases with confidence level
        var_95 = challenge.calculate_VaR(
            portfolio_delta_nonzero, cov_matrix, 0.95, 3)
        var_99 = challenge.calculate_VaR(
            portfolio_delta_nonzero, cov_matrix, 0.99, 3)
        self.assertGreater(var_99, var_95)

        # Test that VaR scales with sqrt of holding period
        var_1day = challenge.calculate_VaR(
            portfolio_delta_nonzero, cov_matrix, 0.95, 1)
        var_4days = challenge.calculate_VaR(
            portfolio_delta_nonzero, cov_matrix, 0.95, 4)
        self.assertAlmostEqual(var_4days, var_1day * 2, places=5)

        # Test that VaR is non-negative for valid inputs
        self.assertGreaterEqual(challenge.calculate_VaR(
            portfolio_delta_nonzero, cov_matrix, 0.5, 1), 0)

    @patch('challenge.read_data')
    @patch('challenge.calculate_VaR')
    def test_main(self, mock_calculate_VaR, mock_read_data):
        mock_data = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=28),
            'Bremen_Apr': np.random.rand(28),
            'Bremen_Mai': np.random.rand(28),
            'Bremen_Jun': np.random.rand(28),
            'Oldenburg_Apr': np.random.rand(28),
            'Oldenburg_Mai': np.random.rand(28),
            'Oldenburg_Jun': np.random.rand(28),
            'TTF_Apr': np.random.rand(28),
            'TTF_Mai': np.random.rand(28),
            'TTF_Jun': np.random.rand(28),
            'GPL_Apr': np.random.rand(28),
            'GPL_Mai': np.random.rand(28),
            'GPL_Jun': np.random.rand(28)
        })
        mock_read_data.return_value = mock_data
        mock_calculate_VaR.return_value = 1000000

        challenge.main()

        mock_read_data.assert_called_once()
        mock_calculate_VaR.assert_called_once()


if __name__ == "__main__":
    unittest.main()
