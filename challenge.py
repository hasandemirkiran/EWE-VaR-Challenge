import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
from typing import Dict, List, Tuple
import yaml

# Set up logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_file: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def offtake(T: float, K: float, A: float, B: float, C: float, D: float) -> float:
    """Calculate gas offtake based on temperature and other parameters."""
    return K * (A / (1 + (B / (T - 40)) ** C) + D)


def read_data(filename: str) -> pd.DataFrame:
    """Read and preprocess data from a CSV file."""
    try:
        data = pd.read_csv(filename, sep=';', decimal=',')
        data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')
        return data
    except Exception as e:
        logging.error(f"Error reading data from {filename}: {e}")
        raise


def extract_temperatures(data: pd.DataFrame, city: str) -> np.ndarray:
    """Extract temperature data for a specific city."""
    return data[[f'{city}_Apr', f'{city}_Mai', f'{city}_Jun']].values


def extract_prices(data: pd.DataFrame, market: str) -> np.ndarray:
    """Extract price data for a specific market."""
    return data[[f'{market}_Apr', f'{market}_Mai', f'{market}_Jun']].values


def calculate_initial_volumes(temperature_expectation: Dict[str, List[float]], params: Dict[str, Dict[str, float]]) -> Dict[str, List[float]]:
    """Calculate initial volumes based on temperature expectations."""
    return {
        city: [offtake(temp, **params[city])
               for temp in temperature_expectation[city]]
        for city in temperature_expectation
    }


def calculate_forecasted_volumes(forecasted_temps: Dict[str, np.ndarray], params: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Calculate forecasted volumes based on temperature forecasts."""
    return {
        city: np.array([offtake(temp, **params[city])
                       for temp in city_temps.flatten()]).reshape(city_temps.shape)
        for city, city_temps in forecasted_temps.items()
    }


def calculate_delta_volumes(forecasted_volumes: Dict[str, np.ndarray], initial_volumes: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    """Calculate the difference between forecasted and initial volumes."""
    return {
        city: forecasted_volumes[city] - np.array(initial_volumes[city])
        for city in forecasted_volumes
    }


def calculate_portfolio_value_changes(delta_volumes: Dict[str, np.ndarray], spot_prices: Dict[str, np.ndarray], initial_prices: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    """Calculate changes in portfolio value."""
    return {
        'Bremen': delta_volumes['Bremen'] * (spot_prices['GPL'] - initial_prices['GPL']),
        'Oldenburg': delta_volumes['Oldenburg'] * (spot_prices['TTF'] - initial_prices['TTF'])
    }


def calculate_cov_matrix(temperature_changes: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate the covariance matrix of temperature changes."""
    temperature_changes_matrix = np.concatenate(
        list(temperature_changes.values()), axis=1)
    return np.cov(temperature_changes_matrix, rowvar=False)


def calculate_VaR(portfolio_delta: np.ndarray, cov_matrix: np.ndarray, confidence_level: float, holding_period: int) -> float:
    """Calculate Value at Risk."""
    portfolio_variance = np.sum(
        [np.dot(delta.T, np.dot(cov_matrix, delta)) for delta in portfolio_delta])
    portfolio_std_dev = np.sqrt(portfolio_variance / portfolio_delta.shape[0])
    return norm.ppf(confidence_level) * portfolio_std_dev * np.sqrt(holding_period)


def main():
    try:
        config = load_config('config.yaml')
        data = read_data(config['data_file'])

        params = config['params']
        temperature_expectation = config['temperature_expectation']
        initial_prices = config['initial_prices']
        confidence_level = config['confidence_level']
        holding_period = config['holding_period']

        initial_volumes = calculate_initial_volumes(
            temperature_expectation, params)
        forecasted_temps = {city: extract_temperatures(
            data, city) for city in params.keys()}

        forecasted_volumes = calculate_forecasted_volumes(
            forecasted_temps, params)
        delta_volumes = calculate_delta_volumes(
            forecasted_volumes, initial_volumes)

        spot_prices = {market: extract_prices(
            data, market) for market in ['TTF', 'GPL']}

        portfolio_value_changes = calculate_portfolio_value_changes(
            delta_volumes, spot_prices, initial_prices)

        daily_temp_changes = {city: np.diff(
            temps, axis=0) for city, temps in forecasted_temps.items()}
        cov_matrix = calculate_cov_matrix(daily_temp_changes)

        portfolio_delta = np.concatenate(list(delta_volumes.values()), axis=1)

        VaR = calculate_VaR(portfolio_delta, cov_matrix,
                            confidence_level, holding_period)

        # logging.info(f"VaR ({holding_period} days, {confidence_level:.0%}): {VaR:.2f}")
        print(f"VaR ({holding_period} days, {confidence_level:.0%}): {VaR:.2f}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
