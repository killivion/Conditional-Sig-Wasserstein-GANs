import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict


class LoadData:
    def __init__(self, data_type: str, params: Dict[str, Union[float, int]], seed: int = None):
        self.data_type_functions = {
            "Blackscholes": self.generate_multiple_gbm
        }
        self.data_type = data_type
        self.params = params
        self.seed = seed

    def create_dataset(self, output_type: str):
        """Create specified dataset."""
        if self.data_type in self.data_type_functions:
            if self.seed is not None:
                np.random.seed(self.seed)
            paths, time = self.data_type_functions[self.data_type](**self.params)#
            # Transforms data - row: time step, column: path
            if output_type == "np.ndarray":
                return paths, time
            elif output_type == "DataFrame":
                return pd.DataFrame(paths, columns=time)
            else:
                raise ValueError(f'output_type={output_type} not implemented.')
        else:
            data_type_list = "', '".join(self.data_type_functions.keys())
            raise ValueError(
                f'Dataset "{self.data_type}" type currently not implemented. ' +
                f'Choose from "{data_type_list}".'
            )


    def generate_multiple_gbm(self, S0, mu, sigma, T, num_steps, num_paths):
        #Generates paths of Black-Scholes

        dt = T / num_steps  # step size
        t = np.linspace(0, T, num_steps + 1)  # time array

        dW = np.random.normal(size=(num_paths, num_steps))
        W = np.cumsum(np.sqrt(dt) * dW, axis=1)
        W = np.hstack([np.zeros((num_paths, 1)), W])

        S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

        return S, t


if __name__ == "__main__": #Testing
    import matplotlib.pyplot as plt

    GBM_parameter = {
        "S0": 1.,  # initial stock price
        "mu": 0.05,  # expected return (drift)
        "sigma": 0.2,  # volatility
        "T": 1,  # time horizon (in years)
        "num_steps": 252, #number of steps
        "num_paths": 30 #number of paths
    }

    gbm = LoadData(data_type="Blackscholes", params=GBM_parameter)
    prices_df = gbm.create_dataset("DataFrame")

    # Assuming df_times_as_index is the DataFrame with times as row indices
    prices_df.T.plot(figsize=(15, 9), alpha=1)
    #prices_df.T.plot(figsize=(15, 9), linewidth=0.1, alpha=0.5, color='blue') # many paths
    #prices_df.T.plot(figsize=(15, 9), linewidth=0.1, alpha=0.2, color='blue') # lot of paths 1000


    plt.title('Simulated Paths over Time')
    plt.xlabel('Timeframe (years)')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
