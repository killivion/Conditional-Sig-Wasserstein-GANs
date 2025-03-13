import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
import yfinance as yf
from tqdm import tqdm
from IPython.display import clear_output


class LoadData:
    def __init__(self, dataset: str, isSigLib: bool, data_params: Dict[str, Union[float, int]], seed: int = None):
        self.dataset_functions = {
            "Blackscholes": self.generate_gbm,
            "correlated_Blackscholes": self.generate_correlated_BS,
            "Heston": self.generate_heston,
            "VarianceGamma": self.generate_vargamma,
            "Kou_Jump_Diffusion": self.generate_kou,
            "Levy_Ito": self.generate_levyito,
            "YFinance": self.get_yfinance_data
        }
        self.dataset = dataset
        self.data_params = data_params
        self.isSigLib = isSigLib
        self.seed = seed

    def create_dataset(self, output_type: str):
        """Create specified dataset."""
        if self.dataset in self.dataset_functions:
            if self.seed is not None:
                np.random.seed(self.seed)
            paths, time = self.dataset_functions[self.dataset](**self.data_params)
            if output_type == "DataFrame" or self.isSigLib == False:  # for testing and TD3
                return pd.DataFrame(paths, columns=time)
            elif output_type == "np.ndarray":  # for GANs
                return paths, time
            else:
                raise ValueError(f'output_type={output_type} not implemented.')
        else:
            dataset_list = "', '".join(self.dataset_functions.keys())
            raise ValueError(
                f'Dataset "{self.dataset}" type currently not implemented. ' +
                f'Choose from "{dataset_list}".'
            )

    """Return of dataset_functions: S: NumPy array, shape (num_paths, window_size); t: A NumPy array of time steps (first point 0)"""

    def generate_gbm(self, mu, sigma, window_size, num_paths, grid_points=252, S0=1):
        """Generates num_paths of Black-Scholes
        mu: Drift, sigma: Volatility"""

        T = window_size/grid_points
        dt = 1/grid_points
        t = np.linspace(0, T, window_size + 1)

        # Wiener Processes
        dW = np.random.normal(0, np.sqrt(dt), size=(num_paths, window_size))
        W = np.cumsum(dW, axis=1)
        W = np.hstack([np.zeros((num_paths, 1)), W])

        # Asset price process from closed form
        S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

        return S, t

    def generate_correlated_BS(self, mu, vola_matrix, window_size, num_paths, num_bm, grid_points=252, S0=1):
        """Generates num_paths of correlated Black-Scholes, from num_bm Brownian Motions
        mu: Drift, vola_matrix: Volatility [corrospondence of asset to Brownian Motion]"""

        T = window_size / grid_points
        dt = 1 / grid_points
        t = np.linspace(0, T, window_size + 1)

        S = np.zeros((num_paths, window_size + 1))
        S[:, 0] = S0

        for i in range(1, window_size+1):
            W = np.random.normal(size=num_bm)
            S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * np.diag(vola_matrix @ vola_matrix.T)) * dt + vola_matrix @ W * np.sqrt(dt))

        #prices_df = pd.DataFrame(S, columns=t)
        #print('Mean and Std of correlated BS %s %s' % (prices_df.iloc[:, -1].mean()**(1 / T) - 1, prices_df.iloc[:, -1].std() / np.sqrt(T)))

        return S, t

    def generate_heston(self, lambda_0, v0_sqrt, kappa, sigma, xi, rho, window_size, num_paths, grid_points=252, S0=1):
        """generates num_paths time series following the Heston model via CIR
        mu: Drift, V0_squared->V0: Initial Variance, kappa: Mean reversion rate of Variance, sigma->theta: long Variance, xi: Volatility of Volatility, rho: Correlation of Wiener"""
        # Feller-Condition: 2*kappa*theta > xi**2

        r = 0.04
        T = window_size / grid_points
        dt = 1 / grid_points
        v0 = v0_sqrt ** 2
        theta = sigma ** 2

        S = np.zeros((num_paths, window_size + 1))
        S[:, 0] = S0
        v_t = np.full(num_paths, v0)

        sqrt_dt = np.sqrt(dt)
        sqrt_cor = np.sqrt(1 - rho ** 2)

        for i in tqdm(range(1, window_size + 1), desc="Heston", leave=False):
            # Generate correlated Wiener Processes
            W1 = np.random.normal(size=num_paths)
            W2 = rho * W1 + sqrt_cor * np.random.normal(size=num_paths)

            # Euler-Maruyama method
            # Variance following the CIR-model
            v_t += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * W2 * sqrt_dt
            v_t = np.maximum(v_t, 0)  # v_t positive [not necessary if Feller Condition is met]

            # Asset price process
            S[:, i] = S[:, i - 1] * np.exp((r + lambda_0 * v_t - 0.5 * v_t) * dt + np.sqrt(v_t) * W1 * sqrt_dt)

        t = np.linspace(0, T, window_size + 1)

        return S, t

    def generate_vargamma(self, mu, sigma, nu, window_size, num_paths, grid_points=252, S0=1):
        """generates num_paths time series following Variance Gamma (VG)
        mu: Drift, sigma: Variance, nu: Jump intensity of the gamma process """

        T = window_size / grid_points
        dt = 1 / grid_points
        t = np.linspace(0, T, window_size + 1)

        # Generate gamma distributed increments for subordinator
        gamma_increments = np.random.gamma(shape=dt / nu, scale=nu, size=(num_paths, window_size))

        # Wiener processes
        dW = np.random.normal(size=(num_paths, window_size))

        # Variance Gamma process paths:
        X = np.cumsum(mu * gamma_increments + sigma * np.sqrt(gamma_increments) * dW, axis=1)
        X = np.hstack([np.zeros((num_paths, 1)), X])  # Initial value

        # Asset price process
        S = S0 * np.exp(X)

        return S, t

    def generate_kou(self, mu, sigma, kou_lambda, p, eta1, eta2, window_size, num_paths, grid_points=252, S0=1):
        """Copied from Peter Reute: generates num_paths time series following Kou-Jump-Diffusion
        mu: Drift, sigma: Volatility, lambda_: Jump intensity, p: Probability of positive jump, eta1: Rate of positive jump's exponential distribution, eta2: Rate of negative jump's exponential distribution"""

        T = window_size / grid_points
        dt = 1 / grid_points

        gbm, t = self.generate_gbm(mu=mu, sigma=sigma, window_size=window_size, num_paths=num_paths, grid_points=grid_points, S0=S0)  # Geometric Brownian Motion
        dv = np.ones((num_paths, len(t)))  # Jump component
        dN = np.random.poisson(kou_lambda * dt, size=(num_paths, len(t) - 1))  # increments of Poisson process

        for i in tqdm(range(num_paths), desc="Kou", leave=False):
            for j in range(1, len(t)):
                # iterate through number of jumps
                # vi contains all jumps that happen in one increment
                vi = np.ones(dN[i, j - 1])
                for k in range(dN[i, j - 1]):
                    # loop over all jumps in one time step
                    gamma = np.random.exponential(1 / eta1) if np.random.rand() < p else -np.random.exponential(
                        1 / eta2)
                    vi[k] = np.exp(gamma)
                # accumulate all jumps which happen in one timestep (1 if empty)
                dv[i, j] = np.prod(vi)
        # Calculate the paths with jumps
        S = gbm * np.cumprod(dv, axis=1)
        return S, t

    def generate_levyito(self, mu, sigma, lambda_large, lambda_small, jump_mean_large, jump_std_large, jump_mean_small, jump_std_small, window_size, num_paths, grid_points=252, S0=1):
        """ generates num_paths time series following Levy Process - by Levy-Ito Decomposition: Xt = sigma*Wt + mu*t + Yt + Zt
        Parameters: sigma: Volatility of the Brownian component, a: Drift term, lambda_large: Rate of large jumps (for Y_t), lambda_small: Rate of small jumps (for Z_t), jump_mean_large, jump_std_large: Mean and standard deviation of large jumps, jump_mean_small, jump_std_small: Mean and standard deviation of small jumps"""

        T = window_size / grid_points
        dt = 1 / grid_points
        t = np.linspace(0, T, window_size + 1)

        # Brownian motion component
        dB = np.random.normal(scale=np.sqrt(dt), size=(num_paths, window_size))
        B = np.cumsum(sigma * dB, axis=1)
        B = np.hstack([np.zeros((num_paths, 1)), B])

        # Large jumps (compound Poisson process, Y_t) and Small jumps (compensated Poisson process, Z_t)
        Y = np.zeros((num_paths, window_size + 1))
        Z = np.zeros((num_paths, window_size + 1))
        for i in tqdm(range(num_paths), desc="LevyIto", leave=False):
            for step in range(1, window_size + 1):
                # Generate small/large jumps based on a Poisson process
                num_large_jumps = np.random.poisson(lambda_large * dt)
                if num_large_jumps > 0:
                    jumps = np.random.normal(jump_mean_large, jump_std_large, size=num_large_jumps)
                    Y[i, step] = np.sum(jumps)
                num_small_jumps = np.random.poisson(lambda_small * dt)
                if num_small_jumps > 0:
                    jumps = np.random.normal(jump_mean_small, jump_std_small, size=num_small_jumps)
                    Z[i, step] = np.sum(jumps)
        Y = np.cumsum(Y, axis=1)  # Cumulative sum to simulate path over time
        Z = np.cumsum(Z - lambda_small * dt * jump_mean_small, axis=1)  # Compensate small jumps to ensure Z_t is zero-mean

        # Asset price process
        S = S0 * np.exp(B + mu*t + Y + Z)

        return S, t

    def get_yfinance_data(self, S0 = 1., ticker="^GSPC", start="2000-01-01", end="2025-01-01", window_size=252, split=False, plot=False):
        """Download and reformat yfinance data starting at S0. "
        Parameters: ticker: List of Stocks that are viewed, start, end: None"""

        data = yf.download(tickers=ticker, start=start, end=end, progress=False)["Adj Close"]
        data = np.array(data).T
        if plot or self.isSigLib == False:
            S = data / data[:, 0].reshape(-1, 1) * S0
            print('YFinance Dataset includes this many days (not only trading days): %s %s' % S.shape)
        else:
            first_stock = True
            for raw_stock in tqdm(data, desc="YFinance", leave=False):
                raw_stock = raw_stock[np.argmax(~np.isnan(raw_stock)):]
                for i in range(1, len(raw_stock)):
                    raw_stock[i] = np.where(np.isnan(raw_stock[i]), raw_stock[i - 1], raw_stock[i])
                if first_stock:
                    S = raw_stock / raw_stock[0].reshape(-1, 1) * S0  # normalize to S0
                    first_stock = False
                else:
                    raw_stock = raw_stock / raw_stock[0].reshape(-1, 1) * S[0][-1]  # normalizes values to the last value of the stock before
                    raw_stock = raw_stock[0][1:].reshape(1, -1)  # remove first value, to avoid 0 return
                    S = np.append(S, raw_stock, axis=1)  # GANs only allow for you one path. We reshape all paths into one since we only look at returns anyway
            # This gives Info about the Data used
            print('YFinance Dataset includes this many days (not only trading days): %s %s' % S.shape)

        # if split, split the data into chunks of length window_size:
        if split:
            returns = S[1:] / S[:-1]
            n = returns.shape[0] // (window_size - 1)
            # split such that some data in beginning of the time series is lost (returns.shape[0] % (n_points - 1) / (n_points - 1))
            n_returns = n * (window_size - 1)
            returns = returns[-n_returns:]
            returns = returns.reshape(n, window_size - 1)
            # t is the annualized time between each return
            t_raw = np.array((data.index[1:] - data.index[0]).days)[-n_returns:]
            t_raw = t_raw.reshape(n, window_size - 1)
            t = np.zeros((n, window_size))
            t[1:, 0] = t_raw[:-1, -1]
            t[:, 1:] = t_raw - t[:, 0].reshape(-1, 1)
            t[:, 0] = 0
            t = t / 365.25  # convert days to years
            S = np.zeros((n, window_size))
            S[:, 0] = S0
            S[:, 1:] = np.cumprod(returns, axis=1)
        else:
            t = np.arange(0, S.shape[1], 1)
            t = t / 365.25  # convert days to years

        return S, t


if __name__ == "__main__": #Testing
    test = 0
    if test == 0:
        import matplotlib.pyplot as plt

        GBM_parameter = {
            #"mu": 0.05,
            #"sigma": 0.2
        }
        Heston_parameter = {
            #"mu": 0.05,
            "v0_sqrt": 0.2,  # This is volatility,
            "kappa": 1.5,
            #"theta": 0.2**2,  # This is Variance, gbm_sigma indicates volatility
            "xi": 0.3, #Feller-Condition: 2*kappa*theta > xi**2
            "rho": -0.5
        }
        VarGamma_parameter = {
            #"mu": 0.05,
            #"sigma": 0.2,
            "nu": 0.02,
        }
        Kou_parameter = {
            #"mu": 0.12,
            #"sigma": 0.2,
            "kou_lambda": 2.0,
            "p": 0.3,
            "eta1": 50.,
            "eta2": 25.
        }
        LevyIto_parameter = {
            #"mu": 0.05,
            #"sigma": 0.2,
            "lambda_large": 2,
            "lambda_small": 300,
            "jump_mean_large": 0.03,
            "jump_std_large": 0.05,
            "jump_mean_small": 0.0005,
            "jump_std_small": 0.0005
        }
        YFinance_parameter = {
            "S0": 1.,
            "plot": False,
            "ticker": [
        # Major Indices
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",
        ]
        }
        """# Large-Cap Tech Stocks (FAANG & Others)
             "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ORCL", "INTC", "CSCO", "IBM", "ADBE", "CRM", "TXN",
    
             # Financial Sector
             "JPM", "BAC", "GS", "C", "WFC", "MS", "SCHW", "BLK", "BK", "AXP", "COF", "USB", "TFC", "CME",
    
             # Consumer Goods & Retail
             "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "TGT", "SBUX", "HD", "LOW", "DG", "TJX", "YUM",
    
             # Healthcare
             "JNJ", "PFE", "UNH", "MRK", "CVS", "LLY", "ABT", "TMO", "BMY", "DHR", "ZTS", "MDT", "BSX",
    
             # Energy Sector
             "XOM", "CVX", "SLB", "COP", "OXY", "PSX", "VLO", "HAL", "MPC", "BKR", "EOG", "FANG", "KMI",
    
             # Industrials
             "BA", "CAT", "MMM", "GE", "HON", "UPS", "UNP", "LMT", "RTX", "FDX", "CSX", "NSC", "WM", "NOC",
    
             # Utilities
             "NEE", "DUK", "SO", "D", "EXC", "AEP", "SRE", "PEG", "WEC", "ED", "XEL", "ES", "AWK", "DTE",
    
             # Telecommunications
             "T", "VZ", "TMUS", "CCI", "AMT", "VOD", "S", "CHT", "TU", "NOK", "ORAN", "BTI", "KT", "PHI",
    
             # Real Estate
             "PLD", "AMT", "CCI", "SPG", "PSA", "EQIX", "EQR", "ESS", "AVB", "O", "MAA", "UDR", "VTR", "HCP",
    
             # Consumer Discretionary
             "DIS", "HD", "MCD", "SBUX", "NKE", "LVS", "GM", "F", "HMC", "TM", "TSLA", "YUM", "MAR", "CCL",
    
             # ETFs and Funds
             "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "TLT", "XLF", "XLK", "XLE", "XLU", "XLI", "XLY", "XLP",
    
             # Commodities
             "CL=F", "GC=F", "SI=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "ZS=F", "LE=F", "HE=F", "KC=F", "CC=F", "CT=F",
    
             # Forex and Cryptocurrency
             "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD",
             "DOT-USD","""
        general_parameter = {
            "mu": 0.08,  # 0.0618421411431207,  # This is the YFinance data mean;
            "sigma": 0.2,  # 0.34787525475010267,  # This is the YFinance data Volatility;
            "S0": 1,
            "grid_points": 252,
            "window_size": 252 * 1,
            "num_paths": 1000
        }
        def generate_random_params(num_paths):
            low_vol = 0.1 * 3*(np.log(1000))**(0.8)/(np.log(num_paths)**(1.8))  # Adjustment of up and lower bound depending on num_paths size (number of correlations)
            up_vol = 0.25 * 3*(np.log(1000))**(0.8)/(np.log(num_paths)**(1.8))  # amounts to 22% vol

            mu = np.random.uniform(0.03, 0.13, size=num_paths)
            volatilities = np.random.uniform(low_vol, up_vol, size=num_paths)
            correlation = np.random.uniform(-1, 1, size=(num_paths, num_paths))
            np.fill_diagonal(correlation, 1)
            correlation = (correlation + correlation.T) / 2
            eigvals, eigvecs = np.linalg.eigh(correlation)
            eigvals[eigvals < 0] = 1e-5
            correlation = eigvecs @ np.diag(eigvals) @ eigvecs.T

            vola_matrix = correlation * np.outer(volatilities, volatilities)
            return mu, vola_matrix
        num_paths = general_parameter['num_paths']
        mu, vola_matrix = generate_random_params(num_paths)
        T = general_parameter['window_size']/general_parameter['grid_points']

        plot = True

        mu = np.array([0.06])
        vola_matrix = np.array([[0.44]])
        window_size = 1000
        models = {
            "Blackscholes": {**GBM_parameter, **general_parameter},
            #"Heston": {**Heston_parameter, **general_parameter},
            #"VarianceGamma": {**VarGamma_parameter, **general_parameter},
            #"Kou_Jump_Diffusion": {**Kou_parameter, **general_parameter},
            #"Levy_Ito": {**LevyIto_parameter, **general_parameter},
            #"YFinance": {"S0": 1},
            "correlated_Blackscholes": {"mu": mu, "vola_matrix": vola_matrix, "window_size": window_size, "num_paths": num_paths, "num_bm": 1}
        }

        if plot:
            # Set up the figure for multiple subplots
            fig, axes = plt.subplots(1, len(models), figsize=(20, 9), sharey=True)
            fig.suptitle('Comparison of Simulated Paths for Different Models')

            for i, (model_name, params) in enumerate(models.items()):
                # Load model and create simulated price data
                model = LoadData(dataset=model_name, data_params=params, isSigLib=False)
                prices_df = model.create_dataset("DataFrame")
                clear_output(wait=True)

                print('%s %s %s' % (model_name, prices_df.iloc[:, -1].mean()**(1 / T) - 1, prices_df.iloc[:, -1].std()/np.sqrt(T)))

                ax = axes[i]
                prices_df.T.plot(ax=ax, alpha=0.5, linewidth=0.3, legend=False)
                ax.set_title(f"{model_name} Model")
                ax.set_xlabel('Timeframe (years)')
                ax.grid(True)

            axes[0].set_ylabel('Price')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        else:
            import time
            for i, (model_name, params) in enumerate(models.items()):
                start_time = time.time()
                # Load model and create simulated price data
                model = LoadData(dataset=model_name, data_params=params, isSigLib=False)
                prices_df = model.create_dataset("DataFrame")

                print('%s %s %s' % (model_name, (prices_df.iloc[:, -1].mean()) ** (1 / T) - 1, prices_df.iloc[:, -1].std() / np.sqrt(T)))

                elapsed = time.time() - start_time
                print('Time Elapsed: %s' % elapsed)

    elif test == 1:
        p = 0.8
        risk_free_rate = 0.04
        total_vola = np.array([[0.2, 0.25]])
        weights = np.array([[0.7, 0.3], [0.2, 0.8]])  # rows sum to one
        mu = np.array([0.05, 0.07])
        vola_matrix = total_vola * np.sqrt(weights)  # [sigma]

        grid_points = 1
        window_size = 1
        num_paths = 2
        num_bm = 2

        T = window_size / grid_points
        dt = 1 / grid_points
        t = np.linspace(0, T, window_size + 1)
        cholesky = np.linalg.cholesky(vola_matrix)

        risky_lambda = mu - risk_free_rate
        analytical_risky_action = 1 / p * risky_lambda.T @ np.linalg.inv(cholesky @ cholesky.T)
        analytical_utility = np.exp((1 - p) * (risk_free_rate + 1 / (2 * p) * risky_lambda.T @ np.linalg.inv(cholesky @ cholesky.T) @ risky_lambda))

        print(f"Drift: {mu - vola_matrix / 2}")
        print(risky_lambda, analytical_risky_action, sum(analytical_risky_action), analytical_utility)

        def generate_gbm(mu, sigma, window_size, num_paths, grid_points=252, S0=1):

            # Wiener Processes
            dW = np.random.normal(0, np.sqrt(dt), size=(num_paths, window_size))
            W = np.cumsum(dW, axis=1)
            W = np.hstack([np.zeros((num_paths, 1)), W])

            # Asset price process from closed form
            S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

            return S[0][window_size]

        def generate_correlated_BS(mu, vola_matrix, window_size, num_paths, num_bm, grid_points=252, S0=1):
            S = np.zeros((num_paths, window_size + 1))
            S[:, 0] = S0

            for i in range(1, window_size + 1):
                Z = np.random.normal(size=num_paths)  # Independent standard normals
                S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * np.sum(vola_matrix, axis=1)) * dt + cholesky @ Z * np.sqrt(dt))

            # prices_df = pd.DataFrame(S, columns=t)
            # print('Mean and Std of correlated BS %s %s' % (prices_df.iloc[:, -1].mean()**(1 / T) - 1, prices_df.iloc[:, -1].std() / np.sqrt(T)))

            return S[0][window_size]

        num_returns = 100000
        prices_df = np.zeros(num_returns)
        prices_df2 = np.zeros(num_returns)

        for j in tqdm(range(num_returns)):
            prices_df[j] = generate_correlated_BS(mu=mu, vola_matrix=vola_matrix, window_size=window_size, num_paths=num_paths,  num_bm=num_bm, grid_points=grid_points, S0=1)
            #prices_df2[j] = generate_gbm(mu=mu[0], sigma=np.sqrt(vola_matrix[0][0]), window_size=window_size, num_paths=num_paths, grid_points=grid_points, S0=1)

        prices_df = pd.DataFrame(prices_df)
        #prices_df2 = pd.DataFrame(prices_df2)
        print(f"Correlated_BS; Mean: {prices_df.iloc[:, -1].mean() ** (1 / T) - 1}, {prices_df.iloc[:, -1].std() / np.sqrt(T)}")
        #print(f"GBM; Mean: {prices_df2.iloc[:, -1].mean() ** (1 / T) - 1}, {prices_df2.iloc[:, -1].std() / np.sqrt(T)}")

        observed_mu = np.log(prices_df).mean()[0]
        observed_sigma = np.log(prices_df).std()[0]
        #observed_mu2 = np.log(prices_df2).mean()[0]
        #observed_sigma2 = np.log(prices_df2).std()[0]
        print(f"Log; Mean: {observed_mu}, {observed_sigma}")
        #print(f"Log; Mean: {observed_mu2}, {observed_sigma2}")

    elif test == 2:
        risk_free_rate = 0.04
        p = 0.8
        #vola_matrix = np.array([[0.04]])
        #mu = np.array([0.06])

        sigma = np.array([0.2, 0.25])
        correlation = np.array([[1, 0.3], [0.3, 1]])
        vola_matrix = np.outer(sigma, sigma) * correlation
        mu = np.array([0.05, 0.07])

        cholesky = np.linalg.cholesky(vola_matrix)

        risky_lambda = mu - risk_free_rate
        analytical_risky_action = 1 / p * risky_lambda.T @ np.linalg.inv(cholesky @ cholesky.T)
        analytical_utility = np.exp((1 - p) * (risk_free_rate + 1/(2 * p) * risky_lambda.T @ np.linalg.inv(cholesky @ cholesky.T) @ risky_lambda))

        # optimal_portfolio *= np.dot([1 - analytical_risky_action[0], analytical_risky_action[0]], stock_data[current_step - 1])
        print(risky_lambda, analytical_risky_action, sum(analytical_risky_action), analytical_utility)

