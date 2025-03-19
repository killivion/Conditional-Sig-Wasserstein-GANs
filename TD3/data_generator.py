import yfinance as yf
import os

import torch
import pandas as pd
import numpy as np

# from lib.algos.base import BaseConfig
from lib.arfnn import SimpleGenerator
from lib.utils import load_pickle



def generate_random_params(num_paths, num_bm):
    if num_paths == 2 and num_bm == 2:
        total_vola = np.array([[0.07, 0.12]])
        weights = np.array([[1, 0], [0, 1]])  # rows sum to one
        mu = np.array([0.06, 0.08])
    elif num_paths == 1 and num_bm == 1:  # 1 path, 1 brownian motion
        total_vola = np.array([[0.04]])
        weights = np.array([[1]])
        mu = np.array([0.06])
    elif num_paths == 2 and num_bm == 3:
        total_vola = np.array([[0.06, 0.14]])
        weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])  # rows sum to one
        mu = np.array([0.08, 0.10])

    else:  # Adjustment of up and lower bound depending on num_paths size (number of correlations), amounts to slightly more than 20% vol
        low_vol = 0.1 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (1.8))
        up_vol = 2.5 * low_vol
        low_mu, up_mu = 0.03, 0.13

        mu = np.random.uniform(low_mu, up_mu, size=num_paths)
        total_vola = np.random.uniform(low_vol, up_vol, size=num_paths)
        weights = np.random.rand(num_paths, num_bm)
        weights = weights / weights.sum(axis=1, keepdims=True)

        """
        correlation = np.random.uniform(-1, 1, size=(num_paths, num_bm))
        np.fill_diagonal(correlation, 1)
        correlation = (correlation + correlation.T) / 2
        eigvals, eigvecs = np.linalg.eigh(correlation)
        eigvals[eigvals < 0] = 1e-5
        correlation = eigvecs @ np.diag(eigvals) @ eigvecs.T  # correlation matrix with p_ij entries
        """

    vola_matrix = np.sqrt(weights * total_vola.T)  # [sigma] = vola_matrix

    return mu, vola_matrix  # mu is drift, vola_matrix


class Data_Puller:
    def __init__(self, args, spec, data_params):
        if args.dataset == 'YFinance' and not args.GAN_sampling:
            print(f"Data from: {data_params['data_params']['ticker']}, {data_params['data_params']['start']}, {data_params['data_params']['end']}")
            self.sample_data = yf.download(tickers=data_params['data_params']['ticker'], start=data_params['data_params']['start'], end=data_params['data_params']['end'], progress=False)['Close']
            self.start_index = 0
        else:
            import DataLoader as DataLoader
            self.loader = DataLoader.LoadData(dataset=args.dataset, isSigLib=False, data_params=data_params)
        if args.GAN_sampling:
            self.experiment_dir = f'./numerical_results/{args.dataset}/{spec}/seed=5/'
            print(self.experiment_dir)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #base_config = BaseConfig(device=device)
            self.p, self.q = args.sig_p, args.sig_q  # base_config.p, base_config.q
            self.x_real = load_pickle(os.path.join(os.path.dirname(self.experiment_dir), 'x_real_test.torch')).to(device)  # change this to x_real.torch
            print(self.x_real.shape)
            dim = self.x_real.shape[-1]

            stats = torch.load(f'./numerical_results/{args.dataset}/{spec}/seed=5/meanstd.pt', weights_only=True)
            self.real_mean, self.reaL_std = stats['mean'], stats['std']

            G_weights = load_pickle(os.path.join(self.experiment_dir, 'SigCWGAN/G_weights.torch'))
            self.G = SimpleGenerator(dim * self.p, dim, 3 * (50,), dim).to(device)
            self.G.load_state_dict(G_weights)


    def pull_data(self, args, data_params):
        if args.GAN_sampling:
            data = self.generate()
        elif args.dataset == 'YFinance':
            if self.start_index + args.window_size + 1 > len(self.sample_data):
                self.start_index = (self.start_index + args.window_size + 1) % len(self.sample_data)
            data = np.array(self.sample_data[self.start_index:self.start_index + args.window_size + 1]).T
            data = pd.DataFrame(data / data[:, 0].reshape(-1, 1)).T
            self.start_index = self.start_index + args.window_size
        elif args.dataset in ['Blackscholes', 'Heston', 'correlated_Blackscholes']:
            data = self.loader.create_dataset(output_type="DataFrame").T
        else:
            raise NotImplementedError('Dataset %s not valid' % args.dataset)
        returns = data.pct_change().dropna().values + 1  # Compute dt change ratio [not dt returns]
        #incremental_risk_free_rate = (1 + args.risk_free_rate) ** (1 / args.grid_points)
        risk_free_column = np.full((returns.shape[0], 1), np.exp(args.risk_free_rate/args.grid_points))
        return np.hstack((risk_free_column, returns)), np.array(data)


    def generate(self):
        with torch.no_grad():
            idx = torch.randint(0, self.x_real.shape[0], (1,)).item()
            x_real_sample = self.x_real[idx:idx+1]
            print(x_real_sample.shape)
            _x_real = x_real_sample.clone()
            print(_x_real.shape)
            x_fake_future = self.G.sample(self.q, _x_real)
            print(x_fake_future.shape)
            #print(f'x_fake_future: {x_fake_future}')
        S_fake_future = self.inverse_transformer(x_fake_future)
        #print(f'S_fake_future: {S_fake_future}')

        return S_fake_future


    def inverse_transformer(self, data):
        from lib.data import Pipeline, StandardScalerTS
        pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
        logrtn_recovered = pipeline.inverse_transform(data, self.real_mean, self.reaL_std)
        logrtn_recovered = logrtn_recovered.detach().cpu().numpy() if isinstance(logrtn_recovered,torch.Tensor) else logrtn_recovered
        logrtn_recovered = logrtn_recovered.squeeze(-1)

        log_prices_reconstructed = np.cumsum(logrtn_recovered, axis=1)
        price_paths_reconstructed = np.exp(log_prices_reconstructed)
        price_paths_reconstructed = np.insert(price_paths_reconstructed, 0, 1)

        return pd.DataFrame(price_paths_reconstructed)


if __name__ == "__main__": #Testing
    x_test = torch.tensor([[[ 0.6958],
         [ 0.0344],
         [ 0.8531],
         [ 1.7649],
         [-0.0655],
         [-0.0655],
         [ 1.8234],
         [ 0.9778],
         [-0.3106],
         [ 0.7436],
         [-0.3043],
         [-0.3067],
         [ 0.4305],
         [-1.8145],
         [-1.6183],
         [-0.4073],
         [-0.8766],
         [ 0.5058],
         [-0.7674],
         [-1.2927]]])

    from lib.data import Pipeline, StandardScalerTS

    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    logrtn_recovered = pipeline.inverse_transform(x_test)
    logrtn_recovered = logrtn_recovered.detach().cpu().numpy() if isinstance(logrtn_recovered, torch.Tensor) else logrtn_recovered
    logrtn_recovered = logrtn_recovered.squeeze(-1)

    log_prices_reconstructed = np.cumsum(logrtn_recovered, axis=1)
    price_paths_reconstructed = np.exp(log_prices_reconstructed)
    price_paths_reconstructed = np.insert(price_paths_reconstructed, 0, 1)

    print(pd.DataFrame(price_paths_reconstructed))

    data_test = np.array([[1.,         1.01393075, 1.00982871, 1.02826051, 1.0731694 , 1.0659443,
  1.05876833 ,1.10676 ,   1.13076921, 1.11573508, 1.13274016, 1.11787054,
  1.10312426, 1.11049313, 1.05205446 ,1.00199408 ,0.98609036 ,0.95819947,
  0.96656656, 0.94200516, 0.90511759]])
    log_prices = np.log(data_test)
    logrtn = np.diff(log_prices, axis=1)
    data_raw = torch.from_numpy(logrtn[..., None]).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_pre = pipeline.transform(data_raw)
