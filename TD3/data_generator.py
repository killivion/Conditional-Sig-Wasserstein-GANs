import yfinance as yf
import os

import torch
import pandas as pd
import numpy as np

# from lib.algos.base import BaseConfig
from lib.arfnn import SimpleGenerator
from lib.utils import load_pickle
import DataLoader as DataLoader


class Data_Puller:
    def __init__(self, args, spec, data_params):
        self.loader = DataLoader.LoadData(dataset=args.dataset, isSigLib=False, data_params=data_params)
        if args.GAN_sampling:
            self.experiment_dir = f'./numerical_results/{args.dataset}/{spec}/seed=42/'
            print(self.experiment_dir)
            torch.random.manual_seed(0)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #base_config = BaseConfig(device=device)
            self.p, self.q = args.sig_p, args.sig_q  # base_config.p, base_config.q
            x_real = load_pickle(os.path.join(os.path.dirname(self.experiment_dir), 'x_real_test.torch')).to(device)  # change this to x_real.torch
            dim = x_real.shape[-1]
            self.x_past = x_real[:, :self.p]

            stats = torch.load(f'./numerical_results/{args.dataset}/{spec}/seed=42/meanstd.pt', weights_only=True)
            self.real_mean, self.reaL_std = stats['mean'], stats['std']

            G_weights = load_pickle(os.path.join(self.experiment_dir, 'SigCWGAN/G_weights.torch'))
            self.G = SimpleGenerator(dim * self.p, dim, 3 * (50,), dim).to(device)
            self.G.load_state_dict(G_weights)


    def pull_data(self, args, data_params):
        if args.GAN_sampling:
            data = self.generate()
        elif args.dataset == 'YFinance':
            ticker = data_params['data_params']['ticker']
            data = yf.download(ticker, start="2020-01-01", end="2024-01-01")['Adj Close']
        else:
            data = self.stochastically_get_data(args.dataset).T
        returns = data.pct_change().dropna().values + 1  # Compute dt change ratio [not dt returns]
        #incremental_risk_free_rate = (1 + args.risk_free_rate) ** (1 / args.grid_points)
        risk_free_column = np.full((returns.shape[0], 1), np.exp(args.risk_free_rate/args.grid_points))
        return np.hstack((risk_free_column, returns)), np.array(data)


    def stochastically_get_data(self, dataset):
        if dataset in ['Blackscholes', 'Heston', 'YFinance', 'correlated_Blackscholes']:
            data = self.loader.create_dataset(output_type="DataFrame")
        else:
            raise NotImplementedError('Dataset %s not valid' % dataset)
        return data


    def generate(self):
        with torch.no_grad():
            idx = torch.randint(0, self.x_past.shape[0], (1,)).item()
            x_past_sample = self.x_past[idx:idx+1]
            _x_past = x_past_sample.clone()
            x_fake_future = self.G.sample(self.q, _x_past)
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
