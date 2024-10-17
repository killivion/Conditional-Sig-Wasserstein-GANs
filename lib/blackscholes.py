import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def blackscholes(S0, mu, sigma, T, dt):

    N = int(T / dt)  # number of time steps
    t = np.linspace(0, T, N)  # time array

    W = np.random.standard_normal(size=N)  # Wiener process
    W = np.cumsum(W) * np.sqrt(dt)  # cumulative sum of the Wiener process

    # Calculate the stock price series
    return S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


def gen_multiple_paths(num_paths=10, S0=100, mu=0.05, sigma=0.2, T=1, dt=1/365):
    #Generates Black-Scholes paths and stores them in a DataFrame

    N = int(T / dt)  # number of time steps
    time_index = np.linspace(0, T, N)  # time array

    df = pd.DataFrame(index=time_index)

    # Generate each time series and store it in the DataFrame
    for i in range(num_paths):
        series = blackscholes(S0, mu, sigma, T, dt)
        df[f'Series_{i + 1}'] = series

    return df

# Black-Scholes parameters
S0 = 100  # initial stock price
mu = 0.05  # expected return (drift)
sigma = 0.2  # volatility
T = 1  # time horizon (in years)
dt = 1 / 365  # time step

num_paths= 10 #number of different time series paths to generate

df_series = gen_multiple_paths(num_paths, S0, mu, sigma, T, dt)
print(df_series)

print(df_series.head())

# Plot all the time series
df_series.plot(figsize=(10, 6))
plt.title('Multiple Black-Scholes Stock Price Simulations')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
