import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, date
import matplotlib.pyplot as plt


'''
Code inspired by and based on article: 
https://www.codearmo.com/blog/pricing-options-monte-carlo-simulation-python


'''

def data_import(ticker="NVDA", hist_days=252):
    tk = yf.Ticker(ticker)
    hist = tk.history(period=f"{hist_days+5}d", interval="1d")
    spot = hist['Close'].iloc[-1]
    prices = hist['Close']
    return spot, prices

def compute_annual_vol(prices, trading_days=252):
    logrets = np.log(prices / prices.shift(1)).dropna()
    vol = logrets.std() * np.sqrt(trading_days)
    return float(vol)

def black_scholes_call_price(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def simulate_gbm_paths(S0, T, r, q, sigma, steps, n_sims):
    dt = T / steps
    S = np.zeros((steps + 1, n_sims))
    S[0] = S0
    for t in range(1, steps + 1):
        S[t] = S[t - 1] * np.exp(
            (r - q - 0.5 * sigma**2) * dt +
            sigma * np.sqrt(dt) * np.random.standard_normal(n_sims)
        )
    return S

def monte_carlo_call(S0, K, T, r, q, sigma, steps=100, n_sims=100000):
    paths = simulate_gbm_paths(S0, T, r, q, sigma, steps, n_sims)
    ST = paths[-1, :] 
    payoffs = np.maximum(ST - K, 0.0)
    price_mc = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs, ddof=1) / np.sqrt(n_sims)
    bs_price = black_scholes_call_price(S0, K, T, r, q, sigma)  
    return {
        'mc_price': price_mc,
        'mc_std_err': std_err,
        'bs_price': bs_price
    }  
def main():
    ticker = "NVDA"
    strike = 183
    expiry = "2026-12-20"  
    n_sims = 200000
    steps = 100 
    
    spot, prices = data_import(ticker, hist_days=252)
    sigma = compute_annual_vol(prices, trading_days=252)
    
    today = date.today()
    expiry_date = datetime.fromisoformat(expiry).date()
    days = (expiry_date - today).days
    T = days / 365.0
    r = 0.0366  # 1yr bonds yield as for 18.10.2025
    q = 0.0    # let s assume that dividends are neglectable
    paths = simulate_gbm_paths(spot, T, r, q, sigma, steps, 100)
    plt.plot(paths)
    plt.xlabel("Time Increments")
    plt.ylabel("Stock Price")
    plt.title("Geometric Brownian Motion")
    plt.show()
    print("Spot:", spot)
    print("Implied vol:", f"{sigma:.2%}")
    print("Time to expiry (years):", T)
    result = monte_carlo_call(spot, strike, T, r, q, sigma, steps, n_sims)
    print('---------------------')
    print("Pricing results:")
    print("Monte Carlo price:", result['mc_price'])
    print("Monte Carlo std err:", result['mc_std_err'])
    print("Black-Scholes price:", result['bs_price'])
if __name__ == "__main__":
    main()