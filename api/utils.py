import numpy as np
from scipy.stats import genpareto

def compute_VaR(weights, returns_data, alpha=0.05):
    portfolio_returns = np.dot(returns_data, weights)
    var = np.percentile(portfolio_returns, alpha * 100)
    return var

def compute_CVaR(weights, returns_data, alpha=0.05):
    portfolio_returns = np.dot(returns_data, weights)
    var = compute_VaR(weights, returns_data, alpha)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return cvar

# Calculate Tail Risk using Generalized Pareto Distribution (GPD)
def compute_tail_risk(returns_data, threshold=0.05):
    excess_returns = returns_data[returns_data > threshold] - threshold
    params = genpareto.fit(excess_returns)
    tail_risk = genpareto.ppf(0.99, *params)  # 99% quantile of the tail
    return tail_risk
