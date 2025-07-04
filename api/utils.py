import numpy as np
from scipy.stats import genpareto

# Calculate Value-at-Risk (VaR) at a given confidence level (e.g., 5%)
def compute_VaR(weights, returns_data, alpha=0.05):
    portfolio_returns = np.dot(returns_data, weights)
    var = np.percentile(portfolio_returns, alpha * 100)  # VaR at the alpha percentile
    return var

# Calculate Conditional Value-at-Risk (CVaR) at a given confidence level (e.g., 5%)
def compute_CVaR(weights, returns_data, alpha=0.05):
    portfolio_returns = np.dot(returns_data, weights)
    var = compute_VaR(weights, returns_data, alpha)  # Get VaR first
    cvar = portfolio_returns[portfolio_returns <= var].mean()  # Average loss beyond VaR threshold
    return cvar

# Calculate Tail Risk using Generalized Pareto Distribution (GPD)
def compute_tail_risk(returns_data, threshold=0.05):
    excess_returns = returns_data[returns_data > threshold] - threshold
    params = genpareto.fit(excess_returns)
    tail_risk = genpareto.ppf(0.99, *params)  # 99% quantile of the tail
    return tail_risk
