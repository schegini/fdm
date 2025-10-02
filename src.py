import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define important variables
Strike = 50                 # Strike price
M = 400                     # Number of stock price steps
S_max = 100                 # Maximum stock price
Time = 0.25                 # Time til expiration (years)
rf_rate = 0.05              # Risk free rate 
sigma = 0.3                 # Volatility

# Compute step sizes
dS = S_max / M
max_dt = (dS**2) / (sigma**2 * S_max**2)
N = int(np.ceil(Time / max_dt))
dt = Time / N

# Print the number of time steps
print("Using N =", N, "so dt =", dt, "<= max_dt", max_dt)

# Build grid and payoff
S = np.linspace(0, 100, M + 1)
V = np.maximum(Strike - S, 0)
V_new = V.copy()

# Create time steps (explicit FDM)
for n in range(N):
    for i in range(1, M):
        a = 0.5 * dt * (sigma**2 * i**2 - rf_rate * i)
        b = 1 - dt * (sigma**2 * i**2 + rf_rate)
        c = 0.5 * dt * (sigma**2 * i**2 + rf_rate * i)
        V_new[i] = a * V[i-1] + b * V[i] + c * V[i+1]

    # Create boundary conditions
    V_new[0] = Strike * np.exp(-rf_rate * ( (n + 1) * dt))
    V_new[M] = 0

    V = V_new.copy()

# Day 3 additions: overlay with Black-Scholes and plot errors

# Create a function to compute the Black-Scholes pricing
def bs_put_price(S, Strike, Time, rf_rate, sigma):
        
    # Create variables to compute d1 and d2 (which are arrays)
    d1 = (np.log(S / Strike) + (rf_rate + 0.5 * sigma**2) * Time) / (sigma * np.sqrt(Time))
    d2 = d1 - sigma * np.sqrt(Time)

    # Create risk-neutral terms
    term1 = Strike * np.exp(-rf_rate * Time) * norm.cdf(-d2)
    term2 = S * norm.cdf(-d1)
        
    return term1 - term2

# Compute the Black-Scholes price curve and produce an array of exact prices
V_bs = bs_put_price(S, Strike, Time, rf_rate, sigma)

# Create an overlay plot
plt.plot(S, V, label="FDM Approximation")
plt.plot(S, V_bs, label="Black-Scholes Price")
plt.xlabel("Stock Price (S)")
plt.ylabel("Put Price")
plt.title("FDM vs. Black-Scholes at t=0")
plt.legend()
plt.grid()
plt.show()

# Compute absolute error array and relative error
error = np.abs(V - V_bs)
rel_error = error / V_bs

# Print max and mean error as well as peak location
i_max = np.argmax(error)
print("Peak error:", error[i_max], "at S =", S[i_max])
print("Max error: ", np.max(error))
print("Mean error: ", np.mean(error))
print("Max relative error: ", np.max(rel_error))

# Plot errors
plt.plot(S, error)
plt.xlabel("Stock Price (S)")
plt.ylabel("Absolute Error")
plt.title("Error: |FDM - Black-Scholes|")
plt.grid()
plt.show()
