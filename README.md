# Overview

This is a pet project that utilizes an Explcit Finite-Difference Method (FDM) to approximate and discretize a value to a European Put Option. Furthermore, it then compares that numerical solution to the Black-Scholes Model's assigned value. Finally, we quantify the error. 

Here is a complete breakdown of how it works:

## 1. Importing libraries

'''python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
'''

We will use NumPy for vector operations and linear transformations to construct our grid, while using SciPy to import a standard normal cumulative distribution function (CDF). 

For context, the standard normal CDF gives the probability that a variable will take on a valule less than or equal to a specific value. It also represents the area under the standard normal probability density function (PDF) from negative infinity up to our variable. More on this when it comes up.

The MatPlotLib library will be used to plot our results as well as our quantified error versus the Black-Scholes Model. 

### 2. Defining our key variables

'''python
Strike = 50                 # Strike price
M = 400                     # Number of stock price steps
S_max = 100                 # Maximum stock price
Time = 0.25                 # Time til expiration (years)
rf_rate = 0.05              # Risk free rate 
sigma = 0.3                 # Volatility
'''
