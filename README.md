# Overview

This is a pet project that utilizes an Explcit Finite-Difference Method (FDM) to approximate and discretize a value to a European Put Option. Furthermore, it then compares that numerical solution to the Black-Scholes Model's assigned value. Finally, we quantify the error. 

Here is a complete breakdown of how it works:

## 1. Importing Libraries

'''python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
'''

We will use NumPy for vector operations and linear transformations to construct our grid, while using SciPy to import a standard normal cumulative distribution function (CDF). 

For context, the standard normal CDF gives the probability that a variable will take on a valule less than or equal to a specific value. It also represents the area under the standard normal probability density function (PDF) from negative infinity up to our variable. More on this when it comes up.

The MatPlotLib library will be used to plot our results as well as our quantified error versus the Black-Scholes Model. 

### 2. Options Basic Overview and Vocabulary

First off, what even is an option? Essentially, an option is a financial contract that gives the holder the right, **but not the obligation,** to buy or sell an asset (usually a stock or index) at a specific price on or before a certain date.
Furthermore, a **call option** gives the holder the right to **buy** the underlying asset at the specific price. A **put option** gives the holder the right to **sell** the underlying asset at the specific price.

Traders use options to provide flexibility and leverage to their existing investing/trading strategy. For example, let's say a trader owns 100 shares of some stock XYZ which is trading at $100 per share at the moment. For some reason, he is worried about a potential loss if XYZ happens to go down in share price to say $50 per share in the future. The trader can purchase a put option giving him the right to sell at $80 to hedge his underlying investment. That way, even if XYZ drops to $50 per share, his contract states that he will be able to sell his position for $80 per share rather than $50 per share. 

We say that a contract is "in-the-money" (ITM) if the price of the underlying asset is 1. **above** the strike price for **call options** or 2. **below** the strike price for **put options**. In our example above, since XYZ was trading lower than $80 per share, our trader's put contract was ITM. 
Consequently, we say that a contract is "out-of-the-money" (OTM) if the price of the underlying is 1. **below** the strike price for **call options** or 2. **above** the strike price for **put options**. In our example, if our trader owned a call option for $80 per share and not a put option, his contract would be OTM.

*Note that this is an oversimplification as option contracts are used in many different ways in countless unique trading strategies, but I trust that the reader gets the point. 

Let's now expand on some vocabulary.

In the world of Option Contract pricing, there are some terms that are confusing to grasp for the common person. So, here is a detailed explanation of each term used:
  Strike Price:   a **predetermined price** at which a contract can be bought or sold at ($80 per XYZ share was the **strike price** in our example above)
  Time:           all option contracts have a date attached that specifies when the contract reaches maturity. In other words, the contract **expires** at a certain date, which then nullifies any intrinsic value that the contract may have
  Risk-Free Rate: a representation of a rate of return that you could earn on a **completely safe investment**. In our FDM method as well as Black-Scholes, we assume a "risk-neutral" valuation framework, basically pretending that all assets grow on average by this risk-free rate. We do this to show that any extra                          expected returns due to risk is "priced out," giving no opportunity for arbitrage
  Sigma:          our annualized volatility variable, or in specificity: the standard deviation of the underlying asset's continuously compounded returns. So, given a sigma of 0.3, we basically are saying, "This stock's log returns over one year have a standard deviation of 30%." On a deeper level, a higher sigma                        means that stock price is expected to take a wilder and more random "path." This makes options contracts more valuable as there is a greater chance that they will end up far in-the-money or out-of-the-money

#### 3. Implementation

'''python
Strike = 50                 # Strike price
M = 400                     # Number of stock price steps
S_max = 100                 # Maximum stock price
Time = 0.25                 # Time til expiration (years)
rf_rate = 0.05              # Risk free rate 
sigma = 0.3                 # Volatility
'''

We set the variables that we will use in our discretization and computing process. The reason that we do this is that for this project, we want to declared parameters for our model. We are not Jane Street and do not have the knowledge or access to complex pricing models. Yet.
We just want to make it crystal-clear what assumptions the model is going off of.

Therefore, our variables are essentially telling us: "I would like to price a 3-month European put option with a strike price of $50, a risk-free rate of 5%, and a constant volatility of 30%."








