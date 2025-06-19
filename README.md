# Overview

This is a pet project that utilizes an Expilcit Finite-Difference Method (FDM) to approximate and discretize a value to a European Put Option. Furthermore, it then compares that numerical solution to the Black-Scholes Model's assigned value. Finally, we quantify the error. 

Here is a complete breakdown of how it works:

## 1. Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
```

We will use NumPy for vector operations and linear transformations to construct our grid, while using SciPy to import a standard normal cumulative distribution function (CDF). 

For context, the standard normal CDF gives the probability that a variable will take on a valule less than or equal to a specific value. It also represents the area under the standard normal probability density function (PDF) from negative infinity up to our variable. More on this when it comes up.

The MatPlotLib library will be used to plot our results as well as our quantified error versus the Black-Scholes Model. 

## 2. Options Basic Overview

First off, what even is an option? Essentially, an option is a financial contract that gives the holder the right, **but not the obligation,** to buy or sell an asset (usually a stock or index) at a specific price on or before a certain date.

Furthermore, a **call option** gives the holder the right to **buy** the underlying asset at the specific price. A **put option** gives the holder the right to **sell** the underlying asset at the specific price.

Traders use options to provide flexibility and leverage to their existing investing/trading strategy. For example, let's say a trader owns 100 shares of some stock XYZ which is trading at $100 per share at the moment. For some reason, he is worried about a potential loss if XYZ happens to go down in share price to say $50 per share in the future. The trader can purchase a put option giving him the right to sell at $80 to hedge his underlying investment. That way, even if XYZ drops to $50 per share, his contract states that he will be able to sell his position for $80 per share rather than $50 per share. 

We say that a contract is "in-the-money" (ITM) if the price of the underlying asset is 1. **above** the strike price for **call options** or 2. **below** the strike price for **put options**. In our example above, since XYZ was trading lower than $80 per share, our trader's put contract was ITM. 

Consequently, we say that a contract is "out-of-the-money" (OTM) if the price of the underlying is 1. **below** the strike price for **call options** or 2. **above** the strike price for **put options**. In our example, if our trader owned a call option for $80 per share and not a put option, his contract would be OTM.

*Note that this is an oversimplification as option contracts are used in many different ways in countless unique trading strategies, but I trust that the reader gets the point. 

## 3. Vocabulary

Let's now expand on some vocabulary.

In the world of Option Contract pricing, there are some terms that are confusing to grasp for the common person. So, here is a detailed explanation of each term used:
  
  Strike Price:   a **predetermined price** at which a contract can be bought or sold at ($80 per XYZ share was the **strike price** in our example above)
  
  Time:           all option contracts have a date attached that specifies when the contract reaches maturity. In other words, the contract **expires** at a certain date, which then nullifies any intrinsic value that the contract may have
  
  Risk-Free Rate: a representation of a rate of return that you could earn on a **completely safe investment**. In our FDM method as well as Black-Scholes, we assume a "risk-neutral" valuation framework, basically pretending that all assets grow on average by this risk-free rate. We do this to show that any extra                          expected returns due to risk is "priced out," giving no opportunity for arbitrage
  
  Sigma:          our annualized volatility variable, or in specificity: the standard deviation of the underlying asset's continuously compounded returns. So, given a sigma of 0.3, we basically are saying, "This stock's log returns over one year have a standard deviation of 30%." On a deeper level, a higher sigma                        means that stock price is expected to take a wilder and more random "path." This makes options contracts more valuable as there is a greater chance that they will end up far in-the-money or out-of-the-money

## 4. Implementation

```python
Strike = 50                 # Strike price
M = 400                     # Number of stock price steps
S_max = 100                 # Maximum stock price
Time = 0.25                 # Time til expiration (years)
rf_rate = 0.05              # Risk free rate 
sigma = 0.3                 # Volatility
```

We set the variables that we will use in our discretization and computing process. The reason that we do this is that for this project, we want to declared parameters for our model. We are not Jane Street and do not have the knowledge or access to complex pricing models. Yet.

We just want to make it crystal-clear what assumptions the model is going off of.

Therefore, our variables are essentially telling us: "I would like to price a 3-month European put option with a strike price of $50, a risk-free rate of 5%, and a constant volatility of 30%."

Let's explain our grid next.

## 5. The Grid and Explicit Finite-Difference Method (FDM)

Think of pricing our European put contract as filling in a spreadsheet where each column is a stock-price level going from $0 to $100 per share in evenly-spaced steps; each row is a moment in time from today back to expiration. 

At the very bottom row, time = T, if the stock price is S then the put is worth it's maximum potential value given by: (Strike - S, 0). So, if we know what the end value is, we can work backwards to figure out what the put is worth today which would be the top row of our spreadsheet.

Here we build our grid into our code:

```python
S = np.linspace(0, 100, M + 1)
V = np.maximum(Strike - S, 0)
V_new = V.copy()
```

Additionally, we compute our steps as such and then print them:

```python
dS = S_max / M
max_dt = (dS**2) / (sigma**2 * S_max**2)
N = int(np.ceil(Time / max_dt))
dt = Time / N
``` 
and

```python
print("Using N =", N, "so dt =", dt, "<= max_dt", max_dt)
```

At each backward (evenly-spaced) step we take in time, we compute each cell of our spreadsheet by taking a weighted average of three cells directly below it: the same-price cell, the one just below and to the left (a slightly lower stock price S), and the cell just below and to the right (a slightly higher price S).
    Note: This comes from two pieces of market data that we defined: sigma (volatility) and the risk free rate (how fast money grows safely)

We can relate this "taking a weighted average" to linear-algebra:

  Imagine we are multiplying a column vector V (which is an element of R^401) by a fixed matrix A that is of dimensions 401 x 401 whose only nonzero entries lie on the main diagonal and the diagonals immediately above and below the main diagonal. 

  A * V now produces a new column vector V_new whose i-th component is given by:             (V_new) = a_i V_i-1 + b_i V_i + c_i V_i+1

  We essentially compute V_new entry-by-entry without explicit loops.

  However, we have to realize that we are pricing a put contract, so the first and last entries need to address specific values. Therefore, the value at S = 0 is given by: Strike * e^(-rf_rate * dt). Additionally, the value at S_max = 100 is given by Strike = 0 (as it would be OTM).

  By iterating our matrix-vector multiplication backwards N times, we end up with the entire top row V(0), which is our FDM estimate of the put price as a function of stock price S.

**Note:** The Black-Scholes formula solves this problem all in one go using integration and normal distribution

Therefore, **theoretically as our grid that we formed (the spreadsheet) gets more refined, the repeated V_new = A * V multiplications will reproduce the values output by Black-Scholes with smaller and smaller errors.**

This is our implementation of the explicit FDM approximation process in Python:

```python
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
```

## 6. Black-Scholes Implementation

So now that we have computed our explicit FDM approximization, how do we **analytically** check it to see if our model is working?

Our answer lies in the Black-Scholes equation.

Previously, we said that in theory, our model would output values **closer and closer to the values generated by Black-Scholes**. So, why don't we just implement the B.S. equation (real...) and check our error?

We can create a function to compute the B.S. price:

```python
def bs_put_price(S, Strike, Time, rf_rate, sigma):
        
    # Create variables to compute d1 and d2 (which are arrays)
    d1 = (np.log(S / Strike) + (rf_rate + 0.5 * sigma**2) * Time) / (sigma * np.sqrt(Time))
    d2 = d1 - sigma * np.sqrt(Time)

    # Create risk-neutral terms
    term1 = Strike * np.exp(-rf_rate * Time) * norm.cdf(-d2)
    term2 = S * norm.cdf(-d1)
        
    return term1 - term2
```

I know, this looks hectic. So let's break it down a bit:

We know that Black-Scholes (for a Euro Put Contract) looks like this: **Put Value = Strike * e^(-rf_rate * T) * N(-d2) - S * N(-d1)** where N represents the standard normal CDF,

Where:
  d1 = ( ln ( S / Strike ) + ( rf_rate + sigma^2 / 2 ) * T ) / ( sigma * sqrt( T ) )
  d2 = d1 - ( sigma * sqrt( T ) )

Or, in Python:

```python
   d1 = (np.log(S / Strike) + (rf_rate + 0.5 * sigma**2) * Time) / (sigma * np.sqrt(Time))
   d2 = d1 - sigma * np.sqrt(Time)
```

So, we initially start by listing all variables that we need to input into our function:
```python
def bs_put_price(S, Strike, Time, rf_rate, sigma):
```

Next, we need to create vectors. We do so by passing in S (stock price) as a column vector in R^401.

After, we use NumPy's natural log function ```np.log()``` and square-root function ```np.sqrt()``` to compute our variables d1 and d2. These variables also are column vectors in R^401 as every operation we have done has been arithmetic based, meaning only the elements have been operated on.

To keep the linear algebra going, if we say:

  d1 = ( np.log( S / Strike ) + ( rf_rate + 0.5 * sigma**2 ) * Time ) / ( sigma * np.sqrt( Time ) ),

we are really saying:
  
  d1 = A * x + b, where x is a **column vector** given by ( ln ( S ) ) and b is a **constant** given by ( rf_rate + 1/2 * sigma^2 ) * T but scaled by 1 / ( sigma * sqrt( T ) )

Effectively, we are doing our operations element-wise, instead of via a matrix A.

## 7. Black-Scholes Implementation (cont.)

Remember how at the beginning of this file I said there would be more on the standard normal cumulative distribution function (CDF) and the standard normal probability density function (PDF) when they came up?

Well, here we are.

Let me explain what they are (from my knowledge at least):

  A standard normal PDF describes the bell-shaped curve of the standard normal distribution. You remember from your Intro to Statistics class, it has a mean of 0 and a std. deviation of 1. You can think of the PDF as a smooth function that weights values greater if they are near zero while tapering rapidly for a        larger absolute value of input. In options pricing, it quantifies how "likely" small deviations are.

  A standard normal CDF measures the area under the PDF curve from -infinity up to x. Intuitively, it tells you that if you pick some value x, then the CDF is the proportion of the bell-curve lying to the **left** of that point. If, for example, you picked x = 0, then the CDF( 0 ) = 0.5 because exactly half the         distribution lies below its mean. In options pricing, the CDF( -d2 ) and CDF( -d1 ) appears because they represent the risk-neutral probabilities of a Euro put contract finishing ITM.

This brings me to our code in which we compute our risk-neutral terms. Recall from the B.S. equation above (N = CDF):

  term1 = Strike * e^(-rf_rate * T) * N(-d2)
  term2 = S * N(-d1)

We compute this in Python and subtract term2 from term1:

```python
 # Create risk-neutral terms
    term1 = Strike * np.exp(-rf_rate * Time) * norm.cdf(-d2)
    term2 = S * norm.cdf(-d1)
        
    return term1 - term2
```

From here, we create a new grid of exact (not approximate) prices given by our B.S. function:

```python
# Compute the Black-Scholes price curve and produce an array of exact prices
V_bs = bs_put_price(S, Strike, Time, rf_rate, sigma)
```

And voila, we have computed an array of prices given by the Black-Scholes model.

## 8. Plotting the Overlay

So far, we have created our Explicit Finite-Difference Method array of estimated prices as well as our Black-Scholes array of exact prices.

Let's compare them visually!

We use MatPlotLib to create a plot:

```python
# Create an overlay plot
plt.plot(S, V, label="FDM Approximation")
plt.plot(S, V_bs, label="Black-Scholes Price")
plt.xlabel("Stock Price (S)")
plt.ylabel("Put Price")
plt.title("FDM vs. Black-Scholes at t=0")
plt.legend()
plt.grid()
plt.show()
```

Which outputs something like this: #TODO: add output plot

Not too bad, huh?
  
# 9. Compute and Plot Errors

So we know that our FDM generated estimated prices against Black-Scholes. Let's see where our absolute errors and relative errors could be and then plot them.

We start by computing absolute error and relative error, and creating an array out of them:

```python
# Compute absolute error array and relative error
error = np.abs(V - V_bs)
rel_error = error / V_bs
```

Next, let's find the max and mean error with location as well:

```python
# Print max and mean error as well as peak location
i_max = np.argmax(error)
print("Peak error:", error[i_max], "at S =", S[i_max])
print("Max error: ", np.max(error))
print("Mean error: ", np.mean(error))
print("Max relative error: ", np.max(rel_error))
```

Finally, let's plot this so we can see it visually:

```python
# Plot errors
plt.plot(S, error)
plt.xlabel("Stock Price (S)")
plt.ylabel("Absolute Error")
plt.title("Error: |FDM - Black-Scholes|")
plt.grid()
plt.show()
```

Which generates something like this: #TODO: add error plot





