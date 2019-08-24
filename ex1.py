
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

pi = np.pi

def gaussian_density(x, mu=0.0, sigma=1.0):
    return 1/(sigma * np.sqrt(2 * pi) ) * np.exp(- (x-mu)**2 / (2 * sigma**2))

# Set random seed
np.random.seed(40)

# Generate random numbers from Gaussian
mu = 0.0
sigma2 = 2.0
n = 1000
array = np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=n)

# Plot histogram
count, bins, _ = plt.hist(array, bins=30, density=True)
plt.plot(bins, gaussian_density(bins, mu, np.sqrt(sigma2)), linewidth=2, color='r')
plt.xlabel('$x$'); plt.ylabel('$f(x)$')
plt.show()
plt.close()

# Inverse Transform Sampling
array = np.random.uniform(0.0, 1.0, size=n)
# Percent point function, same as inverse cdf
transformed = norm.ppf(array, loc=0.0, scale=1.0)

# Plot historgram
count, bins, _ = plt.hist(transformed, bins=30, density=True)
plt.plot(bins, gaussian_density(bins), linewidth=2, color='r')
plt.xlabel('$x$'); plt.ylabel('$f(x)$')
plt.show()
plt.close()

# Linear transformation from N(2, 10) to standard
mu = 2.0
sigma = 10.0
array = np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=n)

transformed = (array - mu) / np.sqrt(sigma2)
count, bins, _ = plt.hist(transformed, bins=30, density=True)
plt.plot(bins, gaussian_density(bins), linewidth=2, color='r')
plt.xlabel('$x$'); plt.ylabel('$f(x)$')
plt.show()
plt.close()