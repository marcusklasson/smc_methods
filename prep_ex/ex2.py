
import numpy as np
import matplotlib.pyplot as plt

# Probability of hitting a dart within a circle, which is within a square? 
# Area of circle = pi * r**2, Area of square = 4 * r**2
# P = Area circle / Area square = pi*r**2 / 4*r**2 = pi / 4 

#np.random.seed(40)

# Estimate pi with MC method
n_samples = 10000

array = []
n_inner = 0
n_total = 0
for i in range(n_samples):
    x = np.random.uniform(low=-1.0, high=1.0, size=1)
    y = np.random.uniform(low=-1.0, high=1.0, size=1)

    # Check if sampled point is inside circle
    origin_dist = x*x + y*y
    if origin_dist <= 1.0:
        n_inner += 1
    n_total += 1

    pi = float(4*n_inner)/n_total
    array.append(pi)
    
    print("Sample {:d}, Estimate of pi: {}".format(i, pi))

# Plot estimate of pi as function over number of samples
plt.plot(list(range(1, n_samples+1)), array, linewidth=1.5, color='b')
plt.xlabel("Number of samples"); plt.ylabel(r"Estimate of $\pi$")
plt.title("Final estimate: {}".format(pi))
plt.grid()
plt.show()
plt.close()