#%%
import numpy as np
from scipy.stats import poisson
import time


tnow = time.time()
print(poisson.cdf(10, 5e6))

print('Elapsed: ', time.time() - tnow)

# %%
