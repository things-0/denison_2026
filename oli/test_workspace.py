import numpy as np
import scipy.signal as sps
import lmfit.models as lm
import matplotlib.pyplot as plt

x = np.arange(-100, 100)
y = -x**2 + 5*x + 6e3 + np.random.normal(0, 700, len(x))

plt.plot(x, y)
plt.axhline(y=0, color='black', linestyle='--')
plt.show()

