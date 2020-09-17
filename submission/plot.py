import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def klDiv(p, q):
    # assuming log as natural log
    if (q == 1) or (q == 0):
        return sys.float_info.max
    if p == 0:
        return math.log(1 / (1 - q))
    elif p == 1:
        return math.log(1 / q)
    return (p * math.log(p / q)) + ((1 - p) * math.log((1 - p) / (1 - q)))

x = np.linspace(0.3, 0.9999, 150)
y = [klDiv(0.3, xVal) for xVal in x]
print(y)
plt.plot(x, np.array(y))
plt.show()
