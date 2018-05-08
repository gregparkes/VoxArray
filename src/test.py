import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(10)
y = np.diff(x, 0)
print(x)
print(y)
plt.plot(x, y, 'x')
plt.show()
