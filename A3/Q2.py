import numpy as np
import matplotlib.pyplot as plt

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data = data

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        return 2 * max(0,1 - np.sum(np.square((x-xi)/self.bandwidth)))/ np.pi

    def evaluate(self, x):
        """Evaluate the KDE at point x."""    
        return 2/(np.pi * (self.bandwidth ** 2)) * np.mean(np.maximum(np.zeros(self.data.shape[0]), 1 - np.sum(np.square((x - self.data)/self.bandwidth), axis=1)))



# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
bandwidth = 1.0
obj = EpanechnikovKDE(bandwidth)

# TODO: Fit the data
obj.fit(data)

# TODO: Plot the estimated density in a 3D plot

x = np.linspace(-6,6,120)
y = np.linspace(-7,7,140)
z = np.array([[obj.evaluate(np.array([i,j])) for i in x] for j in y])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)

surf = ax.plot_surface(X, Y, z, cmap='magma')
fig.colorbar(surf)
# TODO: Save the plot
plt.savefig('transaction_distribution.png')
plt.show()
