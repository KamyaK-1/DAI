import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = None

    def _generate_polynomial_features(self, X):
        """Generates polynomial features up to the given degree."""
        X_poly = np.column_stack([X**i for i in range(self.degree + 1)])
        return X_poly

    def fit(self, X, y):
        """Fits the polynomial regression model to the data."""
        X_poly = self._generate_polynomial_features(X)
        self.coefficients = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        """Predicts output for the given input data."""
        X_poly = self._generate_polynomial_features(X)
        return X_poly @ self.coefficients

    def evaluate(self, X, y):
        
        predictions = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - predictions)**2)
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared

np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 2 * X**2 - 3 * X + 5 + np.random.normal(0, 2, len(X))

degree = 2
model = PolynomialRegression(degree=degree)
model.fit(X, y)

# Make predictions
X_test = np.linspace(-3, 3, 100)
y_pred = model.predict(X_test)

# Evaluate the model
r_squared = model.evaluate(X, y)
print(f"R^2 Score: {r_squared:.4f}")

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label=f'Polynomial Fit (Degree {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title(f'Polynomial Regression (Degree {degree})')
plt.show()
