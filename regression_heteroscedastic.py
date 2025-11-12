import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor


# Function to generate heteroscedastic data
def sample(n, seed):
    X = np.random.default_rng(seed=seed).normal(loc=0, scale=1, size=n)
    epsilon = np.random.default_rng(seed=seed + 1).normal(loc=0, scale=1, size=n)
    variance = (4 / 3) * stats.norm.pdf(X)
    y = X / 2 + variance * epsilon
    return X[:, None], y


# Load or generate data
X, y = sample(2000, seed=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=21
)

# Initialize the regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test, output_type="main", quantiles=[0.05, 0.95])

q1 = predictions["quantiles"][0]
q2 = predictions["quantiles"][1]
preds = predictions["mean"]

# Sort for better visualization
sorted_indices = np.argsort(X_test.flatten())
X_test = X_test[sorted_indices]
y_test = y_test[sorted_indices]
preds = preds[sorted_indices]
q1 = q1[sorted_indices]
q2 = q2[sorted_indices]
coverage = np.mean((y_test >= q1) & (y_test <= q2))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="black", alpha=0.5, label="Test data")
plt.plot(X_test, preds, color="red", linewidth=2, alpha=0.5, label="Predictions")
plt.fill_between(
    X_test.flatten(), q1, q2, color="blue", alpha=0.3, label="90% Prediction Interval"
)
plt.xlabel("X")
plt.ylabel("y")
plt.title(f"TabPFN. Empirical coverage: {coverage * 100:.2f}%")
plt.legend()
plt.show()
