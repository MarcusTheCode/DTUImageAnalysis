import numpy as np

# Define the cost function
def cost_function(x1, x2):
    return x1**2 - x1 * x2 + 3 * x2**2 + x1**3

# Define the gradient of the cost function
def gradient(x1, x2):
    dC_dx1 = 2 * x1 - x2 + 3 * x1**2
    dC_dx2 = -x1 + 6 * x2
    return np.array([dC_dx1, dC_dx2])

# Gradient descent parameters
X = np.array([4.0, 3.0])  # Initial guess
alpha = 0.07              # Step size
threshold = 0.20
max_iterations = 1000

# Gradient descent process
for i in range(1, max_iterations + 1):
    grad = gradient(X[0], X[1])
    X = X - alpha * grad
    cost = cost_function(X[0], X[1])

    # Print first 5 iterations
    if i <= 5:
        print(f"Iteration {i}: x1 = {X[0]:.4f}, x2 = {X[1]:.4f}, Cost = {cost:.4f}")

    # Stop if cost drops below threshold and report
    if cost < threshold:
        print(f"\nCost dropped below 0.20 at iteration {i}")
        print(f"x1 = {X[0]:.4f}, x2 = {X[1]:.4f}, Cost = {cost:.4f}")
        break
