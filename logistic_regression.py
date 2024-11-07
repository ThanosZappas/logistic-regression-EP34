import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionEP34:
    def __init__(self, lr=0.01):
        # Initialize model parameters
        self.w = None
        self.b = None
        self.lr = lr
        self.N = None
        self.f = None
        self.dw = None
        self.db = None
        self.prediction = None

    def init_parameters(self, feature_dim):
        # Initialize weights and bias with small random values
        self.w = np.random.randn(feature_dim) * 0.1
        self.b = np.random.randn() * 0.1  # or = 0

    def forward(self, X):
        # Compute the linear combination of inputs and weights
        z = X @ self.w + self.b
        z = np.clip(z, -500, 500)  # Clip values to avoid overflow
        self.f = np.power(1 + np.exp(-z), -1)  # Apply sigmoid function

    def predict(self, X):
        # Predict probabilities using the sigmoid function
        z = X @ self.w + self.b
        z = np.clip(z, -500, 500)  # Clip values to avoid overflow
        return np.power(1 + np.exp(-z), -1)

    def loss(self, X, y):
        # Compute the binary cross-entropy loss
        self.prediction = self.predict(X)
        epsilon = 1e-9  # Small value to avoid log(0)
        return -np.mean(y * np.log(self.prediction + epsilon) + (1 - y) * np.log(1 - self.prediction + epsilon))

    def backward(self, X, y):
        # Compute gradients for weights and bias
        error = self.f - y
        self.dw = X.T @ error / len(y)
        self.db = np.mean(error)

    def step(self):
        # Update weights and bias using gradients
        self.w = self.w - self.lr * self.dw
        self.b = self.b - self.lr * self.db

    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False):
        # Ensure X and y are numpy arrays
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")

        # Ensure the dimensions of X and y are compatible
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same")

        # Initialize model parameters
        self.init_parameters(X.shape[1])

        # Shuffle the dataset
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        self.N = X.shape[0]

        for i in range(iterations):
            if batch_size is None:
                X_batch = X
                y_batch = y
            else:
                start = (i * batch_size) % self.N
                end = start + batch_size
                if end > self.N:
                    end = self.N
                X_batch = X[start:end]
                y_batch = y[start:end]

            # Perform forward and backward propagation
            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()

            # Print loss every show_step iterations
            if i % show_step == 0:
                loss_value = self.loss(X, y)
                print(f"Step {i}, Loss: {loss_value}")
                if show_line:
                    self.show_line(X, y)

    def show_line(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot data points for two classes, as well as the line
        corresponding to the model.
        """
        if X.shape[1] != 2:
            print("Not plotting: Data is not 2-dimensional")
            return
        idx0 = (y == 0)
        idx1 = (y == 1)
        X0 = X[idx0, :2]
        X1 = X[idx1, :2]
        plt.plot(X0[:, 0], X0[:, 1], 'gx')
        plt.plot(X1[:, 0], X1[:, 1], 'ro')
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
        yline = (self.w[0] * xline + self.b) / (-self.w[1])
        plt.plot(xline, yline, 'b')
        plt.show()