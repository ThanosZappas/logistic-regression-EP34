import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionEP34:
    def __init__(self, lr=0.01):
        self.w = None
        self.b = None
        self.lr = lr
        self.N = None
        self.f = None
        self.dw = None
        self.db = None
        self.prediction = None

    def init_parameters(self, feature_dim):
        self.w = np.random.randn(feature_dim) * 0.1
        self.b = np.random.randn() * 0.1

    def forward(self, X):
        z = X @ self.w + self.b
        self.f = 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = X @ self.w + self.b
        return 1 / (1 + np.exp(-z))

    def loss(self, X, y):
        self.prediction = self.predict(X)
        epsilon = 1e-9
        return -np.mean(y * np.log(self.prediction + epsilon) + (1 - y) * np.log(1 - self.prediction + epsilon))

    def backward(self, X, y):
        self.prediction = self.predict(X)
        self.dw = --1 / self.N * X.T @ (y - self.prediction)
        self.db = -1 / self.N * np.sum(y - self.prediction)

    def step(self):
        self.w = self.w - self.lr * self.dw
        self.b = self.b - self.lr * self.db

    # def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False):
    #     self.N = X.shape[0]
    #     for i in range(iterations):
    #         self.forward(X)
    #         self.backward(X, y)
    #         self.step()
    #
    #         if show_line and i % show_step == 0:
    #             loss_value = self.loss(X, y)
    #             print(f"Iteration {i}, Loss: {loss_value}")

    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False):
        # Επιβεβαίωση ότι τα X και y είναι numpy arrays
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")

        # Επιβεβαίωση ότι οι διαστάσεις των X και y είναι συμβατές
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same")

        # Αρχικοποίηση των παραμέτρων του μοντέλου
        self.init_parameters(X.shape[1])

        # Τυχαία αναδιάταξη των δειγμάτων στον πίνακα X
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

            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()

            if i % show_step == 0:
                loss_value = self.loss(X, y)
                print(f"Iteration {i}, Loss: {loss_value}")
                self.show_line(X,y)

    def show_line(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot data points for two classes, as well as the line
        corresponding to the model.
        """

        if (X.shape[1] != 2):
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



# # Δημιουργία αντικειμένου του μοντέλου
# model = LogisticRegressionEP34()
#
# # Δεδομένα εκπαίδευσης (π.χ., 100 δείγματα, 3 χαρακτηριστικά)
# X_train = np.random.randn(100, 3)
# y_train = np.random.randint(0, 2, 100)
#
# # Αρχικοποίηση παραμέτρων w και b με βάση τη διάσταση χαρακτηριστικών του X_train
# model.init_parameters(feature_dim=X_train.shape[1])
#
# # Υπολογισμός της απώλειας
# loss_value = model.loss(X_train, y_train)
# print("Απώλεια:", loss_value)
