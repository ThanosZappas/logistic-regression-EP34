import numpy as np

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
        self.db = l_grad_b = -1 / self.N * np.sum(y - self.prediction)

    def step(self):
        self.w = self.w - self.lr * self.dw
        self.b = self.b - self.lr * self.db


# Δημιουργία αντικειμένου του μοντέλου
model = LogisticRegressionEP34()

# Δεδομένα εκπαίδευσης (π.χ., 100 δείγματα, 3 χαρακτηριστικά)
X_train = np.random.randn(100, 3)
y_train = np.random.randint(0, 2, 100)

# Αρχικοποίηση παραμέτρων w και b με βάση τη διάσταση χαρακτηριστικών του X_train
model.init_parameters(feature_dim=X_train.shape[1])


# Υπολογισμός της απώλειας
loss_value = model.loss(X_train, y_train)
print("Απώλεια:", loss_value)
