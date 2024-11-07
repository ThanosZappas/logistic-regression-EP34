import numpy as np
from sklearn.model_selection import train_test_split
from generate_dataset import generate_binary_problem
from logistic_regression import LogisticRegressionEP34

# Generate dataset
X, y = generate_binary_problem(centers=np.array([(0,4),(0,4)]), N=1000)

# Split dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=50)

# Initialize the LogisticRegressionEP34 model
model = LogisticRegressionEP34()

# Train the model
model.fit(X_train, y_train, show_line=True, batch_size=None, iterations=10000)

# Calculate accuracy on the test set
predictions = model.predict(X_test) >= 0.5
accuracy = np.mean(predictions == y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")