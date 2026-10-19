import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit_gradient_descent(self, X, y):
        """Trains linear regression using Gradient Descent."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        y = y.reshape(-1)

        for i in range(self.iterations):
            # Hypothesis (predictions)
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def fit_normal_equation(self, X, y):
        """Trains linear regression using the Normal Equation (Closed-form solution)."""
        # Add a column of ones for the bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = y.reshape(-1)
        
        # Normal Equation: theta = (X^T * X)^(-1) * X^T * y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """Makes predictions for input features X."""
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Quick test case with dummy data
    np.random.seed(42)
    X_train = np.random.rand(100, 3) # 3 features (e.g., temperature, solar radiation, humidity)
    true_weights = np.array([1.5, -2.0, 3.5])
    y_train = np.dot(X_train, true_weights) + 10.0 + np.random.randn(100) * 0.1

    print("Training model using Normal Equation...")
    model_ne = LinearRegressionScratch()
    model_ne.fit_normal_equation(X_train, y_train)
    print(f"Learned Weights (Normal Eq): {model_ne.weights}")
    print(f"Learned Bias (Normal Eq): {model_ne.bias:.4f}")

    print("\nTraining model using Gradient Descent...")
    model_gd = LinearRegressionScratch(learning_rate=0.1, iterations=1000)
    model_gd.fit_gradient_descent(X_train, y_train)
    print(f"Learned Weights (GD): {model_gd.weights}")
    print(f"Learned Bias (GD): {model_gd.bias:.4f}")
