import numpy as np


class LinearRegression:
    """This class uses the closed form solution in order to build a linear regression model from scratch"""

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """This fits the function"""
        rows, cols = X.shape
        # X = np.hstack((np.ones((rows, 1)), X))
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.w = w
        # self.b = w[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """This predicts the function"""
        # rows, cols = X.shape
        # X = np.hstack((np.ones((rows, 1)), X))  # add bias term to input
        pred = X @ self.w + self.b
        return pred


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """Fits the function. Updates weights and biases at each epoch"""
        rows, cols = X.shape
        self.w = np.random.randn(X.shape[1], 1)  # initialize weights randomly
        self.b = np.random.randn(1)

        y = y.reshape(-1, 1)

        for i in range(epochs):
            # Forward pass
            preds = X @ self.w + self.b
            err = y - preds

            # calculate loss
            loss = np.sum(np.square(err))
            loss = loss / rows

            # calculate gradient
            dw = -1 * (X.T @ err) / rows
            # db = np.sum(err) / rows
            db = -1 * np.sum(err) / rows

            # update weights and biases
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # rows, cols = X.shape
        # X = np.concatenate((np.ones((rows, 1)), X), axis=1) # add bias term to input
        pred = X @ self.w + self.b
        return pred


# X = np.array([[1, 2], [3, 4]])
# y = np.array([1, 2])
# lr = GradientDescentLinearRegression()
# lr.fit(X, y, epochs=10)
# mse1 = mse(y, lr.predict(X))
