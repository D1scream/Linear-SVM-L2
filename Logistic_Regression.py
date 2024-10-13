import numpy as np

class LogisticRegressionL2:
    def __init__(self, learning_rate=0.01, max_iters=10000, C=0.01, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None
        self.C = C  # Коэффициент регуляризации
        self.tolerance = tol

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        n = len(y)
        # Огранииваем логарифм (0,1)
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = (-1 / n) * (np.dot(y, np.log(y_pred)) + np.dot(1 - y, np.log(1 - y_pred)))
        loss += (self.C / (2 * n)) * np.sum(self.weights ** 2)

        return loss

    def fit(self, X, y):
        n, m = X.shape
        y=np.where(y == 1, 1, 0)
        self.weights = np.zeros(m)
        self.bias = 0
        prev_loss = float('inf')

        for i in range(self.max_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            dw = (1 / n) * np.dot(X.T, (y_pred - y)) + (self.C / n) * self.weights
            db = (1 / n) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            current_loss = self._compute_loss(y, y_pred)
            # if i % 100 == 0:
            #     print(f"Итерация {i}, Потери: {current_loss}")

            if abs(prev_loss - current_loss) < self.tolerance:
                # print(f"Сходимость достигнута на итерации {i}")
                break
            prev_loss = current_loss

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        return np.where(y_pred_proba >= 0.5, 1, -1)  # Предсказанные классы
