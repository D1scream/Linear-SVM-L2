import numpy as np

class LogisticRegressionL2:
    def __init__(self, learning_rate=0.01, max_iters=10000, C=0.01, tol=1e-4):
        self.learning_rate = learning_rate  # Скорость обучения
        self.max_iters = max_iters  # Количество итераций
        self.weights = None  # Веса модели
        self.bias = None  # Смещение (bias)
        self.C = C  # Коэффициент регуляризации
        self.tolerance = tol

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        """
        Формула:
        L(y, y_pred) = (-1/m) * [Σ(y * log(y_pred) + (1 - y) * log(1 - y_pred))] + 
                            (C/(2*m)) * Σ(w_i^2)
        """
        m = len(y)
        # Огранииваем логарифм (0,1)
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = (-1 / m) * (np.dot(y, np.log(y_pred)) + np.dot(1 - y, np.log(1 - y_pred)))

        loss += (self.C / (2 * m)) * np.sum(self.weights ** 2)

        return loss

    def fit(self, X, y):
        m, n = X.shape
        y=np.where(y == 1, 1, 0)
        self.weights = np.zeros(n)
        self.bias = 0
        prev_loss = float('inf')

        for i in range(self.max_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            dw += (self.C / m) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            current_loss = self._compute_loss(y, y_pred)
            # if i % 100 == 0:
            #     print(f"Итерация {i}, Потери: {current_loss}")

            # Условие сходимости
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
