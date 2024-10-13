import numpy as np

class LinearSVCL2:
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, learning_rate=0.01):
        self.C = C 
        self.max_iter = max_iter 
        self.tol = tol 
        self.learning_rate = learning_rate  # Шаг градиентного спуска
        self.weights = None
        self.bias = 0 
    
    def _hinge_loss(self, X, y):
        pred = np.dot(X, self.weights) + self.bias
        hinge_loss = np.maximum(0, 1 - y * pred)
        loss = (1 / (2*X.shape[0])) * np.sum(self.weights ** 2) + self.C * np.mean(hinge_loss)
        
        return loss
    
    def fit(self, X, y):
        n, m = X.shape
        self.weights = np.zeros(m)
        # Оптимизация методом градиентного спуска
        for i in range(self.max_iter):
            pred = np.dot(X, self.weights) + self.bias
            margin = y * pred<1 # Условие для неправильно классифицированных объектов
            dw = self.weights - self.C * np.dot(X.T, margin*y) / n
            db = -self.C * np.sum(y[margin]) / n
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            #if i % 100 == 0:
                #loss = self._hinge_loss(X, y)
                #print(f"Итерация {i}, Потери: {loss}, Норма градиента: {np.linalg.norm(dW)}")
            
            # Проверка сходимости
            if np.linalg.norm(dw) < self.tol:
                #print(f"SVC: Сходимость достигнута на итерации {i}")
                break
    
    def predict(self, X):
        pred = np.dot(X, self.weights) + self.bias
        return np.where(pred >= 0, 1, -1)  # Вектор предсказанных меток (1 или -1)
