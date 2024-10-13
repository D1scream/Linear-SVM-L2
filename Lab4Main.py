import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D

from Linear_SVM import LinearSVCL2
from Logistic_Regression import LogisticRegressionL2
from sklearn.svm import LinearSVC

# Загружаем данные
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Цель классифицировать опухоль как доброкачественную (0) или злокачественную (1).
# Датасет содержит 30 числовых признаков: радиус, текстура, периметр, площадь и другие параметры опухоли, вычисленные на основе изображений ядер клеток.
# Классы: 2 класса (0 — доброкачественная, 1 — злокачественная).

# 1.Радиус ядра (mean radius)
# 2.Текстура (mean texture)
# 3.Периметр (mean perimeter)
# 4.Площадь (mean area)
# 5.Гладкость (mean smoothness)
# ... (и другие характеристики)

X = data.data
y = data.target
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# График зависимости качества модели на тестовой выборке от параметра С
Cs = np.logspace(-3, 3, 4)
log_reg_acc = []
svc_acc = []
sklearn_log_reg_acc = []
sklearn_svc_acc = []
for C in Cs:
    model_log = LogisticRegressionL2(C=C)
    model_log.fit(X_train_scaled, y_train)
    log_reg_acc.append(accuracy_score(y_test, model_log.predict(X_test_scaled)))
    
    sklearn_log_reg = LogisticRegression(C=C)
    sklearn_log_reg.fit(X_train_scaled, y_train)
    sklearn_log_reg_acc.append(accuracy_score(y_test,sklearn_log_reg.predict(X_test_scaled)))

    model_svc = LinearSVCL2(C=C)
    model_svc.fit(X_train_scaled, y_train)
    svc_acc.append(accuracy_score(y_test, model_svc.predict(X_test_scaled)))

    sklearn_model_svc = LinearSVC(C=C)
    sklearn_model_svc.fit(X_train_scaled, y_train)
    sklearn_svc_acc.append(accuracy_score(y_test, sklearn_model_svc.predict(X_test_scaled)))

plt.figure(figsize=(10, 6))
plt.plot(Cs, log_reg_acc, label='Logistic Regression',marker="x")
plt.plot(Cs, sklearn_log_reg_acc, label='sklearn Logistic Regression',marker="o")
plt.plot(Cs, svc_acc, label='Linear SVC',marker="+")
plt.plot(Cs, sklearn_svc_acc, label='sklearn Linear SVC',marker="*")
plt.xscale('log')
plt.xlabel('C (log)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C')
plt.legend()
#plt.show()

# Построение графиков коэффициентов для различных значений C
log_reg_accuracies = {}
sklearn_log_reg_accuracies = {}
svc_accuracies = {}
sklearn_svc_accuracies = {}

plt.figure(figsize=(14, 10))

# Первый график: коэффициенты для логистической регрессии
plt.subplot(2, 2, 1)
for C in Cs:
    log_reg = LogisticRegressionL2(C=C)
    log_reg.fit(X_train_scaled, y_train)
    log_reg_accuracies[C] = accuracy_score(y_test, log_reg.predict(X_test_scaled))
    plt.plot(log_reg.weights.ravel(), label=f'LogRegL2, C={C}')

plt.xlabel('Индекс признака')
plt.ylabel('Значение коэффициента')
plt.title('Значения коэффициентов для LogRegL2')
plt.legend()

# Второй график коэффициенты для sklearn логистической регрессии
plt.subplot(2, 2, 2)
for C in Cs:
    sklearn_log_reg = LogisticRegression(C=C)
    sklearn_log_reg.fit(X_train_scaled, y_train)
    sklearn_log_reg_accuracies[C] = accuracy_score(y_test, sklearn_log_reg.predict(X_test_scaled))
    plt.plot(sklearn_log_reg.coef_.ravel(), label=f'sklearn LogReg, C={C}')

plt.xlabel('Индекс признака')
plt.ylabel('Значение коэффициента')
plt.title('Значения коэффициентов для sklearn LogReg')
plt.legend()

# Третий график: коэффициенты для кастомной Linear SVC
plt.subplot(2, 2, 3)
for C in Cs:
    svc = LinearSVCL2(C=C)
    svc.fit(X_train_scaled, y_train)
    svc_accuracies[C] = accuracy_score(y_test, svc.predict(X_test_scaled))
    plt.plot(svc.weights.ravel(), label=f'LinearSVCL2, C={C}')

plt.xlabel('Индекс признака')
plt.ylabel('Значение коэффициента')
plt.title('Значения коэффициентов для LinearSVCL2')
plt.legend()

# Четвертый график коэффициенты для sklearn Linear SVC
plt.subplot(2, 2, 4) 
for C in Cs:
    sklearn_svc = LinearSVC(C=C)
    sklearn_svc.fit(X_train_scaled, y_train)
    sklearn_svc_accuracies[C] = accuracy_score(y_test, sklearn_svc.predict(X_test_scaled))
    plt.plot(sklearn_svc.coef_.ravel(), label=f'sklearn SVC, C={C}')

plt.xlabel('Индекс признака')
plt.ylabel('Значение коэффициента')
plt.title('Значения коэффициентов для sklearn SVC')
plt.legend()

# Объединение графиков
plt.tight_layout()

best_log_reg_C = max(log_reg_accuracies, key=log_reg_accuracies.get)
best_svc_C = max(svc_accuracies, key=svc_accuracies.get)
sklearn_best_log_reg_C = max(sklearn_log_reg_accuracies, key=sklearn_log_reg_accuracies.get)
sklearn_best_svc_C = max(sklearn_svc_accuracies, key=sklearn_svc_accuracies.get)
print(f"Лучшее C для Logistic Regression: {best_log_reg_C}, accuracy: {log_reg_accuracies[best_log_reg_C]:.4f}")
print(f"Лучшее C для Linear SVC: {best_svc_C}, accuracy: {svc_accuracies[best_svc_C]:.4f}")
print(f"Лучшее C для sklearn Logistic Regression: {sklearn_best_log_reg_C}, accuracy: {sklearn_log_reg_accuracies[sklearn_best_log_reg_C]:.4f}")
print(f"Лучшее C для sklearn Linear SVC: {sklearn_best_svc_C}, accuracy: {sklearn_svc_accuracies[sklearn_best_svc_C]:.4f}")

# Двумерный график границы принятия решений по двум признакам
X_vis = X_train_scaled[:, :2] 
y_vis = y_train

log_reg = LogisticRegressionL2(C=best_log_reg_C)
log_reg.fit(X_vis, y_vis)

svm = LinearSVCL2(C=best_svc_C)
svm.fit(X_vis, y_vis)

log_reg_sklearn = LogisticRegression(C=sklearn_best_log_reg_C)
log_reg_sklearn.fit(X_vis, y_vis)

svm_sklearn = LinearSVC(C=sklearn_best_log_reg_C)
svm_sklearn.fit(X_vis, y_vis)

# Создание сетки значений для построения границы принятия решений
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Предсказания для каждой точки сетки
Z_log_reg = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z_log_reg = Z_log_reg.reshape(xx.shape)

Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

Z_log_reg_sklearn = log_reg_sklearn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_log_reg_sklearn = Z_log_reg_sklearn.reshape(xx.shape)

Z_svm_sklearn = svm_sklearn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm_sklearn = Z_svm_sklearn.reshape(xx.shape)

plt.figure(figsize=(12, 10))

# График для Logistic Regression
plt.subplot(2, 2, 1)
plt.contourf(xx, yy, Z_log_reg, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
plt.title('Logistic Regression')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

# График для Linear SVM
plt.subplot(2, 2, 2)
plt.contourf(xx, yy, Z_svm, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
plt.title('Linear SVC')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

# График для sklearn Logistic Regression
plt.subplot(2, 2, 3)
plt.contourf(xx, yy, Z_log_reg_sklearn, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
plt.title('sklearn Logistic Regression')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

# График для sklearn Linear SVC
plt.subplot(2, 2, 4)
plt.contourf(xx, yy, Z_svm_sklearn, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
plt.title('sklearn Linear SVC')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

plt.tight_layout()
plt.show()
