import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
# параметры контейнера для вывода графика
ax = fig.add_subplot(111, projection='3d')

# Подготовка данных
X = np.arange(-5, 5, 0.25)
Y = np.array(X)
X, Y = np.meshgrid(X, Y)    # расширение векторов X,Y в матрицы
Z = X**2 + Y**2

# Построение графика
# метод для отрисовки графиков с параметрами по умолчанию
ax.plot_surface(X, Y, Z)
plt.show()
