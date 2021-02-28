from random import randint
import prettytable
import numpy as np


size_x = 20
size_a = 10
a = [randint(1, size_a) for _ in range(4)]

# заповнення таблички по Х та розрахунок Y
X1 = np.array([randint(1, size_x) for _ in range(8)])
X2 = np.array([randint(1, size_x) for _ in range(8)])
X3 = np.array([randint(1, size_x) for _ in range(8)])
Y = np.array([(a[0] + a[1]*X1[i] + a[2]*X2[i] + a[3]*X3[i]) for i in range(8)])

# пошук х0
x01 = (max(X1) + min(X1))/2
x02 = (max(X2) + min(X2))/2
x03 = (max(X3) + min(X3))/2
y0 = (max(Y) + min(Y))/2

# інтервал зміни фактора
dx1 = max(X1) - x01
dx2 = max(X2) - x02
dx3 = max(X3) - x03

# нормоване значення
Xn1 = np.array([round((X1[i] - x01)/dx1, 3) for i in range(8)])
Xn2 = np.array([round((X2[i] - x02)/dx2, 3) for i in range(8)])
Xn3 = np.array([round((X3[i] - x03)/dx3, 3) for i in range(8)])

# формуємо всю табличку
table = prettytable.PrettyTable()

table.field_names = ["#", "X1", "X2", "X3", "Y", "Xn1", "Xn2", "Xn3"]
for i in range(8):
    table.add_row([i, X1[i], X2[i], X3[i], Y[i], Xn1[i], Xn2[i], Xn3[i]])

table.add_row(["x0", x01, x02, x03, y0, "--", "--", "--"])
table.add_row(["dx", dx1, dx2, dx3, "--", "--", "--", "--"])

print(table)

# завдання по варіанту
Ye = a[0] + a[1]*x01 + a[2]*x02 + a[3]*x03
print("Еталонне значення Y =", Ye)

sh = [(i - Ye)**2 for i in Y]
print(sh)
print("Шукане мінімальне =", min(sh))
