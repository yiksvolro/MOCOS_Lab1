import numpy as np
from prettytable import PrettyTable
from scipy import integrate
import math
import matplotlib.pyplot as plt

n = 16
num = 50
ak = list()
bk = list()

def f(x):
    return pow(x, n) * math.exp(-pow(x, 2) / n)

def coeficients(x):

    for i in range(num):
        a_k = 1 / math.pi * integrate.quad(lambda x: f(x) * math.cos(i*x), -math.pi, math.pi)[0]
        ak.append(a_k)
        b_k = 1 / math.pi * integrate.quad(lambda x: f(x) * math.sin(i*x), -math.pi, math.pi)[0]
        bk.append(b_k)

    print_coefficients()

    sum = 0
    for i in range(num):
        sum += ak[i] * math.cos(i*x) + bk[i] * math.sin(i*x)

def print_coefficients():
    th = ['k', 'a_k', 'b_k']
    td = []
    print("\nКоефіцієнти тригонометричного ряду Фур'є")

    for k in range(0, num):
        td.append(k)
        if k == 0:
            td.append((round(ak[k], 6)))
            td.append("None")
            continue
        td.append((str(round(ak[k], 6))))
        td.append((str(round(bk[k], 6))))

    columns = len(th)
    table = PrettyTable(th)

    file_object = open("./file.txt", "w", encoding="utf-8")
    file_object.write("Коефіцієнти ряду Фур'є\n")
    while td:
        table.add_row(td[:columns])
        file_object.write('a_{0:<3} = {1:<12} b_{0:<2} = {2:<12}'.format(*td[:columns]) + "\n")
        td = td[columns:]

    print(table)


def relative_error(y_vals, y_series_vals):
    f_norm = np.linalg.norm(y_vals)
    error_norm = np.linalg.norm(np.array(y_vals) - np.array(y_series_vals))
    return error_norm / f_norm


def get_errors(x_vals):
    errors = []
    y_series_vals = np.zeros(len(x_vals))
    for n in range(1, num):
        y_series_vals += ak[n] * np.cos(n * x_vals) + bk[n] * np.sin(n * x_vals)

    f_vals = x_vals ** n * np.exp(-x_vals ** 2 / n)
    for i in range(len(x_vals)):
        errors.append(relative_error(f_vals[i], y_series_vals[i]))
    return errors

def plot_func():
    def plot_the_func(x,y, title):
        fig, ax = plt.subplots()
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.grid(True)

    x = np.linspace(-math.pi, math.pi, 500)
    x2 = np.linspace(-math.pi, 3*math.pi, 500)
    x3 = np.linspace(-3*math.pi, math.pi, 500)

    y = [f(x_i) for x_i in x]
    y2 = [f(x_i) for x_i in x2]
    y3 = [f(x_i) for x_i in x3]

    plot_the_func(x,y,'Графік функції x^16*exp(-x^2/16) на проміжку [-π;π]')
    plot_the_func(x2,y2,'Графік функції x^16*exp(-x^2/16) на проміжку [-π;3π]')
    plot_the_func(x3,y3,'Графік функції x^16*exp(-x^2/16) на проміжку [-3π;π]')


    # Функція a(k) в частотній області
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_title('Функція a(k) в частотній області', fontsize=16)
    plt.grid(True)
    for i in range(0, num):
        a_value = ak[i]
        plt.plot(i, a_value, 'ro-')
        plt.plot([i, i], [0, a_value], 'r-')

    # Функція b(k) в частотній області
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    ax3.set_title('Функція b(k) в частотній області', fontsize=16)
    plt.grid(True)
    for i in range(1, num):
        b_value = bk[i]
        plt.plot(i, b_value, 'bo-')
        plt.plot([i, i], [0, b_value], 'b-')

    # Побудова графіка відносної помилки
    x_vals = np.linspace(-np.pi, np.pi, 500)
    errors = get_errors(x_vals)
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    ax4.set_title('Графік відносної похибки', fontsize=16)
    plt.grid(True)
    plt.plot(x_vals, errors)
    plt.xlabel('x')
    plt.ylabel('Relative error')

    def fourier_series(x, n, k_max, a_coeffs, b_coeffs):
        series_sum = np.zeros_like(x)
        for k in range(1, k_max + 1):
            series_sum += a_coeffs[k] * np.cos(k * x) + b_coeffs[k - 1] * np.sin(k * x)
            yield series_sum

    y_vals = [f(x_i) for x_i in x_vals]
    y_series_vals = fourier_series(x_vals, n, 20, ak, bk)
    fig5, ax5 = plt.subplots(figsize=(10, 10))
    ax5.set_title('Графік поступового наближення функції f з рядом Фурє', fontsize=16)
    plt.grid(True)
    plt.plot(x_vals, y_vals, label='f(x)')
    for k, y_k in enumerate(y_series_vals):
        plt.plot(x_vals, y_k, label=f'N_{k + 1}(x)')
    plt.legend()
    plt.show()

x = input("Введіть х = ")
y = f(int(x))
print("F(" + x + ")=" + str(y))
coeficients(int(x))
plot_func()