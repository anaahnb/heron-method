import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heron_method(func, x0, x1, epsilon):
    """
    Implementa o método de Herão para aproximar a raiz quadrada de um número, verificando as hipóteses.

    :param func: A função para a qual estamos tentando encontrar a raiz quadrada.
    :param x0: O valor inicial inferior do intervalo.
    :param x1: O valor inicial superior do intervalo.
    :param epsilon: A precisão desejada para a aproximação.
    :return: A aproximação da raiz quadrada.
    """
    if not (func(x0) * func(x1) < 0):
        print("A função não é contínua no intervalo fornecido.")
        return None

    if func(x0) > 0 and func(x1) > 0:
        print("O intervalo inicial não contém a raiz quadrada.")
        return None

    x = x0
    n = 0

    # Loop de aproximação
    while abs(x1 - x0) > epsilon:
        n += 1
        x = (x0 + x1) / 2
        if func(x) == 0:
            return x, n
        elif func(x0) * func(x) < 0:
            x1 = x
        else:
            x0 = x

    return (x0 + x1) / 2, n


def teste(x):
    return x**4 - x - 2


x0 = 1
x1 = 2
epsilon = 0.001

# Encontrando a raiz e o número de iterações
raiz, n = heron_method(teste, x0, x1, epsilon)

# Preparando os valores para plotagem
x_values = np.linspace(-2, 3, 400)
y_values = teste(x_values)

# Plotando a função de teste
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='f(x) = x^3 - x - 1', color='blue')

# Plotando a sequência de aproximações
approximations = [x0, (x0 + x1) / 2, x1]
plt.plot(approximations, [teste(approximation)
         for approximation in approximations], 'ro', label='Aproximações')

plt.title('Método de Herão e Função de Teste')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# Criando a tabela
data = [
    [n, x0, teste(x0), teste(x0) * teste((x0 + x1) / 2) < 0, np.format_float_scientific(
        (x0 + x1) / 2 - x0, precision=5), '{}, {}, {}'.format(x0, x1, epsilon)],
]

df = pd.DataFrame(
    data, columns=['n', 'x', 'f(x)', 'Valor de verdade', 'Erro em notação cientifica', 'Dados iniciais: x0, x1, epsilon (respectivamente)'])
print(df)
