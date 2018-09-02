import numpy as np
import matplotlib.pyplot as plt


def sigmoid(a, b, c):
    """
    Return sigmoid function
    """
    return 1 / (1 + np.exp(-(a + b + c)))


a = np.random.randint(1, 4, size=4)
print(a)

b = np.random.randint(1, 4, size=4)
print(b)

c = np.random.randint(5)
print(c)


print(a)
print(b)
print(c)
#matrix = np.zeros((4, 4))
matrix = np.fromfunction(lambda i, j: sigmoid(a[i], b[j], c), (4, 4), dtype=float)
print(matrix)


print(a)
print(b)
print(c)
matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        matrix[i, j] = sigmoid(a[i], b[j], c)

print(matrix)

a_ = np.tile(a, (4, 1)).T
b_ = np.tile(b, (4, 1))
print(a_)
print(b_)

matrix = sigmoid(a_, b_, c)
print(matrix)


vsigmoid = np.vectorize(sigmoid)
matrix = vsigmoid(a, b, c)
print(matrix)

print(a[0])
print(b[0])

print(1 / (1 + np.exp(-(a[0] + b[0] + c))))


# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()
