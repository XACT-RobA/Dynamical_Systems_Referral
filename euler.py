import numpy as np
import matplotlib . pyplot as plt

def f(inputs):
    [x, y] = inputs
    return [(3*x) + (4*y), (4*x) - (3*y)]

x0 = 3.0
y0 = 3.0
dt = 0.001
T = 0.2
t = np.linspace(0 , T, int(T/dt )+1)

X = np.zeros(len(t))
Y = np.zeros(len(t))

X[0] = x0
Y[0] = y0

for i in xrange(1, len(t)):
    [temp_x, temp_y] = f([X[i-1], Y[i-1]])
    [X[i], Y[i]] = [X[i-1]+(temp_x*dt), Y[i-1]+(temp_y*dt)]

plt.plot(X,Y)
plt.title('x(t) against y(t) for t=1:0.2s')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.show()
