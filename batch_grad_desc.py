from numpy import *
import time
import matplotlib.pyplot as plt

start_time = time.time()

# Import data
x = loadtxt('x.txt')
y = loadtxt('y.txt')
t = loadtxt('t.txt')

# Measure data
tlen = len(t)

t_array = [None] * tlen

for i in range(tlen):
    t0 = 1.0
    t1 = t[i]
    t2 = t[i] ** 2
    t3 = t[i] ** 3
    t_array[i] = [t0, t1, t2, t3]

t_wdth = len(t_array[0])

# Preprocess data
# Initialise preprocessing variables
mean = [0.0] * t_wdth
stdev = [0.0] * t_wdth
this_sum = [0.0] * t_wdth
diffs_square = [0.0] * t_wdth

# Normalise t
for j in range(t_wdth):
    # Calculate the mean of each column
    for i in range(tlen):
        this_sum[j] += t_array[i][j]
    mean[j] = this_sum[j] / tlen
    # Calculate the standard deviation of each column
    for i in range(tlen):
        diffs_square[j] += (abs(mean[j] - t_array[i][j]))**2
    stdev[j] = sqrt(diffs_square[j] / tlen)
    # Normalise data
    for i in range(tlen):
        if stdev[j] != 0:
            t_array[i][j] = (t_array[i][j] - mean[j]) / stdev[j]
        else:
            t_array[i][j] = 1.0

# Initialise variables for gradient descent
# Theta is the array of coefficients
theta_x = [1.0] * t_wdth
theta_y = [1.0] * t_wdth
# Alpha is the learning rate
alpha = 0.6
# Array to hold error values
E_x = []
E_y = []
# Max iterations
n = 100000
# Point at which to stop
gamma = 0.0001

# Run first for x
for iteration in range(n):
    this_E = 0.0
    dE = [0.0] * t_wdth
    for i in range(tlen):
        fti = 0.0
        for j in range(t_wdth):
            fti += (t_array[i][j] * theta_x[j])
        this_E += ((1.0/(2.0*tlen)) * ((fti - x[i])**2))
        for j in range(t_wdth):
            dE[j] += (1.0/tlen) * (t_array[i][j] * (fti - x[i]))
    E_x.append(this_E)
    #print('Error x: ' + str(this_E))
    for j in range(t_wdth):
        theta_x[j] -= dE[j] * alpha
    converged = True
    for j in range(t_wdth):
        if abs(dE[j]) > gamma:
            converged = False
    if converged == True:
        print('x converged in ' + str(iteration) + ' iterations')
        break

# Then run for y
for iteration in range(n):
    this_E = 0.0
    dE = [0.0] * t_wdth
    for i in range(tlen):
        fti = 0.0
        for j in range(t_wdth):
            fti += (t_array[i][j] * theta_y[j])
        this_E += ((1.0/(2.0*tlen)) * ((fti - y[i])**2))
        for j in range(t_wdth):
            dE[j] += (1.0/tlen) * (t_array[i][j] * (fti - y[i]))
    E_y.append(this_E)
    #print('Error y: ' + str(this_E))
    for j in range(t_wdth):
        theta_y[j] -= dE[j] * alpha
    converged = True
    for j in range(t_wdth):
        if abs(dE[j]) > gamma:
            converged = False
    if converged == True:
        print('y converged in ' + str(iteration) + ' iterations')
        break

test_t = 0.4
test_x = 0.0
test_y = 0.0
test_t_array = []
for j in range(t_wdth):
    t0 = 1.0
    t1 = test_t
    t2 = test_t ** 2
    t3 = test_t ** 3
    test_t_array = [t0, t1, t2, t3]
for j in range(t_wdth):
    if stdev[j] != 0:
        test_t_array[j] = (test_t_array[j] - mean[j]) / stdev[j]
    else:
        test_t_array[j] = 1.0
    test_x += test_t_array[j] * theta_x[j]
    test_y += test_t_array[j] * theta_y[j]

actual_x = (2*exp(5*test_t))+exp(-5*test_t)
actual_y = exp(5*test_t)+(2*exp(-5*test_t))

diff_x = abs(actual_x - test_x)
diff_y = abs(actual_y - test_y)
print('')
print('predicted x: ' + str(test_x))
print('predicted y: ' + str(test_y))
print('')
print('actual x: ' + str(actual_x))
print('actual y: ' + str(actual_y))
print('')
print('x error: ' + str(diff_x))
print('y error: ' + str(diff_y))
print('')
print('x equation: ' + str(theta_x[0]) + ' + ' + str(theta_x[1]) + '*T + ' + str(theta_x[2]) + '*T^2 + ' + str(theta_x[3]) + '*T^3')
print('y equation: ' + str(theta_y[0]) + ' + ' + str(theta_y[1]) + '*T + ' + str(theta_y[2]) + '*T^2 + ' + str(theta_y[3]) + '*T^3')
print('')
print('T = (t - ' + str(mean[1]) + ') / ' + str(stdev[1]))
print('T^2 = (t^2 - ' + str(mean[2]) + ') / ' + str(stdev[2]))
print('T^3 = (t^3 - ' + str(mean[3]) + ') / ' + str(stdev[3]))
print('')
print('time elapsed: ' + str(time.time() - start_time) + 's')

dt = 0.001
T = 0.2
iters = int(T/dt)
graph_x = [0.0] * iters
graph_y = [0.0] * iters
for i in range(iters):
    current_t = dt*i
    current_t_array = []
    for j in range(t_wdth):
        t0 = 1.0
        t1 = current_t
        t2 = current_t ** 2
        t3 = current_t ** 3
        current_t_array = [t0, t1, t2, t3]
    for j in range(t_wdth):
        if stdev[j] != 0:
            current_t_array[j] = (current_t_array[j] - mean[j]) / stdev[j]
        else:
            current_t_array[j] = 1.0
        graph_x[i] += current_t_array[j] * theta_x[j]
        graph_y[i] += current_t_array[j] * theta_y[j]

plt.plot(graph_x, graph_y)
plt.title('Predicted x(t) against y(t) for t=0:0.2s')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.show()
