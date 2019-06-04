import numpy as np
from gp import GaussianProcess
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPy

# Defined function
def func(x, v = 0.3):
    return x*np.sin(x)+np.random.normal(0, v)

# Number of data points to generate
N = 20

# Generate noisy data
x_data = np.random.uniform(0.4, 4, N).reshape(-1,1) # Generate N data points between 0.4 and 4
y_data = np.array([func(i, 0.2) for i in x_data]) # Add Gaussian noise with std. of 0.2

# Compute real function
x_real = np.linspace(0, 6, 100).reshape(-1,1)
y_real = np.array([func(i, 0) for i in x_real]) 

# Initiate GP with training data
gp_est = GaussianProcess(x_data, y_data.reshape((-1,)), optimize = True, theta=None, algorithm = 'Girard')

# Compute prediction for new data
x_new = np.linspace(0, 6, 100).reshape(-1,1)
means = np.empty(100)
variances = np.empty(100)
for i in range(100):
    means[i], variances[i] = gp_est.predict(x_new[i])

# Plot
plt.plot(x_data, y_data, '+k', label='data')
plt.plot(x_real, y_real, '--k', label='true function')
msl = (means.reshape(1,-1)[0]-np.sqrt(variances))#.reshape(-1,1)
msu = (means.reshape(1,-1)[0]+np.sqrt(variances))#.reshape(-1,1)[0]
plt.plot(x_new, means,'-r', label='prediction mean')
plt.fill_between(x_new.reshape(1,-1)[0], msl, msu, label='prediction std.')
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title('GP')
plt.legend()

plt.show()