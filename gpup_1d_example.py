import numpy as np
import scipy.stats
from gp import GaussianProcess
from gpup import UncertaintyPropagation
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPy

np.random.seed(145453)

# Defined function
def func(x, v = 0.3):
    return x*np.sin(x)+np.random.normal(0, v)

ax1 = plt.subplot2grid((3, 5), (0, 2), colspan=3, rowspan=2)

# Number of data points to generate
M = 20

# Generate noisy data
x_data = np.random.uniform(0.4, 4, M).reshape(-1,1) # Generate M data points between 0.4 and 4
y_data = np.array([func(i, 0.2) for i in x_data]) # Add Gaussian noise with std. of 0.2
plt.plot(x_data, y_data, '+k', label='data')

# Compute real function
x_real = np.linspace(0, 6, 100).reshape(-1, 1)
y_real = np.array([func(i, 0) for i in x_real])
plt.plot(x_real, y_real, '--k', label='true function')

# Initiate GP with training data
gp_est = GaussianProcess(x_data, y_data.reshape((-1,)), optimize = True, theta = None, algorithm = 'Matlab')
# Initiate GPUP with training data
gpup_est = UncertaintyPropagation(x_data, y_data.reshape((-1,)), optimize = True, theta = None, method = 3)

# Compute GP prediction for new data - as in the GP example
x_new = np.linspace(0, 6, 100).reshape(-1,1)
means = np.empty(100)
variances = np.empty(100)
for i in range(100):
    means[i], variances[i] = gp_est.predict(x_new[i])
msl = (means.reshape(1,-1)[0]-np.sqrt(variances))#.reshape(-1,1)
msu = (means.reshape(1,-1)[0]+np.sqrt(variances))#.reshape(-1,1)[0]
ax1.plot(x_new, means,'-r', label='prediction mean')
ax1.fill_between(x_new.reshape(1,-1)[0], msl, msu, label='prediction std.')
plt.legend()

# Generate input Gaussian distribution
x_n = np.array([3.0]) # The mean of a normal distribution
var_n = np.diag([0.2**2]) # The covariance matrix (must be diagonal because of lazy programming)

# Compute the GPUP output
m, s = gpup_est.predict(x_n, var_n)

# We sample N particles in the initial distribution to validate the GPUP computation
N = int(1e4)
X_belief = np.array([np.random.normal(x_n, np.sqrt(var_n)) for _ in range(N)]).reshape(N,1) #
ax4 = plt.subplot2grid((3, 5), (2, 2), colspan=3, rowspan=1)
plt.plot(X_belief, np.tile(0., N), '.k', label='input particles')
x = np.linspace(0, 6, 1000).reshape(-1,1)
plt.plot(x,scipy.stats.norm.pdf(x, x_n, np.sqrt(var_n)), label='input dist.')
plt.xlabel('x')
plt.legend()

# Propagate all particles through the GP to get an approximated distribution 
ax2 = plt.subplot2grid((3, 5), (0, 0), colspan=1, rowspan=2)
means_b, variances_b = gp_est.batch_predict(X_belief) # Use the GP batch prediction
variances_b = np.diag(variances_b)
Y_belief = np.array([np.random.normal(means_b[i], np.sqrt(variances_b[i])) for i in range(N)]).reshape(N,1) #
plt.plot(np.tile(0., N), Y_belief, '.k', label='output particles')
plt.ylabel('p(y)')

ylim = ax1.get_ylim()
mu_Y = np.mean(Y_belief)
sigma2_Y = np.std(Y_belief)
y = np.linspace(ylim[0], ylim[1], 100000).reshape(-1,1)
plt.plot(scipy.stats.norm.pdf(y, mu_Y, sigma2_Y), y, '-b', label='particles output dist.')
plt.plot(scipy.stats.norm.pdf(y, m, np.sqrt(s)), y, ':r', label='GPUP output dist.')
plt.legend()

ax3 = plt.subplot2grid((3, 5), (0, 1), rowspan=2)
plt.hist(means_b, bins=20, orientation='horizontal')

ax2.set_ylim(ylim)
ax3.set_ylim(ylim)

plt.show()