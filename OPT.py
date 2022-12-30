import numpy as np
import matplotlib.pyplot as plt
import copy
import numpy as np

from utils import transform_point_cloud, transform, square_2, line

n = int(120)
m = int(80)

x = square_2(num_points = n).T
y = transform_point_cloud(square_2(num_points = m), transform(30, np.array([1,1]))).T
# x = line(num_points = n).T
# y = transform_point_cloud(line(num_points = m), transform(30, np.array([1,1]))).T
n = x.shape[1]
m = y.shape[1]
x2 = np.sum(x**2,0)
y2 = np.sum(y**2,0)
C = np.tile(y2,(n,1)) + np.tile(x2[:,np.newaxis],(1,m)) - 2*np.dot(x.T,y)
a = np.ones(n)/n
b = np.ones(m)/m

def mina_u(H,epsilon): return -epsilon*np.log( np.sum(a[:,None] * np.exp(-H/epsilon),0) )
def minb_u(H,epsilon): return -epsilon*np.log( np.sum(b[None,:] * np.exp(-H/epsilon),1) )

def mina(H,epsilon): return mina_u(H-np.min(H,0),epsilon) + np.min(H,0)
def minb(H,epsilon): return minb_u(H-np.min(H,1)[:,None],epsilon) + np.min(H,1)

epsilon = .005
rho = 10000
kappa = rho/(rho+epsilon)
# import pdb; pdb.set_trace()
f = np.zeros(n)
niter = 1000
for it in range(niter):
    g = kappa*mina(C-f[:,None],epsilon)
    f = kappa*minb(C-g[None,:],epsilon)
# generate the coupling
P = a[:,None] * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b[None,:]

rho = 50
def plot_P():
    plt.figure(figsize = (10,10))
    glist = [.3,.1,.05,.01]
    niter = 300
    clamp = lambda x,a,b: min(max(x,a),b)
    for k in range(len(glist)):
        epsilon = glist[k]
        kappa = rho/(rho+epsilon)
        # import pdb; pdb.set_trace()
        f = np.zeros(n)
        niter = 1000
        for it in range(niter):
            g = kappa*mina(C-f[:,None],epsilon)
            f = kappa*minb(C-g[None,:],epsilon)
        # generate the coupling
        P = a[:,None] * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b[None,:]
        #imageplot(clamp(Pi,0,np.min(1/np.asarray(N))*.3),"$\gamma=$ %.3f" %gamma, [2,2,k+1])
        plt.subplot(2,2,k+1)
        plt.imshow(np.clip(P,0,np.min(1/np.asarray([n,m]))*.3));
        #"$\gamma=$ %.3f" %gamma, [2,2,k+1])
    plt.show()

V = np.divide(np.dot(y,P.T) - (x*np.sum(P, axis = 1)), np.sum(P, axis = 1))
W = np.sum(P, axis = 1)
v_c = np.sum(W*V, axis = 1)/np.sum(W)
x_c = np.sum(W*x, axis = 1)/np.sum(W)
X = x.T - x_c.T
Y = x.T + V.T - x_c.T - v_c
FF = np.dot(X.T, np.dot(np.diag(W),Y))
U, _ , S = np.linalg.svd(FF)

new_x = np.dot((x.T - x_c.T),np.dot(U,S.T)) + x_c.T + v_c
# other_x = np.dot(np.dot((x.T -x_c.T), np.linalg.inv(np.dot(X.T@np.diag(W), X))), np.dot(X.T@np.diag(W), Y)) + x_c.T + v_c

# plt.scatter(other_x[:, 0], other_x[:, 1], alpha = 0.4, label='Last_o moving point cloud')
plt.scatter(x[0,:], x[1,:], alpha = 0.4, label = "source")
plt.scatter(y[0,:], y[1,:], alpha = 0.4, label = "target")
plt.scatter(new_x[:, 0], new_x[:, 1], alpha = 0.4, label='Last moving point cloud')

# plt.show()

import numpy as np

def gaussian_kernel(x, h):
    """
    Gaussian kernel function.
    """
    return np.exp(-np.sum(x**2, axis = 1) / (2*h**2)) / (np.sqrt(2*np.pi*h**2))

def epanechnikov_kernel(x, h):
    """
    Epanechnikov kernel function.
    """
    return np.maximum(1 - x**2 / h**2, 0) / (np.pi * h**2)

def spline(x, x_data, v, w,  h, kernel_func):
    """
    Nadaraya-Watson spline function.
    """
    
    # Compute the weights using the kernel function
    weights = w*kernel_func(x - x_data.T, h)
    
    # Compute the spline function using the weighted data points
    spline = x + np.sum(w*kernel_func(x - x_data.T,h)*v, axis = 1) / np.sum(weights)
    
    return spline

h = 0.5
# Use the Gaussian kernel
z = np.zeros(x.shape)
for i, _x in enumerate(x.T):
    z[:,i] = spline(_x, x, V, W, h, gaussian_kernel)
plt.scatter(z[0,:], z[1,:], alpha = 0.4, label = 'test')
# test = x + v
# plt.scatter(test[0,:], test[1,:], alpha = 0.4, label = 'euh')
plt.legend()
plt.axis('equal')
plt.show()

import pdb; pdb.set_trace()
# Use the Epanechnikov kernel
# spline_epanechnikov = spline(x, x_data, y_data, h, epanechnikov_kernel)
# print(f"Spline with Epanechnikov kernel: {spline_epanechnikov}")