import copy
import numpy as np
import open3d as o3
from probreg import cpd

# load source and target point cloud
source = o3.io.read_point_cloud('/home/mbenj/3A/3A/MVA/Geometrie/probreg/examples/bunny.pcd')
source.remove_non_finite_points()
target = copy.deepcopy(source)
# transform target point cloud
th = np.deg2rad(30.0)
target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]))
source = source.voxel_down_sample(voxel_size=0.005)
target = target.voxel_down_sample(voxel_size=0.005)
# import pdb; pdb.set_trace()
# compute cpd registration
tf_param, _, _ = cpd.registration_cpd(source, target)
result_cpd = copy.deepcopy(source)
result_cpd.points = tf_param.transform(result_cpd.points)

# draw result
# source.paint_uniform_color([1, 0, 0])
# target.paint_uniform_color([0, 1, 0])
# result.paint_uniform_color([0, 0, 1])
# o3.visualization.draw_geometries([source, target, result])

x = np.asarray(source.points).T
y = np.asarray(target.points).T
N = [x.shape[1], y.shape[1]]
x2 = np.sum(x**2,0)
y2 = np.sum(y**2,0)

# import pdb; pdb.set_trace()
C = np.tile(y2,(N[0],1)) + np.tile(x2[:,np.newaxis],(1,N[1])) - 2*np.dot(np.transpose(x),y)


a = np.ones(N[0])/N[0]
b = np.ones(N[1])/N[1]
epsilon = .5
K = np.exp(-C/epsilon)
v = np.ones(N[1])
niter = 5000
Err_p = []
Err_q = []
for i in range(niter):
    # sinkhorn step 1
    u = a / (np.dot(K,v))
    # error computation
    r = v*np.dot(np.transpose(K),u)
    Err_q = Err_q + [np.linalg.norm(r - b, 1)]
    # sinkhorn step 2
    v = b /(np.dot(np.transpose(K),u))
    s = u*np.dot(K,v)
    Err_p = Err_p + [np.linalg.norm(s - a,1)]


P = np.dot(np.dot(np.diag(u),K),np.diag(v))
P = P*(P>np.max(P)*.2)
V = np.divide(np.dot(y,P.T) - (x*np.sum(P, axis = 1)), np.sum(P, axis = 1))
W = np.sum(P, axis = 1)
v_c = np.sum(W*V, axis = 1)/np.sum(W)
x_c = np.sum(W*x, axis = 1)/np.sum(W)
X = x.T - x_c.T
Y = x.T + V.T - x_c.T - v_c
FF = np.dot(X.T, np.dot(np.diag(W),Y))
U, _ , S = np.linalg.svd(FF)

# import pdb; pdb.set_trace()
new_x = np.dot((x.T - x_c.T),np.dot(U,S)) + x_c.T + v_c
new_x_2 = Y + x_c.T + v_c

result = o3.geometry.PointCloud()
result.points = o3.utility.Vector3dVector(new_x_2)
result_down = result.random_down_sample( N[1]/N[0])

# err = np.linalg.norm(np.asarray(result_down.points) - np.asarray(target.points))

import pdb; pdb.set_trace()

source.paint_uniform_color([1, 0, 0])

# result_cpd.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([source, target, result])
