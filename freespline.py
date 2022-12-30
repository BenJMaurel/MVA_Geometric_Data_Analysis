import numpy as np
import matplotlib.pyplot as plt
import copy
import numpy as np
import open3d as o3
from probreg import cpd
from pycpd import RigidRegistration

def ICP(point_cloud_1, point_cloud_2, max_iterations=5, tolerance=1e-2):
  # Initialize the transformation matrix to the identity matrix
  transform = np.eye(3)
  
  # Initialize the difference between the point clouds to be very large
  difference = np.inf
  
  transform_tot = []
  # Iterate until the difference between the point clouds is below the tolerance or
  # the maximum number of iterations is reached
  for i in range(max_iterations):
    # Find the closest points in point_cloud_1 to point_cloud_2
    closest_points_1, closest_points_2 = find_closest_points(point_cloud_1, point_cloud_2)
    
    # Calculate the transformation matrix that aligns closest_points_1 to closest_points_2
    transform = calculate_transform(closest_points_1, closest_points_2)
    
    # Transform point_cloud_1 using the transformation matrix
    point_cloud_1 = transform_point_cloud(point_cloud_1, transform)
    
    # Calculate the difference between the transformed point_cloud_1 and point_cloud_2
    new_difference = calculate_difference(point_cloud_1, point_cloud_2)
    # If the difference is below the tolerance, stop the iteration
    print(difference)
    transform_tot.append(transform)
    # if i%10 == 0 or i%10!=0:
    #     plt.scatter(point_cloud_1[:, 0], point_cloud_1[:, 1], label='Aligned moving point cloud')
    
    if new_difference < tolerance or difference < new_difference:
      return transform_tot
    difference = new_difference
  return transform_tot


def transform_point_cloud(point_cloud, transform):
  # Convert the point cloud to a Numpy array
  point_cloud = np.array(point_cloud)
  
  # Add a column of ones to the point cloud to allow for translation
  ones = np.ones((point_cloud.shape[0], 1))
  point_cloud_homo = np.hstack((point_cloud, ones))
  
  # Transform the point cloud using the transformation matrix
  transformed_point_cloud = np.dot(point_cloud_homo, transform.T)
  
  # Remove the extra column of ones
  transformed_point_cloud = transformed_point_cloud[:, :2]
  # transformed_point_cloud = point_cloud + transform[:2,2]
  # transformed_point_cloud = transformed_point_cloud@transform[:2,:2]

  
  return transformed_point_cloud

def calculate_difference(point_cloud_1, point_cloud_2):
  # Initialize the difference to 0
  difference = 0
  
  # Iterate through each point in point_cloud_1
  for point_1 in point_cloud_1:
    # Initialize the minimum distance to a very large value
    min_distance = np.inf
    
    # Iterate through each point in point_cloud_2
    for point_2 in point_cloud_2:
      # Calculate the distance between point_1 and point_2
      distance = np.linalg.norm(point_1 - point_2)
      
      # If the distance is smaller than the current minimum distance, update the minimum distance
      if distance < min_distance:
        min_distance = distance
    
    # Add the minimum distance to the difference
    difference += min_distance
  
  # Divide the difference by the number of points in point_cloud_1 to get the average distance
  difference /= len(point_cloud_1)
  
  return difference


def find_closest_points(point_cloud_1, point_cloud_2):
  closest_points_1 = []
  closest_points_2 = []
  
  # Iterate through each point in point_cloud_1
  for point_1 in point_cloud_1:
    # Initialize the minimum distance to a very large value
    min_distance = np.inf
    
    # Initialize the closest point in point_cloud_2 to None
    closest_point_2 = None
    
    # Iterate through each point in point_cloud_2
    for point_2 in point_cloud_2:
      # Calculate the distance between point_1 and point_2
      distance = np.linalg.norm(point_1 - point_2)
      
      # If the distance is smaller than the current minimum distance, update the minimum distance
      # and set the closest point to point_2
      if distance < min_distance:
        min_distance = distance
        closest_point_2 = point_2
    
    # Add the closest point in point_cloud_2 to the list of closest points
    closest_points_2.append(closest_point_2)
    
    # Add the point in point_cloud_1 to the list of closest points
    closest_points_1.append(point_1)
  
  return closest_points_1, closest_points_2

def calculate_transform(points_1, points_2):
  # Convert the lists of points to Numpy arrays
  points_1 = np.array(points_1)
  points_2 = np.array(points_2)
  
  # Calculate the centroid of each set of points
  centroid_1 = np.mean(points_1, axis=0)
  centroid_2 = np.mean(points_2, axis=0)
  
  # Subtract the centroids from each set of points
  points_1_centered = points_1 - centroid_1
  points_2_centered = points_2 - centroid_2
  
  # Calculate the covariance matrix of the centered points
  covariance_matrix = np.dot(points_1_centered.T, points_2_centered)
  
  # Calculate the singular value decomposition of the covariance matrix
  U, S, Vt = np.linalg.svd(covariance_matrix)
  
  # Calculate the rotation matrix from the SVD
  R = np.dot(U, Vt)
  
  # Calculate the translation vector
  t = centroid_2 - np.dot(centroid_1, R)
  
  # Create the transformation matrix from the rotation matrix and translation vector
  transform = np.eye(3)
  transform[:2, :2] = R
  transform[:2, 2] = t
  
  return transform

def transform(degre, center):
  angle = degre # in degrees

  # Calculate the rotation matrix for the given angle
  R = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

  T = np.eye(3)
  T[:2, :2] = R
  T[:2, 2] = center
  return T


def line(num_points = 100, a = np.array([0,0]), b = np.array([1,1]), noise = 0.02, scale = 1):
    v_dir = a-b
    line = []
    for _ in np.linspace(-0.5*scale,0.5*scale, num_points):
      line.append( _*v_dir + np.random.randn(2)*noise + (a+b)/2)
    return np.array(line)

def square_2(num_points = 100, noise = 0.01, scale = 1, center = np.array([0.5,0.5]), ploty = False):
  sommets = [[-0.5,0.5], [0.5,0.5], [0.5, -0.5], [-0.5, -0.5]]
  sommets = np.array(sommets) + center
  taille = num_points//len(sommets)
  all_points = np.zeros((2,taille*len(sommets)))
  for i in range(len(sommets)):
    all_points[:,taille*i:taille*(i+1)] = line(num_points = taille, a = sommets[i-1], b = sommets[i], scale = scale).T
  return all_points.T


def circle(num_points = 100, scale = 1., noise = 0.02, center = 0.5, ploty = False):

    # Generate the points that form the circle
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Scale the points to the desired size
    scaling_factor = scale
    points *= scaling_factor

    # Translate the points to the desired position
    translation = np.array([0.5, 0.5])
    points += translation

    # Add noise to the points to create a more realistic point cloud
    noise = np.random.normal(scale= noise, size=points.shape)
    points += noise

    # Plot the points
    if ploty:
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()

    return points


def square_full(num_points = 1000, scale = 1, center = np.array([0.5,0.5]), ploty = False):
    x = np.random.rand(2,num_points).T*scale - center
    return x

def null():
  # Define the fixed point cloud
  fixed_points = square_2(scale = 1, center = np.array([0,0]))
  # fixed_points = line(num_points = 150)
  # fixed_points = square_full()

  # Define the moving point cloud
  moving_points = square_2(num_points = 150, scale = 1, center = np.array([0,0]))
  # moving_points = line()
  # moving_points = square_full(num_points = 1500)

  T = transform(30, [2,3])
  T_2 = transform(0, [0,0])
  fixed_points = transform_point_cloud(fixed_points, T_2)
  moving_points = transform_point_cloud(moving_points, T)

  # Initialize the moving point cloud as the initial guess for the aligned point cloud
  aligned_points = moving_points
  # plt.scatter(moving_points[:, 0], moving_points[:, 1], alpha = 0.4, label='Started point cloud')
  # plt.scatter(fixed_points[:, 0], fixed_points[:, 1], alpha = 0.4, label='Target point cloud')

  # transform_final = ICP(moving_points, fixed_points)
  # aligned_points_icp = aligned_points
  # for _transform in transform_final:
    # _transform[2,:2] = _transform[:2,2]
    # aligned_points_icp = transform_point_cloud(aligned_points_icp, _transform)


  reg = RigidRegistration(Y=moving_points, X=fixed_points)
  # run the registration & collect the results
  aligned_points_icp, (s_reg, R_reg, t_reg) = reg.register()
  # import pdb; pdb.set_trace()

  plt.scatter(aligned_points_icp[:, 0], aligned_points_icp[:, 1], alpha = 0.4, label='ICP Last moving point cloud')
  # plt.legend()
  # plt.axis('equal')
  # plt.show()

####Optimal Transport Part:
def opt():
  y = fixed_points.T
  x = moving_points.T
  N = [moving_points.shape[0],fixed_points.shape[0]]
  epsilon = 0.5
  x2 = np.sum(x**2,0)
  y2 = np.sum(y**2,0)
  C = np.tile(y2,(N[0],1)) + np.tile(x2[:,np.newaxis],(1,N[1])) - 2*np.dot(np.transpose(x),y)
  a = np.ones(N[0])/N[0]
  b = np.ones(N[1])/N[1]


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

  def plot_P():
    plt.figure(figsize = (10,10))
    glist = [.8,.5,.3,.1]
    niter = 300
    clamp = lambda x,a,b: min(max(x,a),b)
    for k in range(len(glist)):
        epsilon = glist[k]
        K = np.exp(-C/epsilon)
        v = np.ones(N[1])
        for i in range(niter):
            u = a / (np.dot(K,v))
            v = b /(np.dot(np.transpose(K),u))
        P = np.dot(np.dot(np.diag(u),K),np.diag(v))
        #imageplot(clamp(Pi,0,np.min(1/np.asarray(N))*.3),"$\gamma=$ %.3f" %gamma, [2,2,k+1])
        plt.subplot(2,2,k+1)
        plt.imshow(np.clip(P,0,np.min(1/np.asarray(N))*.3));
        #"$\gamma=$ %.3f" %gamma, [2,2,k+1])
    plt.show()

  P = np.dot(np.dot(np.diag(u),K),np.diag(v))
  # P = P*(P>np.max(P)*.8)
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
  plt.scatter(new_x[:, 0], new_x[:, 1], alpha = 0.4, label='Last moving point cloud')
  plt.legend()
  plt.axis('equal')
  plt.show()
  import pdb; pdb.set_trace()