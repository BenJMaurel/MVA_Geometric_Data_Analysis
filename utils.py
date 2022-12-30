
import numpy as np
import matplotlib.pyplot as plt
import copy


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

def line(num_points = 100, a = np.array([0,0]), b = np.array([1,1]), noise = 0.02, scale = 1):
    v_dir = a-b
    line = []
    for _ in np.linspace(-0.5*scale,0.5*scale, num_points):
      line.append( _*v_dir + np.random.randn(2)*noise + (a+b)/2)
    return np.array(line)

def square(num_points = 100, noise = 0.01, scale = 1, center = np.array([0.5,0.5]), ploty = False):
  sommets = [[-0.5,0.5], [0.5,0.5], [0.5, -0.5], [-0.5, -0.5]]
  sommets = np.array(sommets) + center
  taille = num_points//len(sommets)
  all_points = np.zeros((2,taille*len(sommets)))
  for i in range(len(sommets)):
    all_points[:,taille*i:taille*(i+1)] = line(num_points = taille, a = sommets[i-1], b = sommets[i], scale = scale).T
  return all_points.T

def triangle(num_points = 100, noise = 0.01, scale = 1, center = np.array([0.5,0.5]), ploty = False):
  sommets = [[-0.5,0.5], [0.5,0.5], [0.5, -0.5]]
  sommets = np.array(sommets) + center
  taille = num_points//len(sommets)
  all_points = np.zeros((2,taille*len(sommets)))
  for i in range(len(sommets)):
    all_points[:,taille*i:taille*(i+1)] = line(num_points = taille, a = sommets[i-1], b = sommets[i], scale = scale).T
  return all_points.T


def transform(degre, center):
  angle = degre # in degrees

  # Calculate the rotation matrix for the given angle
  R = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

  T = np.eye(3)
  T[:2, :2] = R
  T[:2, 2] = center
  return T