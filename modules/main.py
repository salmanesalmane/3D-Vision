#-------- Global Variables --------
cameras = ["CAM1", "RIGHT", "LEFT"]
videos_folder    = '/content/drive/MyDrive/3D Vision/ULSAN vs DUHAIL/'
images_path      = '/content/drive/MyDrive/3D Vision/calibrated_images/'
DETECTIONS_PATH  = './alphapose_detections/'
calibration_path = '/content/drive/MyDrive/3D Vision/ULSAN vs DUHAIL/calibration_results/0125-0135/'

import os
from os.path import exists, join, basename, splitext
from IPython.display import YouTubeVideo
import torch, torchvision
assert torch.__version__.startswith("1.8")
import PIL
from PIL import Image 
import itertools
from scipy.optimize import fmin
#import mip, pulp
# import some common libraries
import pandas as pd

from numpy.linalg import norm
from numpy.linalg import pinv
from scipy.optimize import linear_sum_assignment

import os, json, cv2, random, math


from camera_calibration import *
from utils import *
from player_initialization import *
from cross_view_tracking import *
from target_detection_classes import *

#-------- Parameters --------
alpha_2D     = 6e-6 #60.0
lambda_a     = 2.0 ##############
w_2D         = 0.4 ###############
alpha_3D     = 1e-5 #############
w_3D         = 0.6
fps          = 25.0
delta_t      = 1.0/fps
nb_frames    = 250
nb_keypoints = 26 
image_height, image_width = 1080,1920

#CAM1
data        = np.genfromtxt('./calibration_results/0125-0135/CAM1/calib.txt',delimiter=',',usecols=(1,2,3,4,5,6))
K1,T1,R1    = get_camera_properties(data,0)
P1          = compute_projection_matrix(K1,T1,R1)
C1          = homogenize(data[0,3:])

#LEFT
data        = np.genfromtxt('./calibration_results/0125-0135/LEFT/calib.txt',delimiter=',',usecols=(1,2,3,4,5,6))
K_l,T_l,R_l = get_camera_properties(data,0)
P_l         = compute_projection_matrix(K_l,T_l,R_l)
C_l         = homogenize(data[0,3:])

#RIGHT
data        = np.genfromtxt('./calibration_results/0125-0135/RIGHT/calib.txt',delimiter=',',usecols=(1,2,3,4,5,6))
K_r,T_r,R_r = get_camera_properties(data,0)
P_r         = compute_projection_matrix(K_r,T_r,R_r)
C_r         = homogenize(data[0,3:])

F = compute_fundamental_matrix(P1,P_r,C1)
F2 = compute_fundamental_matrix(P_r,P1,C_r)


#list of 12 projection matrices, one for each camera
projection_matrices_dict = {"CAM1":P1, "LEFT":P_l, "RIGHT":P_r}
#list of 12 camera locations vectors, one for each camera
camera_locations_dict = {"CAM1":C1, "LEFT":C_l, "RIGHT":C_r}
#------------PATHS----------

def read_datasets(path):
	detections = []
	data = []
	for cam in cameras:
		with open(path+cam+'.json','r') as json_file:
			data.append(json.load(json_file)) 
	for frame in range(nb_frames):
		for cam in range(len(data)):
			i = 0
			tmp = 'frame'+str(frame)+'.png'
			for i in range(len(data[cam])):
				if (data[cam][i]['image_id']==tmp and data[cam][i]['category_id']==1):
					homogenized_keypoints = []
					keypoints = data[cam][i]['keypoints']
					for k in range(0, 3*nb_keypoints, 3):
						homogenized_keypoints.append(homogenize([int(keypoints[k]),int(keypoints[k+1])]).tolist())

					detection_class = Detection(homogenized_keypoints, cameras[cam], frame, projection_matrices_dict[cameras[cam]])
					detections.append(detection_class)

	return detections

def algorithm1(list_new_detec, list_prev_targets, list_prev_unmatched_detec):
  list_new_targets = []
  M = len(list_new_detec)
  N = len(list_prev_targets)
  A = np.zeros((N, M))

  ''' Cross view Assiciation '''
  if list_prev_targets:
    A = affinity_matrix(list_new_detec, list_prev_targets)

    index_T, index_D = hungarian(A)

    ''' Target Update '''
    for targ in index_T:
      for detec in index_D:
        list_prev_targets[targ].Incremental3DReconstruction(detec)
        list_new_targets.append(list_prev_targets[targ])
  else:
    index_D = []
    index_T = []

  ''' Target Initialization '''
  for detec in range(M):
    if detec in index_D:
      continue

    list_prev_unmatched_detec.append(list_new_detec[detec])

  A_unmatch = initialisation_affinity_matrix(list_prev_unmatched_detec)
  #print(A_unmatch)
  B = target_initialization(A_unmatch, list_prev_unmatched_detec)

  total_clusters = []
  for cluster in GraphPartition(B):

    for element in cluster:
      total_clusters.append(element)

    tar = Target(list_prev_unmatched_detec[cluster[0]], cameras)
    nbr_detec_clust = len(cluster)
    for i in range(1, nbr_detec_clust):
      detec = cluster[i]
      tar.Incremental3DReconstruction(list_prev_unmatched_detec[detec])

  list_new_unmatched_detec = []
  for detec in list_prev_unmatched_detec:
    if detec not in total_clusters:
      list_new_unmatched_detec.append(detec)

  return list_new_targets, list_prev_unmatched_detec



if __name__ == "__main__":
	detections = read_datasets(DETECTIONS_PATH)
	l = []
	algorithm1(detections[73:153],l,  detections[1:73])