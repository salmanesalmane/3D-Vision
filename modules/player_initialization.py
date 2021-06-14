import numpy as np
from pulp import *
from camera_calibration import *
from utils import *
from cross_view_tracking import *
from target_detection_classes import *
cameras = ["CAM1", "RIGHT", "LEFT"]
#-------- Parameters --------
alpha_2D     = 6e-6 
lambda_a     = 2.0 
w_2D         = 0.4 
alpha_3D     = 1e-5 
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


#------------ Player Initialization ------------

def epipolar_affinity(D1,D2,F):
    #D1 and D2: list of 25 homogenized keypoint locations for one person from one camera
    #F :  fundamental matrix of the pair of cameras (C1,C2), meaning l2=F*x1 and l1=transpose(F)*x2
    F2 = np.transpose(F)
    total = 0.
    for i in range(nb_keypoints):
        x1 = np.array(D1[i])
        x2 = np.array(D2[i])
        l1 = F2 @ x2
        l2 = F @ x1
        d1 = D1[i]
        d2 = D2[i]
        measurement_per_keypoint = 1. - ((line_point_distance_2D(d1,l1)+line_point_distance_2D(d2,l2))/(2.*alpha_2D))
        total += measurement_per_keypoint
    return total


def initialisation_affinity_matrix(unmatched_detections):
    #unmatched_detections: list with all the unmatched detections including camera and frame
    #unmatched_detections = [frame,'camera',D1],
    #                        frame,'camera',D2],...]
    #where D1=[[x1,y2,1],[x2,y2,1],...] list of 25 lists: the pair of coordinates of each keypoint 
    n = len(unmatched_detections)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Di = unmatched_detections[i].x_2D
            Dj = unmatched_detections[j].x_2D
            Ci = unmatched_detections[i].cam
            Cj = unmatched_detections[j].cam
            if Ci==Cj:
                #A[i,j] = -math.inf
                A[i,j] = -1e10
            else:
                #Pi, Pj, C = unmatched_detections[i].P, unmatched_detections[j].P, camera_locations[i]
                Pi, Pj, C = unmatched_detections[i].P, unmatched_detections[j].P, camera_locations_dict[unmatched_detections[i].cam]
                F = compute_fundamental_matrix(Pi,Pj,C)
                A[i,j] = epipolar_affinity(Di,Dj,F)
    return A

def target_initialization(A, detections):
  cam_set_end = []
  camera_locations = []
  unmatched_detections = []

  nb_detections = len(detections)
  for i in range(nb_detections):
    if i > 1 and detections[i].cam != detections[i-1].cam:
      cam_set_end.append(i)
    unmatched_detections.append(detections[i])
  print(cam_set_end)
    #projection_matrices.append(projection_matrices_dict[detections[i].P])
    #camera_locations.append(camera_locations_dict[detections[i].cam])

  n1, n2 = cam_set_end[0], cam_set_end[1]
  n = len(A)
  prob = LpProblem("Matching_detections", LpMaximize)
  y = LpVariable.dicts("pair", [(i,j) for i in range(n) for j in range(n)] ,cat='Binary')

  #add function to maximize
  prob += lpSum([A[i,j] * y[(i,j)] for i in range(n) for j in range(n)])

  #add constraints
  for i in range(n):
    for j in range(n):
      prob += y[(i,j)] == y[(j,i)]

  for i in range(n1):
    for j in range(n1,n2):
      if j == i:
        continue
      else:
        for k in range(n2, n):
          if k == i or k == j:
            continue
          else:  
            prob += y[(i,j)]+y[(j,k)]-y[(i,k)]  <= 1
            prob += y[(i,k)]+y[(k,j)]-y[(i,j)]  <= 1
            prob += y[(j,i)]+y[(i,k)]-y[(j,k)]  <= 1

  prob.solve()

  B = np.zeros((n,n))
  for i in range(n):
      for j in range(n):
          B[i,j] = y[(i,j)].varValue
  return B



def GraphPartition(B):
  '''
    - This functions takes a matrix containing binary elements (1,0) whre 1 at position (i,j) indicates
      a match between detection i and j. It extracts the different partitions using the transitivity criteria 
      that is to say (if A is matched with B, and B is matched with C, then A is matched with C, and the partition contains all of A,B and C)
  '''
  n = len(B)
  total_list = []
  for i in range(n):
    if i in total_list:
      continue

    l = [i]
    for j in range(n):
      if j in total_list:
        continue
      if B[i, j] != 0:
        l.append(j)
        l = list(set(l))
        non_zero_elem = [k for k, e in enumerate(B[j, :]) if e != 0]
        for elem in non_zero_elem:
          l.append(elem)
          non_zero_elem_2 = [k for k, e in enumerate(B[elem, :]) if e != 0]
          for elem2 in non_zero_elem_2:
            l.append(elem2)
					
        l = list(set(l))
    total_list += l
    total_list = list(set(total_list))
    yield l