from numpy.linalg import pinv
import numpy as np
from camera_calibration import *
from utils import *
from player_initialization import *
from cross_view_tracking import *

nb_keypoints = 26 


#------------ Definitions of classes ------------

class Detection:
  def __init__(self, x_2D, camera, t, P):
    '''
    x_2D   : LIST OF #nb_keypoints ARRAYS containing the homogenized 2D coordinates
             of the detection (Player)
    camera : 
    t      : Time / nbr of frame
    P      : Projection matrix of the camera at the given time frame
    '''

    self.P = P
    self.cam = camera
    self.t = t
    self.x_2D = x_2D
    if type(self.x_2D) is not np.ndarray:
      self.x_2D = np.array(self.x_2D)
    self.C = self.calculate_C()

  def calculate_C(self):
    C = []
    global nb_keypoints
    for i in range(nb_keypoints):
      x_2D_l = np.reshape(self.x_2D[i], 3).tolist()
      x_2D_x = [[0, - x_2D_l[2], x_2D_l[1]], [x_2D_l[2], 0, -x_2D_l[0]], [-x_2D_l[1], x_2D_l[0], 0]]
      C_redundant = x_2D_x @ self.P
      C.append(np.reshape(C_redundant[1:3, :], (2, -1)))
    return C


class Target:
  def __init__(self, detection, cameras):
    '''
    Inputs : - detection : class detection
             - cameras   : list of all the used cameras

    Fields : - self.X and self.X_prev are LIST OF #nb_keypoints ARRAYS containing
               the homogenized 3D position of the given target
             - self.C is a LIST OF #nb_keypoints 2D ARRAYS containing the C (2x4) matrices
               for the different joints
    '''
    self.X = None
    self.X_prev = None
    self.t = detection.t
    self.latest_x_2D_cam = {cam : None for cam in cameras} # The last detection corresponding to each camera
    self.nb_updates = 0
    self.ID = 0
    self.prev_detec = []
    self.C = []
    self.Incremental3DReconstruction(detection) # We will store all of the detections
    

  def Incremental3DReconstruction(self, detection):
    self.t = detection.t
    self.nb_updates += 1
    self.prev_detec.append(detection)
    self.calculate_big_c_matrix(detection)
    self.latest_x_2D_cam[detection.cam] = detection.x_2D
    self.X_prev = self.X
    self.X = self.new_3D_pos()

  def calculate_big_c_matrix(self, detection):
    if self.nb_updates == 1:
      for i in range(nb_keypoints):
        self.C = detection.C
    else:
      for i in range(nb_keypoints):
        self.C[i] = np.append(self.C[i], detection.C[i], axis = 0)
  
  def new_3D_pos(self):
    new_pose = []
    for i in range(nb_keypoints):
      A = (self.C[i])[:, 0:3]
      b = -(self.C[i])[:, 3]
      new_pose.append(homogenize(pinv(A)@b))
    return new_pose