import numpy as np
import math
image_height, image_width = 1080,1920

#------------ Camera Calibration ------------

def Rx(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    A = np.array([[1, 0, 0],
                  [0, rcos, -rsin],
                  [0, rsin, rcos]])
    return A


def Ry(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)

    A = np.array([[rcos, 0, -rsin],
                  [0, 1, 0],
                  [rsin, 0, rcos]])
    return A

def get_camera_properties(data,frame):

    cx,cy = image_width/2., image_height/2.

    theta, phi, f, Cx, Cy, Cz = data[frame]
    R = Rx(phi).dot(Ry(theta).dot(np.array([[1,0,0],[0,-1,0],[0,0,-1]])))
    T = -R.dot(np.array([[Cx], [Cy], [Cz]]))
    K = np.eye(3, 3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, cx, cy
    return K, T, R

def compute_projection_matrix(K,T,R):
    P = K @ np.append(R,T, axis=1)
    return P

def compute_fundamental_matrix(P1,P2,C1):
    P1_pinv = np.linalg.pinv(P1)
    e2      = P2 @ C1
    e2_x    = np.array([[0, -e2[2], e2[1]],
                        [e2[2], 0, -e2[0]],
                        [-e2[1], e2[0], 0]])
    F       = e2_x @ P2 @ P1_pinv

    # -To deal with the case where the fundamental matrix is badly conditioned,
    #  we add a small term to avoid problems in the calculations.

    if np.abs(F).sum() == 0 :
        F += (1e-2)*np.identity(3)
    return F