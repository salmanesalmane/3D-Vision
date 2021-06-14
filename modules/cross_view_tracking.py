#------------ Cross-view Tracking functions ------------

def hungarian(A):
  # Input : 
  #         - A       : the NxM  affinity matrix between targets and detections
  # Output: 
  #         - row_ind and their corresponding col_ind
  row_ind, col_ind = linear_sum_assignment(A)
  return row_ind, col_ind


def affinity_2D (detection, target):
    #Inputs: 2 2D homogenized joints with corresponding frame numbers
    #        x1 is the current detected joint, x2 is the previously detected joint from the same camera
    #Outputs: 2D affinity measurement
    for i in reversed(target.prev_detec):
        if i.cam==detection.cam:
            prev_detection = i 
            break
    
    time_delta = abs(detection.t-prev_detection.t)*delta_t
    A_2D = 0.0
    for i in range(nb_keypoints):
        x1 = detection.x_2D[i]
        x2 = prev_detection.x_2D[i]
        A_2D += w_2D*(1-point_point_distance_2D(x1,x2)/(alpha_2d*time_delta))*math.exp(-lambda_a*time_delta)
    return A_2D

def affinity_3D (detection, target):
    #Inputs: X2 is the 3D homogenized joint location from the previous reconstruction
    #        x1 is the 2D homogenized joint detection
    #        P is the projection matrix of the current detection and C the camera loaction
    #Outputs: 3D affinity measurement
    time_delta = abs(detection.t-target.t)*delta_t

    #back-project the detected joint into 3-space as a parametrized line
    C  = camera_locations_dict[detection.cam]             #the direction vector of the line
    pinv_P = np.linalg.pinv(detection.P)
    
                           #the point on the line

    #predict joint location
    ##estimate velocity using least-square method
    A_3D = 0.0 
    for i in range(nb_keypoints):
        X1 = P_inv @ detection.x_2D[i]
        velocity = (target.X[i]-target.X_prev[i])/((target.t - target.prev_detec[-1].t)*delta_t)
        X2_pred  = homogenize(dehomogenize_3D(target.X[i]) + velocity*time_delta)
        A_3D += w_3D*(1-line_point_distance_3D(C,X1,X2_pred)/alpha_3D)*math.exp(-lambda_a*time_delta)

    return A_3D


def affinity_matrix(detections, targets):
    #Inputs: detections is a list with all the current detections in this frame
    #detections = [detection1, detection2, ....]
    #targets = [target1, target2,...]

    m = len(detections)
    n = len(targets)
    A = np.zeros((n,m))
    for i in range(n):
        #iterate over all targets
        for j in range(m):
            #iterate over all detections
            A[i,j] =  affinity_2D(detections[j],targets[i])+affinity_3D(detections[j],targets[i])

    return A