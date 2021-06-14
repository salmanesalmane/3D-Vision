import numpy as np 

#------------ Basic functions ------------

def homogenize(x):
    return np.append(x,1)

def dehomogenize_2D(x):
    return (x/x[2])[0:2]

def dehomogenize_3D(x):  #TODO: group the two dehomogenize function in only one 
    return (x/x[3])[:3]

def line_point_distance_2D(l,x): 
    #line to point distance with 2D homogeneous coordinates
    x        = x/x[2]
    distance = np.dot(l, x)/(np.sqrt(l[0]**2+l[1]**2))
    return distance

def point_point_distance_2D(x1,x2):
    #point to point distance with 2D homogeneous coordinates
    x1       = x1/x1[2]
    x2       = x2/x2[2]
    distance = np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
    return distance

def point_point_distance_3D(x1,x2):
    #point to point distance with 3D homogeneous coordinates
    x1       = dehomogenize_3D(x1)
    x2       = dehomogenize_3D(x2)
    distance = np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2+(x1[2]-x2[2])**2)
    return distance

def line_point_distance_3D(v,y,x):
    #line to point distance with 3D homogeneous coordinates
    #we have the parametric representaion for the line
    #v is the direction vector and y is the point on the line
    #x is the point we want to measure the distance
    #https://onlinemschool.com/math/library/analytic_geometry/p_line/
    def f(mu):
        l = y + mu*v
        return point_point_distance_3D(l,x)
    
    distance = fmin(f,0.0)
    return distance