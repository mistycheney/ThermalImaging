import numpy as np
import numpy.linalg as linalg
from ThermalImager import config


def pick_vector_normal_to_vector(axis, u):
    proj_vector = project_to_plane(np.array([0,0,1]), axis, np.array([0,0,0]))
    print 'proj_vector', proj_vector 
    v0 = proj_vector/linalg.norm(proj_vector)
    theta = u*config.SAMPLE_NORMAL_ANGLE_DEVIATION*np.pi/180
    vrot = rodrigues_rotation(v0, axis, theta)
    return vrot

def sample_vector_normal_to_vector(axis, size):
    '''
    axis = [a,b,c], unit directional vector
    initial guess v0 is the closest to (0,0,1) 
    '''
    proj_vector = project_to_plane(np.array([0,0,1]), axis, np.array([0,0,0]))
    print 'proj_vector', proj_vector 
    v0 = proj_vector/linalg.norm(proj_vector)
#    v0 = np.array([axis[1],-axis[0],0], dtype=np.float)
#    thetas = np.random.random(size)*np.pi
    thetas = (1-2*np.random.random(size))*config.SAMPLE_NORMAL_ANGLE_DEVIATION*np.pi/180
#    import cPickle as pickle
#    pickle.dump((1-2*np.random.random(size)), open('sample_vectors.p','wb'))
    vrots = [rodrigues_rotation(v0, axis, sample_theta) for sample_theta in thetas]
    vrots = np.array(vrots, dtype=np.float)
    return vrots


def signed_distance_origin_to_plane_batch(plane_normal, plane_points):
    '''
    plane_points: n*3
    '''
    d = np.dot(plane_points, plane_normal)
    return d


def distance_origin_to_plane_batch(plane_normal, plane_points):
    '''
    plane_points: n*3
    '''
    d = -np.dot(plane_points, plane_normal)
    return abs(d)

def distance_origin_to_plane(plane_normal, plane_point):
    d = -np.dot(plane_normal, plane_point)
    return abs(d)

def is_on_same_side_as_with_projs_batch(target_points, line_direction, line_point,
                                         plane_normal, ref_point, target_projs):
    line_normal_on_plane = np.cross(plane_normal, line_direction)
    sign_target = np.dot(target_projs-line_point, line_normal_on_plane) > 0
    sign_ref = np.dot(line_normal_on_plane, ref_point-line_point) > 0
    return sign_target == sign_ref

def is_on_same_side_as_batch(target_points, line_direction, line_point, plane_normal, ref_point):
    line_normal_on_plane = np.cross(plane_normal, line_direction)
    target_projs = project_to_plane_batch(target_points, plane_normal, line_point)
    sign_target = np.dot(target_projs-line_point, line_normal_on_plane) > 0
    sign_ref = np.dot(line_normal_on_plane, ref_point-line_point) > 0
    return sign_target == sign_ref

def is_on_same_side_as(target_point, line_direction, line_point, plane_normal, ref_point):
    '''
    Test whether an in-plane point is on the same side of a planar line as the in-plane ref_point 
    '''
    line_normal_on_plane = np.cross(plane_normal, line_direction)
    target_proj = project_to_plane(target_point, plane_normal, line_point)
    sign_target = np.dot(line_normal_on_plane, target_proj-line_point) > 0
    sign_ref = np.dot(line_normal_on_plane, ref_point-line_point) > 0
    return sign_target == sign_ref

def adjust_normal_direction(plane_normal, plane_point):
    '''
    Make sure all normal vectors are pointing away from the origin
    '''
    if np.dot(plane_point, plane_normal) > 0:
        return plane_normal
    else:
        return -plane_normal

def project_to_plane_only_distance(target_point, plane_normal, plane_point):
    proj_len = np.dot(target_point - plane_point, plane_normal)
    return abs(proj_len)

def project_to_plane(target_point, plane_normal, plane_point):
    proj_len = np.dot(target_point-plane_point, plane_normal)
    proj = target_point - proj_len*plane_normal
    return proj

def find_boundingbox_project(target_points, plane_normal, plane_point, direction_vector):
    projs = project_to_plane_batch(target_points, plane_normal, plane_point)
    offsets = np.dot(projs - plane_point, direction_vector)
    endpoint_max = plane_point + offsets.max()*direction_vector
    endpoint_min = plane_point + offsets.min()*direction_vector
    line_normal_on_plane = np.cross(plane_normal, direction_vector)
    dists = abs(np.dot(projs - plane_point, line_normal_on_plane))
    h = dists.max()
    return endpoint_min, direction_vector, line_normal_on_plane, linalg.norm(endpoint_max-endpoint_min), h

    
def project_to_plane_only_distance_batch(target_points, plane_normal, plane_point):
    if target_points.size == 0:
        return np.nan
    else:
        proj_lens = np.dot(target_points - plane_point, plane_normal)
        return abs(proj_lens)

def project_to_plane_with_distance_batch(target_points, plane_normal, plane_point):
    '''
    target_points: n by 3
    '''
    proj_lens = np.dot(target_points - plane_point, plane_normal)
    projs = target_points - np.outer(proj_lens, plane_normal)
#    dists = np.sum(np.abs(target_points - projs)**2,axis=1)**(1./2)
    return projs, abs(proj_lens)

def project_to_plane_batch(target_points, plane_normal, plane_point):
    '''
    target_points: n by 3
    '''
    proj_lens = np.dot(target_points - plane_point, plane_normal)
    projs = target_points - np.outer(proj_lens, plane_normal)
    return projs

def rodrigues_rotation(v, axis, theta):
    vrot = v*np.cos(theta) + np.cross(axis, v)*np.sin(theta) + axis * np.dot(axis,v)*(1-np.cos(theta))
    return vrot

def plane_to_hessian_normal_form(plane_normal, plane_points):
    '''
    plane_points: n by 3
    '''
    d = -np.dot(plane_normal, plane_points)
    return np.append(plane_normal, d)

def point_to_plane_distance(target_points, plane_normal, plane_point):
    '''
    target_points: n by 3
    '''
    d = -np.dot(plane_point, plane_normal)
    dists = abs(np.dot(target_points, plane_normal)+d)/linalg.norm(plane_normal)
    return dists

def fit_plane_indices(p_arr, indices):
    selected_points = p_arr[indices,:]
    xs = selected_points[:,0]
    ys = selected_points[:,1]
    zs = selected_points[:,2]
    A = np.column_stack((xs, ys, np.ones(xs.size)))
    plane_params, resid,rank,sigma = np.linalg.lstsq(A,zs)
    normal = np.array([plane_params[0],plane_params[1],-1])
    normal = normal/linalg.norm(normal)
    return plane_params, normal

def PCA_line_regression(points):
    
    mean = np.mean(points, axis=0)
    std =  np.std(points, axis=0)
    points_normalized = (points - mean) / (0.000001+std)
    
    import sys
    try:
        U,s,Vt = linalg.svd(points_normalized)
    except linalg.LinAlgError as e:
        print e
        print points
        sys.exit()
    V = Vt.T
    ind = np.argsort(s)[::-1]
    U = U[:,ind]
    s = s[ind]
    V = V[:,ind]
    S = np.diag(s)
    projected_points = np.dot(U[:,:1],np.dot(S[:1,:1],V[:,:1].T)) * std + mean
    residual = np.sqrt(np.sum((points - projected_points)**2,axis=1))
    
    projected_points = projected_points[projected_points[:,2].argsort()] 
    diff = projected_points[-1] - projected_points[0]
    line_direction = diff / linalg.norm(diff)
    center_point = (projected_points[0] + projected_points[-1])/2
    regress_line = np.hstack((center_point, line_direction, projected_points[0], projected_points[-1]))
    return regress_line, residual.mean()

def project_points_to_line_only_distance(points, line_direction, line_point):
    d = points - line_point
    projs = line_point + np.outer(np.dot(d, line_direction), line_direction)
    diff = points - projs
    dists = np.sqrt(np.sum(diff**2, axis=1))
    return dists

def project_points_to_line_only_projection(points, line_direction, line_point):
    d = points - line_point
    projs = line_point + np.outer(np.dot(d, line_direction), line_direction)
    return projs

def project_points_to_line_with_distance(points, line_direction, line_point):
    d = points - line_point
    projs = line_point + np.outer(np.dot(d, line_direction), line_direction)
    diff = points - projs
    dists = np.sqrt(np.sum(diff**2, axis=1))
    return projs, dists

def project_point_to_line(points, line):
    '''
    points: n*2
    return n*2
    '''
    a,b = line
    P0 = np.vstack((points.T,np.ones((points.shape[0],1)).T))
    A = np.array([[b*b, -a*b, -a], [-a*b, a*a, -b], [0,0,a*a+b*b]])
    P = np.dot(A,P0)
    p = P[:2,:]/A[2,2]
    return p.T

##--------------------------------------------------------------------------------
# import pcl
# 
# def statistical_outlier_removal(points, kd=None):
#     cloud = pcl.PointCloud()
#     cloud.from_array(points)
#     if kd is None:
#         kd = cloud.make_kdtree_flann()
#     indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, 100)
#     mean_distances = np.mean(np.sqrt(sqr_distances), axis=1)
#     std_distances = np.std(mean_distances)
#     print mean_distances.mean(), std_distances
#     return points[mean_distances < 0.3]

#    return points[mean_distances < mean_distances.mean() + 0.2*std_distances]
    
#    import matplotlib.pyplot as plt
#    bins_in = np.arange(0,1,0.001)
#    hist, bins = np.histogram(mean_distances, bins_in)
#    width = 0.7*(bins[1]-bins[0])
#    center = (bins[:-1]+bins[1:])/2
#    plt.figure("mean_distances")
#    plt.bar(center, hist, align = 'center', width = width)
#    plt.show()

def compute_cloud_normals(points, kd=None, k=50):
    cloud = pcl.PointCloud()
    cloud.from_array(points)
    if kd is None:
        kd = cloud.make_kdtree_flann()
    indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, k)
    normals = np.zeros((points.shape[0],3))
    
    for point_ind, neighbor_indices in enumerate(indices):
#        print "neighbors of ",point_ind
        patch_points = points[neighbor_indices]
#        print patch_points 
    
        mean = np.mean(patch_points, axis=0)
        std =  np.std(patch_points, axis=0)
#        if (std == 0).any():
#            print patch_points
        patch_points_normalized = (patch_points - mean) / (std+0.0000001)
    
        try:
            U,s,Vt = linalg.svd(patch_points_normalized)
        except linalg.LinAlgError as e:
            print 'PCA error', patch_points_normalized
            normals[point_ind] = normals[point_ind-1]
            continue
            
        V = Vt.T
        ind = np.argsort(s)[::-1]
        U = U[:,ind]
        s = s[ind]
        V = V[:,ind]
        S = np.diag(s)
        Mhat = np.dot(U[:,:2],np.dot(S[:2,:2],V[:,:2].T)) * std + mean
        
        found = False
        for i in range(k-1):
            v1 = Mhat[0]-Mhat[i]
            if not (v1 == 0).all():
                for j in range(i+1, k-1):
                    v2 = Mhat[0]-Mhat[j]
                    patch_normal = np.cross(v1, v2)
                    if linalg.norm(patch_normal) != 0:
                        found = True
                        break
            if found: break
            
#        patch_normal = np.cross(v1, v2)
        if linalg.norm(patch_normal) == 0:
            print v1, v2
        patch_normal = patch_normal/linalg.norm(patch_normal)
        patch_normal = adjust_normal_direction(patch_normal, points[point_ind])
        normals[point_ind] = patch_normal
        
    return normals
    



