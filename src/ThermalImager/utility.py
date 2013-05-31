import numpy as np
import matplotlib.pyplot as plt
from ThermalImager import geometry


def get_heatmap(input):
    my_cmap = plt.get_cmap('jet')
    rgba_img = my_cmap(input)
    rgb_img = np.delete(rgba_img, 3, 2)*255
    return rgb_img.astype(np.uint8)

def draw_axis():
    origin = add_to_pointcloud_color(None, np.array([0,0,0]), np.array([255,255,255]))
    origin = add_to_pointcloud_color(origin, np.outer(np.arange(0.01,1,0.01),np.array([1,0,0])),\
                                      np.array([255,0,0]))
    origin = add_to_pointcloud_color(origin, np.outer(np.arange(0.01,1,0.01),np.array([0,1,0])),\
                                     np.array([0,255,0]))
    origin = add_to_pointcloud_color(origin, np.outer(np.arange(0.01,1,0.01),np.array([0,0,1])),\
                                      np.array([0,0,255]))
    return origin


def generate_multipointcloud_multicolor(clouds, colors):
    p_multicolor = None
    for cloud, color in zip(clouds, colors):
        p_multicolor = add_to_pointcloud_color(p_multicolor, cloud, color)
    return p_multicolor

def draw_lines(e1, e2):
    a = np.arange(0,1,0.01)
    line_cloud = np.outer(a, e1[0]) + np.outer((1-a), e2[0])
    for i in range(1, e1.shape[0]):
        line_cloud = np.vstack((line_cloud, np.outer(a, e1[i]) + np.outer(1-a, e2[i]))) 
    return line_cloud

def draw_line(e1, e2):
    a = np.arange(0,1,0.01)
    line_cloud = a*e1 + (1-a)*e2 
    return line_cloud    

def draw_box(x,direction1,direction2,w,h, color):
    top = add_to_pointcloud_color(None, x+np.outer(np.arange(0,w,0.01),direction1), color)
    top_left = add_to_pointcloud_color(top, x+np.outer(np.arange(0,h,0.01),direction2), color)
    right_lower_point = x+h*direction2+w*direction1
    top_left_bottom = add_to_pointcloud_color(top_left, right_lower_point-np.outer(np.arange(0,w,0.01),direction1), color)
    top_left_bottom_right = add_to_pointcloud_color(top_left_bottom, right_lower_point-np.outer(np.arange(0,h,0.01),direction2), color)
    return top_left_bottom_right

#def exclude_pointcloud_from(c, c1):
#    
#    
#    
#    return c[accum]
        

def generate_stairs_plane_frame_batch_multicolor(edge_points, direction_vector, rise_normal,
                                                 tread_normal, colors):
    plane_frames = []
    for edge_ind, edge_point in enumerate(edge_points):
#        print edge_ind
        w = 4
        
#        print 'rise'
        right_center = edge_point + direction_vector*w/2
        left_center = edge_point - direction_vector*w/2
        line_normal = np.cross(direction_vector, rise_normal)
        top_center = edge_point
        if edge_ind == 0:
            h = 4
        else:
            h = geometry.project_to_plane_only_distance(edge_point, tread_normal, edge_points[edge_ind-1])
#        print 'h', h
        bottom_center = edge_point - line_normal*h
        left_border_points = left_center + np.outer(-np.arange(0,h,0.01), line_normal)
        right_border_points = right_center + np.outer(-np.arange(0,h,0.01), line_normal)
        top_border_points = top_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
        bottom_border_points = bottom_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
        rise_frame = np.vstack((left_border_points,right_border_points,top_border_points,bottom_border_points))      
        plane_frames.append(rise_frame)
        
#        print 'tread'
        line_normal = np.cross(direction_vector, tread_normal)
        top_center = edge_point
        if edge_ind == edge_points.shape[0]-1:
            h = 4
        else:
            h = geometry.project_to_plane_only_distance(edge_point, rise_normal, edge_points[edge_ind+1])
#        print 'h', h
        bottom_center = edge_point + line_normal*h
        left_border_points = left_center + np.outer(np.arange(0,h,0.01), line_normal)
        right_border_points = right_center + np.outer(np.arange(0,h,0.01), line_normal)
        top_border_points = top_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
        bottom_border_points = bottom_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
        tread_frame = np.vstack((left_border_points,right_border_points,top_border_points,bottom_border_points))      
        plane_frames.append(tread_frame)
    
    plane_frames_multicolor = generate_multipointcloud_multicolor(plane_frames, colors)
    return plane_frames_multicolor

def generate_plane_frame_batch_multicolor(plane_points, direction_vector, plane_normal, size, colors):
    plane_frames = []
    for plane_point in plane_points:
        plane_frame = generate_plane_frame(plane_point, direction_vector,
                                                    plane_normal, size)      
        plane_frames.append(plane_frame)
    plane_frames_multicolor = generate_multipointcloud_multicolor(plane_frames, colors)
    return plane_frames_multicolor

def generate_plane_frame_batch(plane_points, direction_vector, plane_normal, size):
    plane_frames = None
    for plane_point in plane_points:
        plane_frame = generate_plane_frame(plane_point, direction_vector,
                                                    plane_normal, size)
        plane_frames = add_to_pointcloud(plane_frames, plane_frame)
    return plane_frames

def generate_plane_frame(plane_point, direction_vector, plane_normal, size):
    w,h = size
    right_center = plane_point + direction_vector*w/2
    left_center = plane_point - direction_vector*w/2
    line_normal = np.cross(direction_vector, plane_normal)
    top_center = plane_point + line_normal*h/2
    bottom_center = plane_point - line_normal*h/2
    left_border_points = left_center + np.outer(np.arange(-h/2.,h/2.,0.01), line_normal)
    right_border_points = right_center + np.outer(np.arange(-h/2.,h/2.,0.01), line_normal)
    top_border_points = top_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
    bottom_border_points = bottom_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
    all_points = np.vstack((left_border_points,right_border_points,top_border_points,bottom_border_points))
    return all_points

def add_to_pointcloud_color(p, p1, color):
    p1_color = paint_pointcloud(p1, color)
    p = add_to_pointcloud(p, p1_color)
    return p

def add_to_pointcloud(p, p1):
    if p is None:
        p = p1
    else:
        p = np.vstack((p,p1))
    return p

def merge_pointclouds(clouds):
    merged = clouds[0]
    for cloud in clouds[1:]:
        merged = np.vstack((merged, cloud))
    return merged

def paint_pointcloud(p, color):
    if p.ndim == 1:
        p_color = np.hstack((p, color_to_float(color)))
    else:
        p_color = np.hstack((p, np.ones((p.shape[0],1))*color_to_float(color)))
    return p_color

def write_XYZRGB(points, filename):
    header = '# .PCD v0.7 - Point Cloud Data file format\n\
VERSION 0.7\n\
FIELDS x y z rgb\n\
SIZE 4 4 4 4\n\
TYPE F F F F\n\
COUNT 1 1 1 1\n\
WIDTH %d\n\
HEIGHT 1\n\
VIEWPOINT 0 0 0 1 0 0 0\n\
POINTS %d\n\
DATA ascii\n' % (points.shape[0], points.shape[0])
    f = open(filename, 'w')
    f.write(header)
    for p in points: 
        f.write(' '.join([str(v) for v in p]) + '\n')
    f.close()


def color_to_float(color):
    import struct
    if color.size == 1:
        color = [color] * 3
    rgb = (color[0] << 16 | color[1] << 8 | color[2]);
    rgb_hex = hex(rgb)[2:-1]
    s = '0' * (8 - len(rgb_hex)) + rgb_hex.capitalize()
#            print color, rgb, hex(rgb)
    rgb_float = struct.unpack('!f', s.decode('hex'))[0]
#            print rgb_float
    return rgb_float


def draw_normals(points, normals):
    normals_cloud = None
    sample_indices = np.random.randint(0,points.shape[0],10000)
    for p,n in zip(points[sample_indices], normals[sample_indices]):
        normals_cloud = add_to_pointcloud(normals_cloud, p + np.outer(np.arange(0,0.05,0.005), n))
    return normals_cloud
    
    
ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')
    
    