import open3d as o3d
import registration.global_registration as gr
import time
import copy

#######################################################
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance = distance_threshold))
    return result

#######################################################
def refine_registration(source, target, corrs):
    start = time.time()
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    result = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corrs))
    elapsed_time = time.time() - start
    return result, elapsed_time

#######################################################
def initializePointCoud(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

#######################################################
def preProcessData(pts, flag):
    return initializePointCoud(pts[flag])

#######################################################
def selectFunction(name):
    if name == 'global':
        return gr.execute_global_registration
    elif name == 'fast':
        return execute_fast_global_registration
    else:
        raise ValueError('Please select a valid function: global or fast')

#######################################################
def globalRegistration(source, target, execute_registration):
    voxel_size = 0.05
    source_down, source_fpfh = gr.preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = gr.preprocess_point_cloud(target, voxel_size)
    voxel_size = 0.05
    start = time.time()
    result = execute_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    elapsed_time = time.time() - start
    return result, elapsed_time, source_down, target_down

#######################################################
def makeMatchesSet(corres):
    set = []
    corres = corres.reshape(corres.shape[1])
    for idx, c in enumerate(corres):
        if c == 1:
            set.append([idx, idx])
 
    return set

#######################################################
def draw_reg_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])