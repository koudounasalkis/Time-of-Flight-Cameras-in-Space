import open3d as o3d
import time
from global_registration import *

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/libroyale/python/pcd_registration/'


#######################################################
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


#######################################################
def optimization_based_registration(source, target):

    vol = o3d.visualization.read_selection_polygon_volume(root_dir + 'cropped.json')
    source_pointcloud = vol.crop_point_cloud(source)
    source_pointcloud.paint_uniform_color([1, 0.706, 0])
    
    aabb = source_pointcloud.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = source_pointcloud.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    print("\nDetecting the satellite model...")
    o3d.visualization.draw_geometries([source_pointcloud, aabb, obb])

    source = source_pointcloud
    voxel_size = 0.05 
    print("Drawing initial alignment...")
    start = time.time()
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(source, target, voxel_size)

    result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, 
                                                   target_fpfh, voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print("Drawing final alignment...")
    draw_registration_result(source, target, result_fast.transformation)
