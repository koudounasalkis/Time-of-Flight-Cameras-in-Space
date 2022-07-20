import open3d as o3d
import numpy as np
import copy
import time 

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/libroyale/python/pcd_registration/'


#######################################################
def custom_draw_geometry_with_rotation(source, target):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False
    o3d.visualization.draw_geometries_with_animation_callback([source, target], rotate_view)


#######################################################
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target) 
    source_temp.paint_uniform_color([1, 0.706, 0])  
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.io.write_point_cloud(root_dir + "source.ply", source_temp)
    # o3d.io.write_point_cloud(root_dir + "target.ply", target_temp)
    custom_draw_geometry_with_rotation(source_temp, target_temp)


#######################################################
def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


#######################################################
def prepare_dataset(source, target, voxel_size):
    
    source = source
    target = target
    trans_init = np.identity(4)
    draw_registration_result(source, target, trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    return source, target, source_down, target_down, source_fpfh, target_fpfh


#######################################################
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    return result


#######################################################
def refine_registration(source, target, result_ransac, voxel_size):
    
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result


#######################################################
def feature_learning_registration(source, target):
    
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
           
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    print("Drawing final alignment with ransac...")
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_icp = refine_registration(source, target, result_ransac, voxel_size)
    print(result_icp)
    print("Global registration with refinement took %.3f sec.\n" % (time.time() - start))
    print("Drawing final alignment with refinment icp...")
    draw_registration_result(source, target, result_icp.transformation)    