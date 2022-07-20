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
    o3d.io.write_point_cloud(root_dir + "source.ply", source_temp)
    o3d.io.write_point_cloud(root_dir + "target.ply", target_temp)
    custom_draw_geometry_with_rotation(source_temp, target_temp)


#######################################################
def optimization_based_registration(source, target):
    
    vol = o3d.visualization.read_selection_polygon_volume(root_dir + 'cropped.json')
    source_pointcloud = vol.crop_point_cloud(source)
    source_pointcloud.paint_uniform_color([1, 0.706, 0])
    
    aabb = source_pointcloud.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = source_pointcloud.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    o3d.visualization.draw_geometries([source_pointcloud, aabb, obb])

    source = source_pointcloud
    threshold = 0.0002
    trans_init = np.identity(4)
    draw_registration_result(source, target, trans_init)

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    start = time.time()
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000000))
    print("ICP registration took %.3f sec.\n" % (time.time() - start))
    # print(f"ICP Transformation is:\n {reg_p2p.transformation}")
    draw_registration_result(source, target, reg_p2p.transformation)