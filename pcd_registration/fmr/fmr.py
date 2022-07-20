import os
import sys
import copy
import torch
import torch.utils.data
import logging
import time
import numpy as np
import open3d as o3d
from model import FMR

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/libroyale/python/pcd_registration/'


#######################################################
def custom_draw_geometry_with_rotation(source, target):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False
    o3d.visualization.draw_geometries_with_animation_callback([source, target], rotate_view)


#######################################################
def draw_registration_result(source, target, transformation, type):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if type == 'initial':
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    else:
         source_temp.paint_uniform_color([0, 0.651, 0.929])
         target_temp.paint_uniform_color([1, 0.706, 0])  
    source_temp.transform(transformation)
    # o3d.io.write_point_cloud(root_dir + "source.ply", source_temp)
    # o3d.io.write_point_cloud(root_dir + "target.ply", target_temp)
    custom_draw_geometry_with_rotation(source_temp, target_temp)



#######################################################
def fmr(p0, p1, p0_pcd, p1_pcd):
    fmr = FMR()
    model = fmr.create_model()
    pretrained_path = root_dir + "fmr/checkpoint/fmr_model_7scene.pth"
    # pretrained_path = root_dir + "checkpoint/fmr_model_modelnet40.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    device = "cpu"
    model.to(device)
    start = time.time()
    T_est = fmr.evaluate(model, p0, p1, device)
    print("Feature-Metric Registration took %.3f sec.\n" % (time.time() - start))  
    
    print("Drawing final alignment...")
    draw_registration_result(p1_pcd, p0_pcd, T_est, 'final')


#######################################################
def learning_based_registration(source, target):

    vol = o3d.visualization.read_selection_polygon_volume(root_dir + 'cropped.json')
    source_pointcloud = vol.crop_point_cloud(source)
    source_pointcloud.paint_uniform_color([1, 0.706, 0])
    
    aabb = source_pointcloud.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = source_pointcloud.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    print("\nDetecting the satellite model...")
    o3d.visualization.draw_geometries([source_pointcloud, aabb, obb])

    p0_src = source_pointcloud
    p1 = target
    trans_init = np.identity(4)
    print("\nDrawing initial alignment...")
    draw_registration_result(source, target, trans_init, 'initial')

    downpcd0 = p0_src.voxel_down_sample(voxel_size=0.05)
    p0 = np.asarray(downpcd0.points)
    p0 = np.expand_dims(p0,0)

    downpcd1 = p1.voxel_down_sample(voxel_size=0.05)
    p1 = np.asarray(downpcd1.points)
    p1 = np.expand_dims(p1, 0)

    fmr(p0, p1, downpcd0, downpcd1)