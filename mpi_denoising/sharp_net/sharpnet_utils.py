import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import time
from PIL import Image
import sharpnet as net
import dataset as ds

flg = False
dtype = tf.float32 


#######################################################
def tof_net_func(features, mode):

  graph = tf.compat.v1.get_default_graph()
  with graph.as_default():  

    depth = None
    amplitude = None

    depth_outs = None
    depth = features['Depth']
    amplitude = features['Amplitude']

    inputs = tf.concat([depth, amplitude], axis=-1)
    inputs = inputs[None,:,:,:]
    depth_outs, depth_residual_input = net.get_network(x=inputs, 
                                                      flg=mode == tf.estimator.ModeKeys.PREDICT,
                                                      denoising_mode='medium')

    predictions = {
        "depth_input": depth,
        "amplitude_input": amplitude,
        "depth": depth_outs,
      }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


#######################################################
def denoising(pred, result_path, model_dir, checkpoint_steps, denoising_mode): 

    root_dir = result_path
    if denoising_mode == 'simple':
        name_network = 'ToF-KPN'
        pred_depth_png_dir = os.path.join(root_dir, 'tof_kpn/pred_depth_png')
        depth_input_png_dir = os.path.join(root_dir, 'tof_kpn/input_depth_png')
    elif denoising_mode == 'medium':
        name_network = 'SHARP-Net without Refine Fusion'
        pred_depth_png_dir = os.path.join(root_dir, 'sharpnet_no_refine_fusion/pred_depth_png')
        depth_input_png_dir = os.path.join(root_dir, 'sharpnet_no_refine_fusion/input_depth_png')
    else:
        name_network = 'SHARP-Net'
        pred_depth_png_dir = os.path.join(root_dir, 'sharpnet/pred_depth_png')
        depth_input_png_dir = os.path.join(root_dir, 'sharpnet/input_depth_png')       

    if not os.path.exists(pred_depth_png_dir):
      os.mkdir(pred_depth_png_dir)
    if not os.path.exists(depth_input_png_dir):
      os.mkdir(depth_input_png_dir)

    configuration = tf.estimator.RunConfig(model_dir=model_dir,
                                           log_step_count_steps = 10,
                                           save_summary_steps = 5)

    tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=configuration)
    
    start = time.time()
    result = list(tof_net.predict(
                      input_fn=lambda: ds.get_input_fn(pred, shuffle=False, repeat_count=1),
                      checkpoint_path=model_dir + '/model.ckpt-' + checkpoint_steps,
                      yield_single_examples=False))
    print(f"\nMPI and Shot Denoising with {name_network} took %.3f sec." % (time.time() - start))

    pred_depth = np.squeeze(result[0]['depth'])
    input_depth = np.squeeze(result[0]['depth_input'])

    pred_depth_png_path = os.path.join(pred_depth_png_dir, name_network + 'pred_depth.png')
    pred_depth_png = Image.fromarray(pred_depth * 1)
    pred_depth_png = pred_depth_png.convert("L")
    pred_depth_png.save(pred_depth_png_path)

    depth_input_png_path = os.path.join(depth_input_png_dir, 'input_depth.png')
    input_depth_png = Image.fromarray(input_depth * 1)
    input_depth_png = input_depth_png.convert("L")
    input_depth_png.save(depth_input_png_path)

    pred_depth = np.reshape(pred_depth, -1).astype(np.float32)
    return pred_depth
