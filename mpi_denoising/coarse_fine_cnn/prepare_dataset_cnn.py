import numpy as np
from tqdm import tqdm
from tensorflow.keras import datasets, layers, models, initializers
import scipy.io as sio
import tensorflow as tf

#######################################################
def prepare_dataset():
  
  # Define values to be changed in order to consider all the maps for all the scenes
  scene_number = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', 
                  '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', 
                  '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029',
                  '0030', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039']
  freq = [20, 50, 60]
  type_of_map = ['amplitude.mat', 'depth.mat', 'intensity.mat']

  gt_path = "/content/gdrive/MyDrive/synthetic_dataset/training_set/ground_truth/scene_"
  path1 = "_depth.mat"
  gt = {}

  for n in tqdm(range(40), desc = 'Ground Truth Data Preparation'):
    mat_cont = sio.loadmat(gt_path + scene_number[n] + path1)
    gt1 = np.resize(mat_cont["depth_GT_radial"], (128,128,1))
    gt[n] = gt1

  gtp = []
  for i in range(40):
    for j in range(10):
      gtp.append(gt[i][None,:,:,:])

  path = "/content/gdrive/MyDrive/synthetic_dataset/training_set/scene_"
  path2 = "_MHz"

  a20 = {}
  a20_patches = {}

  a50 = {}
  a50_patches = {}

  a60 = {}
  a60_patches = {}

  d20 = {}
  d20_patches = {}

  d50 = {}
  d50_patches = {}

  d60 = {}
  d60_patches = {}

  d20_60 = {} 
  d20_60_patches = {}

  d50_60 = {} 
  d50_60_patches = {}

  a20_60 = {}
  a20_60_patches = {}

  a50_60 = {}
  a50_60_patches = {}

  ig = {}

  data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation((-0.5, 0.5)),
  ])

  for m in tqdm(range(40), desc = 'Data Preparation and Augmentation'):
    for i in range(3):
      for j in range(3):
        mat_cont = sio.loadmat(path + scene_number[m] + path2 + str(freq[i]) + '_' + type_of_map[j])
        if j == 0:
          if i == 0:
            a20[m] = np.resize(mat_cont["amplitude"], (240,320,1))
            a20_arr = np.expand_dims(a20[m], axis=0)
            for n in range(10):
              a20_patches[n] = tf.image.random_crop(value=a20_arr, size=(1,128,128,1))
              a20_patches[n] = data_augmentation(a20_patches[n])           
            a20[m] = np.reshape(np.array(list(a20_patches.values())), (10,128,128,1))
          elif i == 1: 
            a50[m] = np.resize(mat_cont["amplitude"], (240,320,1))
            a50_arr = np.expand_dims(a50[m], axis=0)
            for n in range(10):
              a50_patches[n] = tf.image.random_crop(value=a50_arr, size=(1,128,128,1))
              a50_patches[n] = data_augmentation(a50_patches[n])           
            a50[m] = np.reshape(np.array(list(a50_patches.values())), (10,128,128,1))
          else:
            a60[m] = np.resize(mat_cont["amplitude"], (240,320,1))
            a60_arr = np.expand_dims(a60[m], axis=0)
            for n in range(10):
              a60_patches[n] = tf.image.random_crop(value=a60_arr, size=(1,128,128,1))
              a60_patches[n] = data_augmentation(a60_patches[n])           
            a60[m] = np.reshape(np.array(list(a60_patches.values())), (10,128,128,1))
        elif j == 1:
          if i == 0:
            d20[m] = np.resize(mat_cont["depth_PU"], (240,320,1))
            d20_arr = np.expand_dims(d20[m], axis=0)
            for n in range(10):
              d20_patches[n] = tf.image.random_crop(value=d20_arr, size=(1,128,128,1))
              d20_patches[n] = data_augmentation(d20_patches[n])
            d20[m] = np.reshape(np.array(list(d20_patches.values())), (10,128,128,1))
          elif i == 1: 
            d50[m] = np.resize(mat_cont["depth_PU"], (240,320,1))
            d50_arr = np.expand_dims(d50[m], axis=0)
            for n in range(10):
              d50_patches[n] = tf.image.random_crop(value=d50_arr, size=(1,128,128,1))
              d50_patches[n] = data_augmentation(d50_patches[n])
            d50[m] = np.reshape(np.array(list(d50_patches.values())), (10,128,128,1))
          else:
            d60[m] = np.resize(mat_cont["depth_PU"], (240,320,1))
            d60_arr = np.expand_dims(d60[m], axis=0)
            for n in range(10):
              d60_patches[n] = tf.image.random_crop(value=d60_arr, size=(1,128,128,1))
              d60_patches[n] = data_augmentation(d60_patches[n])
            d60[m] = np.reshape(np.array(list(d60_patches.values())), (10,128,128,1))
        else:   
          int_maps = np.resize(mat_cont["intensity"], (240,320,1))

    d20_60[m] = np.subtract(d20[m], d60[m])
    d50_60[m] = np.subtract(d50[m], d60[m])
    a20_60[m] = np.zeros((10,128,128,1))
    np.divide(a20[m], a60[m], out=a20_60[m], where=a60[m]!=0)
    a50_60[m] = np.zeros((10,128,128,1))
    np.divide(a50[m], a60[m], out=a50_60[m], where=a60[m]!=0)
    ig[m] = np.concatenate((d60[m], d20_60[m], d50_60[m], a20_60[m], a50_60[m]), axis=-1) 

  d60p = np.concatenate(np.array(list(d60.values())), axis=0)
  d2060p = np.concatenate(np.array(list(d20_60.values())), axis=0)
  d5060p = np.concatenate(np.array(list(d50_60.values())), axis=0)
  a2060p = np.concatenate(np.array(list(a20_60.values())), axis=0)
  a5060p = np.concatenate(np.array(list(a50_60.values())), axis=0)
  # igp = np.concatenate((d60p, d2060p, d5060p, a2060p, a5060p), axis=0)
  igp = np.concatenate((d60p, d2060p, d5060p, a2060p, a5060p), axis=-1)
  # print(igp.shape)

  dataset_size = 400
  batch_size = 4

  igp = np.resize(igp, ((dataset_size // batch_size), batch_size, 128, 128, 5))
  gtp = np.resize(gtp, ((dataset_size // batch_size), batch_size, 128, 128, 1))

  return igp, gtp, dataset_size, batch_size