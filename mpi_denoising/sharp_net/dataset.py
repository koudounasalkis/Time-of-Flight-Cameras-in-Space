import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

flg = False
dtype = tf.float32


#######################################################
def im2col(input, kernel_size = 3):

    h_pos_list = []
    w_pos_list = []

    h_max = tf.shape(input)[1]
    w_max = tf.shape(input)[2]
    batch_size = tf.shape(input)[0]

    padding_size = int((kernel_size - 1) / 2)
    input_padding = tf.pad(input, paddings=[[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]])
    w_pos, h_pos = tf.meshgrid(tf.range(1, w_max + 1), tf.range(1, h_max + 1))
    w_pos = tf.expand_dims(tf.expand_dims(w_pos, 0), -1)
    h_pos = tf.expand_dims(tf.expand_dims(h_pos, 0), -1)
    w_pos = tf.cast(w_pos, dtype=tf.float32)
    h_pos = tf.cast(h_pos, dtype=tf.float32)

    for i in range(0-padding_size, padding_size + 1, 1):
        for j in range(0-padding_size, padding_size + 1, 1):
            h_pos = h_pos + tf.cast(i, dtype=tf.float32)
            w_pos = w_pos + tf.cast(j, dtype=tf.float32)
            h_pos_list.append(h_pos)
            w_pos_list.append(w_pos)

    h_pos = tf.concat(h_pos_list, axis=-1)
    w_pos = tf.concat(w_pos_list, axis=-1)
    h_pos = tf.tile(h_pos, multiples=[batch_size, 1, 1, 1])
    w_pos = tf.tile(w_pos, multiples=[batch_size, 1, 1, 1])

    tensor_batch = tf.range(batch_size)
    tensor_batch = tf.reshape(tensor_batch, [batch_size, 1, 1, 1])
    tensor_batch = tf.tile(tensor_batch, multiples=[1, h_max, w_max, kernel_size ** 2])
    tensor_batch = tf.cast(tensor_batch, dtype=tf.float32)

    tensor_channel = tf.zeros(shape=[kernel_size ** 2], dtype=tf.float32)
    tensor_channel = tf.reshape(tensor_channel, [1, 1, 1, kernel_size ** 2])
    tensor_channel = tf.tile(tensor_channel, multiples=[batch_size, h_max, w_max, 1])
    tensor_channel = tf.cast(tensor_channel, dtype=tf.float32)

    idx = tf.stack([tensor_batch, h_pos, w_pos, tensor_channel], axis=-1)

    idx = tf.reshape(idx, [-1, 4])
   
    params = input_padding
    indices = tf.cast(idx, dtype=tf.int32)
    im = tf.gather_nd(params, indices)

    output = tf.reshape(im, [batch_size, h_max, w_max, kernel_size ** 2])
    return output


#######################################################
@tf.function
def get_input_fn(pred, shuffle=False, repeat_count=1): 
    
    def _parse_dataset(element):
      depth_raw, amplitude_raw = tf.unstack(element, axis=-1)
      features = { 'Depth': depth_raw, 'Amplitude': amplitude_raw }
      return features  

    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():

      pred = tf.convert_to_tensor(pred)
      dataset = tf.data.Dataset.from_tensor_slices(pred)
      dataset = dataset.map(_parse_dataset)
      if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
      dataset = dataset.repeat(repeat_count)
      return dataset
