import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()
import dataset as ds

flg = False
dtype = tf.float32


##################################################
# Activation Functions
def leaky_relu(x):
    alpha = 0.1
    x_pos = tf.nn.relu(x)
    x_neg = tf.nn.relu(-x)
    return x_pos - alpha * x_neg

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x) - 0.5


##################################################
# Model
def feature_extractor_subnet(x, flg):
      
    pref = 'feature_extractor_subnet_'  
    train_ae = flg      # Whether to train flag

    # Define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            flag_init = False   # Select certain variables
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                
                from tensorflow.python.framework import dtypes        # Self define the initializer function
                from tensorflow.python.ops.init_ops import Initializer
                exec("class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): \
                    self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): \
                    return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self): \
                    return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # Autoencoder
    n_filters = [
        16, 16,
        32, 32,
        64, 64,
        96, 96,
        128, 128,
        192, 192 ]

    filter_sizes = [
        3, 3,
        3, 3,
        3, 3,
        3, 3,
        3, 3,
        3, 3 ]
    
    pool_sizes = [ 
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1 ]

    pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1,
        2, 1 ]
    
    ae_inputs = tf.identity(x, name='ae_inputs')            # Change space   
    current_input = tf.identity(ae_inputs, name="input")     # Prepare input
    
    # Convolutional layers: feature extractor
    features = []
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # Define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        # Convolution
        current_input = tf.keras.layers.Conv2D(filters=n_filters[i],
                                               kernel_size=[filter_sizes[i], filter_sizes[i]],
                                               padding="same",
                                               activation=relu,
                                               trainable=train_ae,
                                               kernel_initializer=kernel_init,
                                               bias_initializer=bias_init,
                                               name=name)(current_input)

        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            current_input = current_input
            if (i == len(n_filters) - 1) or (pool_sizes[i + 1] == 2 and pool_strides[i + 1] == 2):
                features.append(current_input)
        else:
            current_input = tf.keras.layers.MaxPooling2D( pool_size=[pool_sizes[i], pool_sizes[i]],
                                                         strides=pool_strides[i],
                                                         name=pref + "pool_" + str(i))(current_input)           
    return features


##################################################
def depth_residual_regression_subnet(x, flg, subnet_num):

    pref = 'depth_regression_subnet_' + str(subnet_num) + '_'
  
    train_ae = flg      # Whether to train flag

    # Define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            
            flag_init = False       # Select certain variables
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())

                from tensorflow.python.framework import dtypes  # Self define the initializer function
                from tensorflow.python.ops.init_ops import Initializer
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): \
                    self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): \
                    return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):\
                    return {\"dtype\": self.dtype.name}"
                    )
                inits.append(name_f)

    # Autoencoder
    n_filters = [
        128, 96,
        64, 32,
        16, 1, ]
    filter_sizes = [
        3, 3,
        3, 3,
        3, 3 ]
    pool_sizes = [
        1, 1,
        1, 1,
        1, 1 ]
    pool_strides = [
        1, 1,
        1, 1,
        1, 1 ]
    
    # Change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # Prepare input
    current_input = tf.identity(ae_inputs, name="input")

    # Convolutional layers: depth regression
    feature = []
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # Define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # Convolution
        current_input = tf.keras.layers.Conv2D(filters=n_filters[i],
                                               kernel_size=[filter_sizes[i], filter_sizes[i]],
                                               padding="same",
                                               activation=activation,
                                               trainable=train_ae,
                                               kernel_initializer=kernel_init,
                                               bias_initializer=bias_init,
                                               name=name)(current_input)

        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            feature.append(current_input)
        else:
            feature.append(tf.keras.layers.MaxPooling2D(pool_size=[pool_sizes[i], pool_sizes[i]],
                                                        strides=pool_strides[i],
                                                        name=pref + "pool_" + str(i))(current_input))
        current_input = feature[-1]

    depth_coarse = tf.identity(feature[-1], name='depth_coarse_output')
    
    return depth_coarse


##################################################
def unet_subnet(x, flg):

    pref = 'unet_subnet_'

    train_ae = flg      # Whether to train flag

    # Define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            flag_init = False           # Select certain variables
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                from tensorflow.python.framework import dtypes  # Self define the initializer function
                from tensorflow.python.ops.init_ops import Initializer
                exec("class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): \
                    self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): \
                    return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self): \
                    return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # Autoencoder
    n_filters = [
        16, 16,
        32, 32,
        64, 64,
        128, 128 ]
    filter_sizes = [
        3, 3,
        3, 3,
        3, 3,
        3, 3 ]
    pool_sizes = [
        1, 1,
        2, 1,
        2, 1,
        2, 1 ]
    pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1 ]
    skips = [ 
        False, False,
        True, False,
        True, False,
        True, False ]
    
    
    ae_inputs = tf.identity(x, name='ae_inputs')            # Change space  
    current_input = tf.identity(ae_inputs, name="input")    # Prepare input

    # Convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # Define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        # Convolution
        conv.append(tf.keras.layers.Conv2D(filters=n_filters[i],
                                            kernel_size=[filter_sizes[i], filter_sizes[i]],
                                            padding="same",
                                            activation=relu,
                                            trainable=train_ae,
                                            kernel_initializer=kernel_init,
                                            bias_initializer=bias_init,
                                            name=name)(current_input))
        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            pool.append(conv[-1])
        else:
            pool.append(tf.keras.layers.MaxPooling2D(pool_size=[pool_sizes[i], pool_sizes[i]],
                                                     strides=pool_strides[i],
                                                     name=pref + "pool_" + str(i))(conv[-1]))
        current_input = pool[-1]
    
    # Convolutional layer: decoder
    upsamp = []
    current_input = pool[-1]
    for i in range((len(n_filters) - 1) - 1, 0, -1):
        name = pref + "upsample_" + str(i)

        # Define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        # Change the kernel size in upsample process
        if skips[i] == False and skips[i + 1] == True:
            filter_sizes[i] = 4

        # Upsampling
        current_input = tf.keras.layers.Conv2DTranspose(filters=n_filters[i],
                                                        kernel_size=[filter_sizes[i], filter_sizes[i]],
                                                        strides=(pool_strides[i], pool_strides[i]),
                                                        padding="same",
                                                        activation=relu,
                                                        trainable=train_ae,
                                                        kernel_initializer=kernel_init,
                                                        bias_initializer=bias_init,
                                                        name=name)(current_input)
        upsamp.append(current_input)

        # Skip connection
        if skips[i] == False and skips[i - 1] == True:
            current_input = tf.concat([current_input, pool[i + 1]], axis=-1)
    
    features = tf.identity(upsamp[-1], name='ae_output')
    
    return features


##################################################
def depth_output_subnet(inputs, flg, kernel_size): 
    
    pref = 'depth_output_subnet_'
    current_input = inputs
    
    train_ae = flg      # Whether to train flag  
    
    # Define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():           
            flag_init = False  
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                # Self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec("class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): \
                    self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): \
                    return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self): \
                    return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    n_filters_mix = [kernel_size ** 2]
    filter_sizes_mix = [1]
    mix = []
    for i in range(len(n_filters_mix)):
        name = pref + "conv_" + str(i)

        # Define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        if i == (len(n_filters_mix) - 1):
            activation = sigmoid
        else:
            activation = relu

        # Convolution
        mix.append(tf.keras.layers.Conv2D(filters=n_filters_mix[i],
                                          kernel_size=[filter_sizes_mix[i], filter_sizes_mix[i]],
                                          padding="same",
                                          activation=activation,
                                          trainable=train_ae,
                                          kernel_initializer=kernel_init,
                                          bias_initializer=bias_init,
                                          name=name)(current_input))
        current_input = mix[-1]

    return current_input


##################################################
def dear_kpn(x, flg):

    kernel_size = 3
    features = unet_subnet(x, flg)
    weights = depth_output_subnet(features, flg, kernel_size=kernel_size)
    weights = weights / tf.math.reduce_sum(tf.abs(weights) + 1e-6, axis=-1, keepdims=True)
    column = ds.im2col(x, kernel_size=kernel_size)
    current_output = tf.math.reduce_sum(column * weights, axis=None, keepdims=True)
    depth_output = tf.identity(current_output, name='depth_output')

    return depth_output


##################################################
def residual_output_subnet(x, flg, subnet_num):

    pref = 'residual_output_subnet_' + str(subnet_num) + '_'

    train_ae = flg      # Whether to train flag

    # Define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            flag_init = False   
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                from tensorflow.python.framework import dtypes     # Self define the initializer function
                from tensorflow.python.ops.init_ops import Initializer
                exec("class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): \
                    self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): \
                    return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self): \
                    return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # Autoencoder
    n_filters = [1]
    filter_sizes = [1]
    pool_sizes = [1]
    pool_strides = [1]
    
    ae_inputs = tf.identity(x, name='ae_inputs')            # Change space
    current_input = tf.identity(ae_inputs, name="input")    # Prepare input

    # Convolutional layers: depth regression
    feature = []
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)
        if name + '_bias' in inits:                         # Define the initializer
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # Convolution
        current_input = tf.keras.layers.Conv2D(filters=n_filters[i],
                                               kernel_size=[filter_sizes[i], filter_sizes[i]],
                                               padding="same",
                                               activation=activation,
                                               trainable=train_ae,
                                               kernel_initializer=kernel_init,
                                               bias_initializer=bias_init,
                                               name=name)(current_input)

        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            feature.append(current_input)
        else:
            feature.append(tf.keras.layers.MaxPooling2D(pool_size=[pool_sizes[i], pool_sizes[i]],
                                             strides=pool_strides[i],
                                             name=pref + "pool_" + str(i))(current_input))
        current_input = feature[-1]

    depth_residual_coarse = tf.identity(feature[-1], name='depth_coarse_residual_output')
    
    return depth_residual_coarse


##################################################
def sharpnet(x, flg):

    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():

      depth_residual = []
      depth_residual_input = []
      h_max = tf.shape(x)[1]
      w_max = tf.shape(x)[2]

      depth = tf.expand_dims(x[:, :, :, 0], axis=-1)
      amplitude = tf.expand_dims(x[:, :, :, 1], axis=-1)
      depth_and_amplitude = tf.concat([depth, amplitude], axis=-1)
      features = feature_extractor_subnet(depth_and_amplitude, flg)

      for i in range(1, len(features) + 1):
          if i == 1:
              inputs = features[len(features) - i]
          else:
              feature_input = features[len(features) - i]
              h_max_low_scale = tf.shape(feature_input)[1]
              w_max_low_scale = tf.shape(feature_input)[2]
              depth_coarse_input = tf.compat.v1.image.resize_bicubic(depth_residual[-1], 
                                                                     size=(h_max_low_scale, w_max_low_scale), 
                                                                     align_corners=True)
              inputs = tf.concat([feature_input, depth_coarse_input], axis=-1)
          current_depth_residual = depth_residual_regression_subnet(inputs, flg, subnet_num=i)
          depth_residual.append(current_depth_residual)

          current_depth_residual_input = tf.compat.v1.image.resize_bicubic(current_depth_residual, 
                                                                            size=(h_max, w_max), 
                                                                            align_corners=True)
          depth_residual_input.append(current_depth_residual_input)
        
      depth_coarse_residual_input = tf.concat(depth_residual_input, axis=-1)
      final_depth_residual_output = residual_output_subnet(depth_coarse_residual_input, flg, subnet_num=0)

      current_final_depth_output = depth + final_depth_residual_output
      final_depth_output = dear_kpn(current_final_depth_output, flg)
      depth_residual_input.append(final_depth_residual_output)
      depth_residual_input.append(final_depth_output - current_final_depth_output)
      return final_depth_output, depth_residual_input


##################################################
def sharpnet_no_refine_fusion(x, flg):

    depth_residual = []
    depth_residual_input = []
    h_max = tf.shape(x)[1]
    w_max = tf.shape(x)[2]

    depth = tf.expand_dims(x[:, :, :, 0], axis=-1)
    depth_and_amplitude = x[:, :, :, 0:2]
    features = feature_extractor_subnet(depth_and_amplitude, flg)

    for i in range(1, len(features) + 1):
        if i == 1:
            inputs = features[len(features) - i]
        else:
            feature_input = features[len(features) - i]
            h_max_low_scale = tf.shape(feature_input)[1]
            w_max_low_scale = tf.shape(feature_input)[2]
            depth_coarse_input = tf.compat.v1.image.resize_bicubic(depth_residual[-1], 
                                                                    size=(h_max_low_scale, w_max_low_scale),
                                                                    align_corners=True)
            inputs = tf.concat([feature_input, depth_coarse_input], axis=-1)
        current_depth_residual = depth_residual_regression_subnet(inputs, flg, subnet_num=i)
        depth_residual.append(current_depth_residual)

        current_depth_residual_input = tf.compat.v1.image.resize_bicubic(current_depth_residual, 
                                                                         size=(h_max, w_max),
                                                                         align_corners=True)
        depth_residual_input.append(current_depth_residual_input)

    final_depth_residual_output = depth_residual_input[-1]
    current_final_depth_output = depth + final_depth_residual_output
    final_depth_output = current_final_depth_output
    depth_residual_input.append(final_depth_residual_output)
    depth_residual_input.append(final_depth_output - current_final_depth_output)
    return final_depth_output, depth_residual_input


##################################################
def simple_tof_kpn(x, flg):

    kernel_size = 3
    depth_input = x[:, :, :, 0]
    depth_input = tf.expand_dims(depth_input, axis=-1)
    depth_and_amplitude = x[:, :, :, 0:2]
    features = unet_subnet(depth_and_amplitude, flg)
    biases, weights = depth_output_subnet(features, flg, kernel_size=kernel_size)
    weights = weights / tf.reduce_sum(tf.abs(weights) + 1e-6, axis=-1, keep_dims=True)
    inputs = depth_input + biases
    column = ds.im2col(inputs, kernel_size=kernel_size)
    current_output = tf.reduce_sum(column * weights, axis=-1, keep_dims=True)
    depth_output = tf.identity(current_output, name='depth_output')

    return depth_output


##################################################
def get_network(x, flg, denoising_mode):
    if denoising_mode == 'simple':
        return simple_tof_kpn(x, flg)
    elif denoising_mode == 'medium':
        return sharpnet_no_refine_fusion(x, flg)
    else:
        return sharpnet(x, flg) 