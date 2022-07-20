import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


#######################################################
def coarseCNN(): 

    inputs_coarse = tf.keras.Input(shape=(128, 128, 5))

    conv1_coarse = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', 
                                        activation='relu', 
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(inputs_coarse)
    maxP1_coarse = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv1_coarse)

    conv2_coarse = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', 
                                        activation='relu', 
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(maxP1_coarse)
    maxP2_coarse = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv2_coarse)

    conv3_coarse = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', 
                                        activation='relu',
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(maxP2_coarse)

    conv4_coarse = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', 
                                        activation='relu', 
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(conv3_coarse)

    conv5_coarse = tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', 
                                        activation='relu', 
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(conv4_coarse)

    outputs_coarse = tf.keras.layers.UpSampling2D(size=(4,4), interpolation='bilinear', name='upsamp')(conv5_coarse)

    coarseCNN = tf.keras.Model(inputs=inputs_coarse, outputs=outputs_coarse)

    coarseCNN.summary()
    plot_model(coarseCNN, to_file='coarseCNN_graph.png')

    return coarseCNN