import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import coarseCNN as cCNN


#######################################################
def coarsefineCNN(): 

    # inputs_fine = tf.keras.Input(shape=(128, 128, 5))

    conv1_fine = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', 
                                        activation='relu', 
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(cCNN.inputs_coarse)

    conv2_fine = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', 
                                        activation='relu',
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(conv1_fine)

    conv3_fine = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', 
                                        activation='relu',
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(conv2_fine)

    concat_fine = tf.keras.layers.concatenate([cCNN.outputs_coarse, conv3_fine])

    conv4_fine = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', 
                                        activation='relu',
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(concat_fine)

    outputs_fine = tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same',
                                        activation='relu',
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=l2(0.0001), 
                                        bias_regularizer=l2(0.0001))(conv4_fine)

    coarsefineCNN = tf.keras.Model(inputs=cCNN.inputs_coarse, outputs=outputs_fine)

    coarsefineCNN.summary()
    plot_model(coarsefineCNN, to_file='coarsefineCNN_graph.png')

    return coarsefineCNN