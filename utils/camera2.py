import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import queue
import threading
import cv2
import roypy
import model as mod
import dataset as ds

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/'
output_dir = root_dir + 'Results'
model_dir = root_dir + 'SHARP-Net'

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
    depth_outs, depth_residual_input = mod.get_network(x=inputs, 
                                                      flg=mode == tf.estimator.ModeKeys.PREDICT,
                                                      regular=0.1)

    predictions = {
        "depth_input": depth,
        "amplitude_input": amplitude,
        "depth": depth_outs,
      }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


#######################################################
class MyListener(roypy.IDepthDataListener):

    #######################################################
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q
        self.depth_and_amplitude = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 224, 320, 1, 2))
        self.configuration = tf.estimator.RunConfig(model_dir=model_dir,
                                           log_step_count_steps = 10,
                                           save_summary_steps = 5)
        self.tof_net = tf.estimator.Estimator(model_fn=tof_net_func, config=self.configuration)

    #######################################################
    def onNewData(self, data):
        p = data
        self.queue.put(p)

    #######################################################
    def paint (self, data):
        self.lock.acquire()

        zImage = np.zeros((data.height,data.width),np.float32)
        grayImage = np.zeros((data.height,data.width),np.float32)

        k = 0
        xVal = 0
        yVal = 0
        for x in zImage:        
            for y in x:
                if data.getDepthConfidence(k)> 0:
                    zImage[xVal,yVal] = self.adjustZValue(data.getZ(k))
                    grayImage[xVal,yVal] = self.adjustGrayValue(data.getGrayValue(k))
                k = k + 1
                yVal=yVal+1
            yVal = 0
            xVal = xVal+1

        zImage32 = np.reshape(zImage, (171, 224, 1, 1))
        grayImage32 = np.reshape(grayImage, (171, 224, 1, 1))
        self.depth_and_amplitude = np.concatenate((zImage32,grayImage32), axis=-1)

        zImage8 = np.uint8(zImage)
        grayImage8 = np.uint8(grayImage)

        # Apply undistortion
        if self.undistortImage:
            zImage8 = cv2.undistort(zImage8,self.cameraMatrix,self.distortionCoefficients)
            grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)

        # Show the images in real-time
        win1_name = 'Depth'
        cv2.moveWindow(win1_name, 300, 100)
        zImage8 = cv2.resize(zImage8, (320, 224))
        cv2.imshow(win1_name, zImage8)
        zImage8 = np.expand_dims(zImage8, axis=-1)

        win2_name = 'Amplitude'
        cv2.moveWindow(win2_name, 700, 100)
        grayImage8 = cv2.resize(grayImage8, (320, 224))
        cv2.imshow(win2_name, grayImage8)
        grayImage8 = np.expand_dims(grayImage8, axis=-1)

        self.lock.release()
        self.done = True
        
        # MPI-Denoising of the data acquired
            # Version 1
        thread = threading.Thread(target=self.denoising)
        thread.start()
        thread.join()
            # Version 2
        # self.denoising()

    #######################################################
    def setLensParameters(self, lensParameters):

        # Construct the camera matrix
        self.cameraMatrix = np.zeros((3,3),np.float32)
        self.cameraMatrix[0,0] = lensParameters['fx']
        self.cameraMatrix[0,2] = lensParameters['cx']
        self.cameraMatrix[1,1] = lensParameters['fy']
        self.cameraMatrix[1,2] = lensParameters['cy']
        self.cameraMatrix[2,2] = 1

        # Construct the distortion coefficients
        self.distortionCoefficients = np.zeros((1,5),np.float32)
        self.distortionCoefficients[0,0] = lensParameters['k1']
        self.distortionCoefficients[0,1] = lensParameters['k2']
        self.distortionCoefficients[0,2] = lensParameters['p1']
        self.distortionCoefficients[0,3] = lensParameters['p2']
        self.distortionCoefficients[0,4] = lensParameters['k3']

    #######################################################
    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()
    
    #######################################################
    def adjustZValue(self,zValue):
        clampedDist = min(2.5,zValue)
        newZValue = clampedDist / 2.5 * 255
        return newZValue

    #######################################################
    def adjustGrayValue(self,grayValue):
        clampedVal = min(100,grayValue)
        newGrayValue = clampedVal / 180 * 255
        return newGrayValue

    #######################################################
    def denoising(self):
        depth_and_amplitude = self.depth_and_amplitude[None,:,:,:,:]
        result = list(self.tof_net.predict(
                      input_fn=lambda: ds.get_input_fn(depth_and_amplitude, shuffle=False, repeat_count=1),
                      checkpoint_path=model_dir + '/model.ckpt-200000',
                      yield_single_examples=False))
        root_dir = output_dir
        pred_depth_dir = os.path.join(root_dir, 'pred_depth')
        pred_depth_png_dir = os.path.join(root_dir, 'pred_depth_png')
        depth_input_dir = os.path.join(root_dir, 'depth_input')
        depth_input_png_dir = os.path.join(root_dir, 'depth_input_png')

        if not os.path.exists(pred_depth_dir):
            os.mkdir(pred_depth_dir)
        if not os.path.exists(pred_depth_png_dir):
            os.mkdir(pred_depth_png_dir)
        if not os.path.exists(depth_input_dir):
            os.mkdir(depth_input_dir)
        if not os.path.exists(depth_input_png_dir):
            os.mkdir(depth_input_png_dir)

        for i in range(len(result)):
            pred_depth_path = os.path.join(pred_depth_dir, str(i))
            pred_depth_png_path = os.path.join(pred_depth_png_dir, str(i)+'.png')
            depth_input_path = os.path.join(depth_input_dir, str(i))
            depth_input_png_path = os.path.join(depth_input_png_dir, str(i)+'.png')

            pred_depth = np.squeeze(result[i]['depth'])
            pred_depth_png = pred_depth * 100
            # print(pred_depth_png.shape)

            input_depth = np.squeeze(result[i]['depth_input'])
            input_depth_png = input_depth * 100
            # print(input_depth_png.shape)

            pred_depth = np.reshape(pred_depth, -1).astype(np.float32)
            input_depth = np.reshape(input_depth, -1).astype(np.float32)

            # print(input_depth.shape)
            # print(pred_depth.shape)

            pred_depth.tofile(pred_depth_path)
            pred_depth_png = Image.fromarray(pred_depth_png)
            # pred_depth_png = Image.fromarray((pred_depth_png * 1).astype(np.uint8))
            pred_depth_png = pred_depth_png.convert("L")
            pred_depth_png.save(pred_depth_png_path)

            input_depth.tofile(depth_input_path) 
            input_depth_png = Image.fromarray(input_depth_png)
            # input_depth_png = Image.fromarray((input_depth_png * 1).astype(np.uint8))
            input_depth_png = input_depth_png.convert("L")
            input_depth_png.save(depth_input_png_path)


#######################################################
def process_event_queue (q, painter, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:           
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range (0, len (q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            break
        else:
            painter.paint(item)
            currentKey = cv2.waitKey(1)
            if currentKey == ord('d'):
                painter.toggleUndistort()
            if currentKey == 27:             # Close if escape key pressed
                break