import tensorflow as tf
import numpy as np
import time
import queue
import threading
import cv2
import roypy

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/'
output_dir = root_dir + 'Results'
model_dir = root_dir + 'SHARP-Net'

#######################################################
class MyListener(roypy.IDepthDataListener):
    
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q
        self.depth_and_amplitude = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 224, 320, 1, 2))
        # self.thread = threading.Thread(target=self.denoising, args=self.depth_and_amplitude)

    def onNewData(self, data):
        p = data
        self.queue.put(p)

    def paint (self, data):
        # mutex to lock out changes to the distortion while drawing
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

        print(zImage.shape)
        print(np.uint8(zImage).shape)
        zImage32 = np.reshape(zImage, (171, 224, 1, 1))
        grayImage32 = np.reshape(grayImage, (171, 224, 1, 1))
        # depth_and_amplitude = np.concatenate((zImage32,grayImage32), axis=-1)
        self.depth_and_amplitude = np.concatenate((zImage32,grayImage32), axis=-1)

        zImage8 = np.uint8(zImage)
        grayImage8 = np.uint8(grayImage)

        # apply undistortion
        if self.undistortImage:
            zImage8 = cv2.undistort(zImage8,self.cameraMatrix,self.distortionCoefficients)
            grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)

        # finally show the images
        # cv2.imshow('Depth',zImage8)
        # cv2.imshow('Gray',grayImage8)
        
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
        
        thread = threading.Thread(target=self.denoising)
        thread.start()
        # thread.join()

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

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()
    
    def adjustZValue(self,zValue):
        clampedDist = min(2.5,zValue)
        newZValue = clampedDist / 2.5 * 255
        return newZValue
    
    def adjustGrayValue(self,grayValue):
        clampedVal = min(100,grayValue)
        newGrayValue = clampedVal / 180 * 255
        return newGrayValue

    def denoising(self):
        depth_and_amplitude = self.depth_and_amplitude[None,:,:,:,:]
        utils.denoising(result_path=output_dir, pred=depth_and_amplitude, 
                        model_dir=model_dir, checkpoint_steps='200000')


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
