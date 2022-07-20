import tensorflow as tf
import numpy as np
import time
import queue
import threading
import cv2
import roypy
import open3d as o3d
import sys
import os

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/libroyale/python/'
sys.path.insert(1, root_dir + 'mpi_denoising/sharp_net')
sys.path.insert(1, root_dir + 'registration')
sys.path.insert(1, root_dir + 'registration/fmr')
output_dir = root_dir + 'denoising_results'
model_dir = root_dir + 'mpi_denoising/sharp_net/checkpoints'

directory_output_images = r'/Users/alkiskoudounas/Desktop/MasterThesis/Images'
os.chdir(directory_output_images)
depthImage = 'depthImage.jpg'
amplitudeImage = 'amplitudeImage.jpg'
pointCloudImage = 'pointCloudImage.png'

import sharpnet_utils_2 as sharpnet
import global_registration as gr
import fast_global_registration as fgr
import fmr as fmr
import icp as icp

denoising_mode = 'medium'
registration_mode = 'learning_based'


#######################################################
class MyListener(roypy.IDepthDataListener):
    
    def __init__(self, q1, q2, q3):
        super(MyListener, self).__init__()
        
        # OPEN 3D
        self.queue1 = q1
        self.figSetup = False
        self.firstTime = True
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='PicoFlexx Point Cloud', width=1920, height=1080, 
                                left=250, top=250, visible=True)

        # DEPTH MAP + AMPLITUDE IMAGE in OPENCV
        self.queue2 = q2
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False

        # DENOISING
        self.queue3 = q3
        self.depth_to_pcd = None


    def onNewData(self, data):       
        # OPEN3D 
        pc = data.npoints ()       
        px = pc[:,:,0]
        py = pc[:,:,1]
        pz = pc[:,:,2]
        stack1 = np.stack([px,py,pz], axis=-1)
        stack2 = stack1.reshape(-1, 3)    
        self.queue1.put(stack2)

        # DEPTH MAP + AMPLITUDE IMAGE in OPENCV
        p = data
        self.queue2.put(p)
       
        # DENOISING 
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
        self.queue3.put(self.depth_and_amplitude)


    def paint (self, data, case):
        
        if case == "3d":     # OPEN3D
            data = data[np.all(data != 0, axis=1)]
            vec3d = o3d.utility.Vector3dVector(data)
            if (self.firstTime):
                self.pointcloud = o3d.geometry.PointCloud(vec3d)
                self.pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                self.pointcloud.paint_uniform_color([1, 0.706, 0])
                self.vis.add_geometry(self.pointcloud)
                vc = self.vis.get_view_control()
                # vc.set_front([0.,0.,-1.])
                self.firstTime = False
            self.pointcloud.points = vec3d
            self.pointcloud.paint_uniform_color([1, 0.706, 0])
            result = self.vis.update_geometry(self.pointcloud)
            self.vis.poll_events()
            self.vis.update_renderer()
            # self.vis.capture_screen_image(pointCloudImage)

        elif case == "cv":   # DEPTH MAP + AMPLITUDE IMAGE in OPENCV
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
            zImage8 = np.uint8(zImage)
            grayImage8 = np.uint8(grayImage)
            if self.undistortImage:
                zImage8 = cv2.undistort(zImage8,self.cameraMatrix,self.distortionCoefficients)
                grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)           
            
            win1_name = 'Depth Map'
            cv2.moveWindow(win1_name, 300, 100)
            zImage8 = cv2.resize(zImage8, (320, 224))
            zImage8 = cv2.flip(zImage8, 0)
            cv2.imshow(win1_name, zImage8)
            cv2.imwrite(depthImage, zImage8)
            zImage8 = np.expand_dims(zImage8, axis=-1)
            
            win2_name = 'Amplitude Image'
            cv2.moveWindow(win2_name, 700, 100)
            grayImage8 = cv2.resize(grayImage8, (320, 224))
            grayImage8 = cv2.flip(grayImage8, 0)
            cv2.imshow(win2_name, grayImage8)
            cv2.imwrite(amplitudeImage, grayImage8)

            grayImage8 = np.expand_dims(grayImage8, axis=-1)
            self.lock.release()
            self.done = True

        else:   
            self.denoising() 


    def setLensParameters(self, lensParameters):
        self.cameraMatrix = np.zeros((3,3),np.float32)              # Construct the camera matrix
        self.cameraMatrix[0,0] = lensParameters['fx']
        self.cameraMatrix[0,2] = lensParameters['cx']
        self.cameraMatrix[1,1] = lensParameters['fy']
        self.cameraMatrix[1,2] = lensParameters['cy']
        self.cameraMatrix[2,2] = 1

        self.distortionCoefficients = np.zeros((1,5),np.float32)    # Construct the distortion coefficients
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
        if denoising_mode == 'simple': 
            depth_raw, amplitude_raw = tf.unstack(self.depth_and_amplitude, axis=-1)      
            resize = tf.compat.v1.keras.layers.Resizing(320, 224, interpolation='bilinear', 
                                                        crop_to_aspect_ratio=False)       
            depth_raw = tf.reshape(resize(depth_raw), (320,224,1,1))
            amplitude_raw = tf.reshape(resize(amplitude_raw), (320,224,1,1))
            depth_and_amplitude = tf.concat([depth_raw,amplitude_raw], axis=-1)[None,:,:,:,:]
        else:
           depth_and_amplitude = self.depth_and_amplitude[None,:,:,:,:]
        
        print("---------------------- STARTING MPI AND SHOT DENOISING... ----------------------")
        pred_depth = sharpnet.denoising(pred=depth_and_amplitude, result_path=output_dir, 
                                        model_dir=model_dir, checkpoint_steps='200000',
                                        denoising_mode=denoising_mode)
        self.depth_to_pcd = pred_depth
        self.vis.update_renderer()
        print("MPI and Shot Denoising process finished!\n")


    def pcd_registration(self, target_pcd, registration_mode):
        if registration_mode == 'optimization_based':
            print("------------ STARTING OPTIMIZATION-BASED POINT CLOUD REGISTRATION... -----------")
            # icp.optimization_based_registration(self.pointcloud, target_pcd)
            fgr.optimization_based_registration(self.pointcloud, target_pcd)
        elif registration_mode == 'feature_learning':
            print("-------------- STARTING FEATURE-BASED POINT CLOUD REGISTRATION... --------------")
            gr.feature_learning_registration(self.pointcloud, target_pcd)
        else:
            print("-------- STARTING END-TO-END LEARNING-BASED POINT CLOUD REGISTRATION... --------")
            fmr.learning_based_registration(self.pointcloud, target_pcd)
        print("Point Cloud Registration process successfully completed!")


#######################################################
def process_event_queue (qopen3d, qopencv, qdenoising, listener, target_pcd, seconds):
    start = time.time()
    end = start + seconds
    while time.time() < end:
        try:           
            if len(qopencv.queue) == 0:
                itemcv = qopencv.get(True, 1)
            else:
                for i in range (0, len (qopencv.queue)):
                    itemcv = qopencv.get(True, 1)
            if len(qopen3d.queue) == 0:
                item3d = qopen3d.get(True, 1)
            else:
                for i in range (0, len (qopen3d.queue)):
                    item3d = qopen3d.get(True, 1)
            if len(qdenoising.queue) == 0:
                itemdenoising = qdenoising.get(True, 1)
            else:
                for i in range (0, len (qdenoising.queue)):
                    itemdenoising = qdenoising.get(True, 1)
        except queue.Empty:
            break
        else:
            listener.paint(itemcv, "cv")
            currentKey = cv2.waitKey(1)
            if currentKey == ord('d'):
                listener.toggleUndistort()
            if currentKey == 27:                # Close if escape key pressed
                break
            listener.paint(item3d, "3d")
            listener.paint(itemdenoising, "denoising")
            listener.pcd_registration(target_pcd, registration_mode)
    # listener.paint(itemdenoising, "denoising") 
    # listener.pcd_registration(target_pcd, registration_mode)