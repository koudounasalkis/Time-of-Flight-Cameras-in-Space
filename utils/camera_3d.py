import numpy as np
import time
import queue
import roypy
import open3d as o3d


#######################################################
class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q
        self.figSetup = False
        self.firstTime = True
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()


    def onNewData(self, data):
        pc = data.npoints ()
        
        px = pc[:,:,0]
        py = pc[:,:,1]
        pz = pc[:,:,2]
        stack1 = np.stack([px,py,pz], axis=-1)
        stack2 = stack1.reshape(-1, 3)
        
        self.queue.put(stack2)


    def paint (self, data):
        
        data = data[np.all(data != 0, axis=1)]   
        vec3d = o3d.utility.Vector3dVector(data)
        
        if (self.firstTime):
            self.pointcloud = o3d.geometry.PointCloud(vec3d)
            self.vis.add_geometry(self.pointcloud)
            vc = self.vis.get_view_control()
            vc.set_front([0.,0.,-1.])
            self.firstTime = False
        
        self.pointcloud.points = vec3d
        result = self.vis.update_geometry(self.pointcloud)
        self.vis.poll_events()
        self.vis.update_renderer()


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
