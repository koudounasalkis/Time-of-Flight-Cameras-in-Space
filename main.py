from logging import root
import queue
import argparse
import sys

import open3d as o3d
from roypy_sample_utils import CameraOpener, add_camera_opener_options, select_use_case
import camera_imagedata_3d as pf

root_dir = '/Users/alkiskoudounas/Desktop/MasterThesis/Code/libroyale/python/pcd_registration/'
seconds = 1

#######################################################
def uploading_target_pcd():
    mesh = o3d.io.read_triangle_mesh(root_dir + 'satellite_model.stl', print_progress=True)
    return mesh.sample_points_poisson_disk(3)


#######################################################
def print_camera_info (cam, id=None):
    print("\n------------------------------ PRINTING CAMERA INFO ------------------------------\n")
    if id:
        print("Id:              " + id)
    print("Type:            " + cam.getCameraName())
    print("Width:           " + str(cam.getMaxSensorWidth()))
    print("Height:          " + str(cam.getMaxSensorHeight()))
    print("Operation modes: " + str(cam.getUseCases().size()))
    listIndent = "    "
    noteIndent = "        "
    useCases = cam.getUseCases()
    for u in range(useCases.size()):
        print(listIndent + useCases[u])
        numStreams = cam.getNumberOfStreams(useCases[u])
        if (numStreams > 1):
            print(noteIndent + "this operation mode has " + str(numStreams) + " streams")
    try:
        lensparams = cam.getLensParameters()
        print("Lens parameters: " + str(lensparams.size()))
        for u in lensparams:
            print(listIndent + "('" + u + "', " + str(lensparams[u]) + ")")
    except:
        print("Lens parameters not found!")
    camInfo = cam.getCameraInfo()
    print("CameraInfo items: " + str(camInfo.size()))
    for u in range(camInfo.size()):
        print(listIndent + str(camInfo[u]))


#######################################################
def main(target_pcd):
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    options = parser.parse_args()
   
    opener = CameraOpener(options)

    try:
        cam = opener.open_camera ()
    except:
        print("could not open Camera Interface")
        sys.exit(1)

    print_camera_info (cam)
    print("\n----------------------------- CONNECTING THE CAMERA ----------------------------\n")
    curUseCase = select_use_case(cam)

    try:
        # Retrieve the interface available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount: ", replay.frameCount())
        print ("File version: ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")

    q1 = queue.Queue()
    q2 = queue.Queue()
    q3 = queue.Queue()
    l = pf.MyListener(q1, q2, q3)

    # cam.registerDepthImageListener(l)  # DEPTH
    cam.registerDataListener(l)          # OPENCV (DEPTH + AMPLITUDE) + OPEN3D
    print ("Setting use case : " + curUseCase)
    cam.setUseCase(curUseCase)

    print(f"\n------------------------ START STREAMING FOR {seconds} SECONDS ------------------------\n") 
    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)
    pf.process_event_queue(q1, q2, q3, l, target_pcd, seconds)

    cam.stopCapture()
    print(f"\n------------------- STOP STREAMING AS {seconds} SECONDS HAVE PASSED -------------------\n") 


#######################################################
if (__name__ == "__main__"):
    print("\n-------------------------- TARGET POINT CLOUD DETECTION --------------------------")
    print("\nUploading target point cloud...")
    target_pcd = uploading_target_pcd()
    print("Target point cloud successfully uploaded!\n")
    main(target_pcd)