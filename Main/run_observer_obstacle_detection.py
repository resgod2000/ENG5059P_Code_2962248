#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#-------------------------Lab 4 - Obstacle Detection--------------------------#
#-----------------------------------------------------------------------------#

import os
import sys
import subprocess
from pathlib import Path

# Imports
from pal.products.qbot_platform import (
    QBotPlatformDriver,
    Keyboard,
    QBotPlatformCSICamera,
    QBotPlatformLidar,
    IS_PHYSICAL_QBOTPLATFORM
)
from hal.content.qbot_platform_functions import QBPVision, QBPRanging
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import numpy as np
import cv2
from pal.utilities.math import Calculus
from qlabs_setup import setup

RAW_PLOT_MAX_RANGE_METERS = 400.0 / 50.0  # frameSize / pixelsPerMeter from observer0
PROCESSED_PLOT_MAX_RANGE_METERS = 400.0    # second plot uses pixelsPerMeter=1
MAX_STREAM_POINTS = 256  # keep payload small enough for Observer stream buffers

EMBEDDED_OBSERVER_SCRIPT = """
from pal.utilities.probe import Observer

observer = Observer()
observer.add_plot(numMeasurements=1680,
                  frameSize=400,
                  pixelsPerMeter=50,
                  scalingFactor=4,
                  name='Leishen M10P Lidar')
observer.add_plot(numMeasurements=410,
                  frameSize=400,
                  pixelsPerMeter=50,
                  scalingFactor=1,
                  name='Plotting')
observer.launch()
"""


def launch_embedded_observer():
    """Start the Observer UI using the embedded observer0 definition."""
    cmd = [sys.executable, "-u", "-c", EMBEDDED_OBSERVER_SCRIPT]
    return subprocess.Popen(cmd)


def sanitize_lidar_ranges(ranges, max_range):
    """Clamp lidar ranges to keep observer math from overflowing."""
    if ranges is None:
        return None
    safe = np.nan_to_num(
        np.asarray(ranges, dtype=np.float64),
        nan=0.0,
        posinf=max_range,
        neginf=0.0,
    )
    return np.clip(safe, 0.0, max_range)


def sanitize_lidar_angles(angles):
    """Ensure angles stay finite for observer rendering."""
    if angles is None:
        return None
    return np.nan_to_num(np.asarray(angles, dtype=np.float64), nan=0.0)


def compress_polar_data(ranges, angles, max_points=MAX_STREAM_POINTS):
    """Down-sample polar data so Observer's network buffers remain within limits."""
    if ranges is None or angles is None:
        return None, None
    rng = np.asarray(ranges, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64)
    n = min(len(rng), len(ang))
    if n == 0:
        return None, None
    if n <= max_points:
        return rng[:n], ang[:n]
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return rng[idx], ang[idx]


# Section A - Setup

observer_proc = None
try:
    observer_proc = launch_embedded_observer()
    time.sleep(2.0)
except Exception as exc:
    observer_proc = None
    print(f"Warning: Unable to auto-start Observer ({exc}). Start it manually if required.")

setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)
ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype = np.float64), 0, True
frameRate, sampleRate = 60.0, 1/60.0
counter, counterDown, counterLidar = 0, 0, 0
endFlag, obstacle, offset, forSpd, turnSpd = False, False, 0, 0, 0
startTime = time.time()
def elapsed_time():
    return time.time() - startTime
timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

try:
    # Section B - Initialization
    myQBot       = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam      = QBotPlatformCSICamera(frameRate=frameRate, exposure = 39.0, gain=17.0)
    lidar        = QBotPlatformLidar()
    keyboard     = Keyboard()
    vision       = QBPVision()
    ranging      = QBPRanging()
    probe        = Probe(ip = ipHost)
    # probe.add_display(imageSize = [200, 320, 1], scaling = True, scalingFactor= 2, name='Raw Image')
    probe.add_plot(numMeasurements=1680, scaling=True, scalingFactor=4, name='Raw Lidar')
    probe.add_plot(numMeasurements=410, scaling=False, name='Plotting Lidar')
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    # Main loop
    while noKill and not endFlag:
        t = elapsed_time()

        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # Keyboard Driver
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space
                lineFollow = keyboard.k_7
                if obstacle:
                    arm = 0
                keyboardCmd = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # Section C - toggle line following
            if not lineFollow:     
                forSpd=keyboardCmd[0]
                turnSpd=keyboardCmd[1]
                commands = np.array([forSpd, turnSpd], dtype = np.float64) # robot spd command
            else:
                commands = np.array([forSpd, turnSpd], dtype = np.float64) # robot spd command
            # QBot Hardware
            newHIL = myQBot.read_write_std(timestamp = time.time() - startTime,
                                            arm = arm,
                                            commands = commands, userLED=obstacle)
            if newHIL:
                timeHIL     = time.time()
                newDownCam       = downCam.read()
                newLidar         = lidar.read()

                if newDownCam:
                    counterDown += 1

                    # Section D - Image processing 
                    # Undistort and resize the image
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm = cv2.resize(undistorted, (320, 200))

                    # Subselect a part of the image and perform thresholding
                    binary = vision.subselect_and_threshold(gray_sm, 50, 100, 225, 255)

                    # Blob Detection via Connected Component Labeling
                    col, row, area = vision.image_find_objects(binary, 8, 500, 2000)

                    # Speed command from blob information
                    if lineFollow:
                        forSpd, turnSpd = line2SpdMap.send((col, 0.5, 0.4))

                if newLidar:
                    counterLidar += 1

                    # Section E - LiDAR processing 

                    rangesC, anglesC = ranging.adjust_and_subsample(lidar.distances, lidar.angles,1260,3)
                    rangesC, anglesC = ranging.correct_lidar([8.75*0.0254, 0], rangesC, anglesC)
                   
                    # Section F - Obstacle detection
                    
                    rangesP, anglesP, obstacle = ranging.detect_obstacle(rangesC, anglesC, forSpd,0.7, turnSpd, 10, 0.2858, 10)
            
                # if counterDown%4 == 0:
                #     sending = probe.send(name='Raw Image', imageData=gray_sm)
                if counterLidar%2 == 0:
                    safe_raw_ranges = sanitize_lidar_ranges(lidar.distances, RAW_PLOT_MAX_RANGE_METERS)
                    safe_raw_angles = sanitize_lidar_angles(np.pi/2 - lidar.angles)
                    safe_raw_ranges, safe_raw_angles = compress_polar_data(
                        safe_raw_ranges, safe_raw_angles
                    )
                    if safe_raw_ranges is not None and safe_raw_angles is not None:
                        sending = probe.send(name='Raw Lidar', lidarData=(safe_raw_ranges, safe_raw_angles))
                if counterLidar%2 == 1:
                    safe_plot_ranges = sanitize_lidar_ranges(rangesP, PROCESSED_PLOT_MAX_RANGE_METERS)
                    safe_plot_angles = sanitize_lidar_angles(anglesP)
                    safe_plot_ranges, safe_plot_angles = compress_polar_data(
                        safe_plot_ranges, safe_plot_angles
                    )
                    if safe_plot_ranges is not None and safe_plot_angles is not None:
                        sending = probe.send(name='Plotting Lidar', lidarData=(safe_plot_ranges, safe_plot_angles))

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('User interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    lidar.terminate()
    downCam.terminate()
    myQBot.terminate()
    keyboard.terminate()
    probe.terminate()
    if observer_proc is not None:
        observer_proc.terminate()
        try:
            observer_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            observer_proc.kill()
    cv2.destroyAllWindows()
