#-----------------------------------------------------------------------------#
#--------------------Quanser Interactive Labs Setup for-----------------------#
#---------------------------Mobile Robotics Lab-------------------------------#
#-----------------(Environment: QBot Platform / Warehouse)--------------------#
#-----------------------------------------------------------------------------#

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qbot_platform import QLabsQBotPlatform
from qvl.qbot_platform_flooring import QLabsQBotPlatformFlooring
from qvl.real_time import QLabsRealTime
from qvl.basic_shape import QLabsBasicShape
import pal.resources.rtmodels as rtmodels
import time
import numpy as np
import os
import subprocess

#------------------------------ Main program ----------------------------------

def setup(
        locationQBotP       = [-3.2, 0.0, 0.1],
        rotationQBotP       = [0,0,0],
        verbose             = True,
        rtModel_workspace   = rtmodels.QBOT_PLATFORM,
        rtModel_driver      = rtmodels.QBOT_PLATFORM_DRIVER
        ):

    subprocess.Popen(['quanser_host_peripheral_client.exe', '-q'])
    time.sleep(2.0)
    subprocess.Popen(['quanser_host_peripheral_client.exe', '-uri', 'tcpip://localhost:18444'])

    qrt = QLabsRealTime()
    if verbose: print("Stopping any pre-existing RT models")
    qrt.terminate_real_time_model(rtModel_workspace)
    time.sleep(1.0)
    qrt.terminate_real_time_model(rtModel_driver)
    time.sleep(1.0)
    qrt.terminate_all_real_time_models()

    qlabs = QuanserInteractiveLabs()
    if verbose: print("Connecting to QLabs ...")
    try:
        qlabs.open("localhost")
    except:
        print("Unable to connect to QLabs")
        return
    if verbose: print("Connected!")

    qlabs.destroy_all_spawned_actors()

    #---------------------------- QBot Platform ---------------------------
    if verbose: print("Spawning QBot Platform ...")
    hQBot = QLabsQBotPlatform(qlabs)
    hQBot.spawn_id_degrees(actorNumber=0,
                        location=locationQBotP,
                        rotation=rotationQBotP,
                        scale=[1,1,1],
                        configuration=1,
                        waitForConfirmation= False)
    hQBot.possess(hQBot.VIEWPOINT_TRAILING)
    #------------------------------- Walls --------------------------------
    # (Removed for open environment)
    #--------------------- Cylindrical and Cone Obstacles --------------------
    cylinder_positions = [
        [-2.5, -1.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 1.0, 0.5],
    ]
    cone_positions = [
        [2.0, -0.5, 0.3],
        [1.5, 2.5, 0.3],
    ]
    hObstacle = QLabsBasicShape(qlabs)

    for idx, position in enumerate(cylinder_positions, start=10):
        _ = hObstacle.spawn_id(
            actorNumber=idx,
            location=position,
            rotation=[0, 0, 0],
            scale=[0.1, 0.1, 1.0],
            configuration=hObstacle.SHAPE_CYLINDER,
            waitForConfirmation=False)
        hObstacle.set_material_properties([0.8, 0.1, 0.1], waitForConfirmation=False)
        hObstacle.set_enable_collisions(True, waitForConfirmation=False)
        hObstacle.set_enable_dynamics(False, waitForConfirmation=False)

    for idx, position in enumerate(cone_positions, start=20):
        _ = hObstacle.spawn_id(
            actorNumber=idx,
            location=position,
            rotation=[0, 0, 0],
            scale=[0.3, 0.3, 0.6],
            configuration=hObstacle.SHAPE_CONE,
            waitForConfirmation=False)
        hObstacle.set_material_properties([1.0, 0.4, 0.0], waitForConfirmation=False)
        hObstacle.set_enable_collisions(True, waitForConfirmation=False)
        hObstacle.set_enable_dynamics(False, waitForConfirmation=False)
    #----------------------------- Flooring -------------------------------
    if verbose: print("Spawning flooring ...")
    floor_shape = QLabsBasicShape(qlabs)
    base_floor_color = [0.02, 0.02, 0.02]
    floor_thickness = 0.005
    floor_scale = [40.0, 30.0, floor_thickness]
    _ = floor_shape.spawn_id(
        actorNumber=500,
        location=[0, 0, floor_thickness / 2],
        rotation=[0, 0, 0],
        scale=floor_scale,
        configuration=floor_shape.SHAPE_CUBE,
        waitForConfirmation=False
    )
    floor_shape.set_material_properties(base_floor_color, waitForConfirmation=False)
    floor_shape.set_enable_collisions(False, waitForConfirmation=False)
    floor_shape.set_enable_dynamics(False, waitForConfirmation=False)

    if verbose: print("Drawing thin white track ...")
    line_shape = QLabsBasicShape(qlabs)
    line_thickness = 0.008
    line_width = 0.04
    line_color = [0.95, 0.95, 0.95]
    loop_half_width = 5.0
    loop_half_height = 3.0
    segment_length = 0.5
    segments = []
    radius_x = loop_half_width
    radius_y = loop_half_height
    for theta in np.linspace(0, 2*np.pi, 48, endpoint=False):
        x = radius_x * np.cos(theta)
        y = radius_y * np.sin(theta)
        dx = -radius_x * np.sin(theta)
        dy = radius_y * np.cos(theta)
        angle = np.arctan2(dy, dx)
        segments.append({
            "pos":[x, y, floor_thickness + line_thickness/2],
            "scale":[segment_length, line_width, line_thickness],
            "yaw":angle
        })
    actor_line_id = 600
    for segment in segments:
        _ = line_shape.spawn_id(
            actorNumber=actor_line_id,
            location=segment["pos"],
            rotation=[0, 0, segment["yaw"]],
            scale=segment["scale"],
            configuration=line_shape.SHAPE_CUBE,
            waitForConfirmation=False
        )
        line_shape.set_material_properties(line_color, waitForConfirmation=False)
        line_shape.set_enable_collisions(False, waitForConfirmation=False)
        line_shape.set_enable_dynamics(False, waitForConfirmation=False)
        actor_line_id += 1
    #----------------------------- RT Models -------------------------------
    if verbose: print("Starting RT models...")
    time.sleep(2)
    qrt.start_real_time_model(rtModel_workspace, userArguments=False)
    time.sleep(1)
    qrt.start_real_time_model(rtModel_driver, userArguments=True, additionalArguments="-uri tcpip://localhost:17098")
    if verbose: print('QLabs setup completed')
    return hQBot

if __name__ == '__main__':
    setup(locationQBotP = [-3.0, -3.0, 0.1], rotationQBotP=[0,0,90], verbose=True)
