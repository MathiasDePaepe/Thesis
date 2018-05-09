"""
Copyright 2017-2018, Mathias De Paepe.
This mission is the script for objectdetection and objectavoidance with pathplanning
to reach an endpoint.
The working principle of the script is as follows:
1) The drone takes off to an altitude of 1m.
2) The drone takes a picture.
3) With picture is determined if endpoint is reached or how many objects to avoid.
4) If there are no objects, the drone moves a distance forward.
   If there are objects, the drone swerfs left or right and moves past the object
   in the direction of the endpoint.
5) When endpoint is reached, the drone will land.

# NOTE: The initial distance between drone and endpoint must be hardcoded,
        otherwise it's not possible to fly in the right direction.
# NOTE: Be safe when testing. A lot of unexpected behaviors may occur in GUIDED_NOGPS mode.
        Always watch the drone movement, and make sure that you are in dangerless environment.
        Land the drone as soon as possible when it shows any unexpected behavior.

Tested in Python 2.7.12
# NOTE: following packages installed:
    - dronekit: sudo pip install dronekit
    - opencv: sudo apt-get install python-pip python-opencv python-opencv-apps python-zbar
              zbar-tools vim-python-jedi vim-python-jedi python-editor eric idle vim-nox-py2
    - numpy: sudo pip install Cython numpy
    - pyrealsense: sudo pip install pyrealsense
    - matplotlib: python -mpip install -U matplotlib

Full documentation is provided in the thesis
"Depth camera based sense and avoid for drones" of UGent
"""

## Bring print function of Python 3 in Python 2.7

from __future__ import print_function

## Imported packages for movement

from dronekit import connect, VehicleMode
import time

## Imported packages for R200 camera and image processing

import pyrealsense as pyrs
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys


## Set up option parsing to get connection string

import argparse
parser = argparse.ArgumentParser(description='Command to connect drone')
parser.add_argument('--connect',
                    help="Vehicle connection target string (example: 'tcp:127.0.0.1:5760').")
args = parser.parse_args()
connection_string = args.connect

## Quit program when connection_string is not given as argument

if not connection_string:
    print("ERROR: argument not given or invalid")
    sys.exit()


#################################   CONSTANTS  ##################################################

hfov = 54.4     # horizontal field of view VGA resolution (in degrees)
vfov = 42       # vertical field of view VGA reolution (in degrees)
a = 0.0584      # distance between colorcamera and IR2-camera (in meters)
l = 0.71        # length of drone with 10cm safety space each side (in meters)

THRESHOLD1 = 0.6    # the closest distance where the camera can detect an object (in meters)
THRESHOLD2 = 2      # the farest distance where the camera can detect an object (in meters)
D_ENDPOINT = 200    # the initiate distance between the drone and the endpoint (in meters)

DEFAULT_TAKEOFF_THRUST = 0.7
SMOOTH_TAKEOFF_THRUST = 0.6

#################################   OBJECT DETECTION  ###########################################

class DetectedObject:
    def __init__(self, x, y, w, h, d):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.d = d

    def get_attributes(self):
        return self.x, self.y, self.w, self.h, self.d

def get_objects(color, depth, threshold1, threshold2):
    """
    Uses the SURF algoritm to detect the objects
    """

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    surf = cv2.xfeatures2d.SURF_create(500)

    # find and draw the keypoints
    kp = surf.detect(blur,None)

    pts = [p.pt for p in kp]
    xpts = []
    ypts = []

    # evaluate the keypoints and only save the keypoints who are between the given threshold
    depth_values = []
    for i in range(0,len(pts)):
        xco = int(pts[i][0])
        yco = int(pts[i][1])
        depth_value = depth[yco][xco]
        if depth_value >= float(threshold1) and depth_value <= float(threshold2):
            xpts.append(xco)
            ypts.append(yco)
            depth_values.append(depth_value)

    # make histogram of x coordinates of the saved keypoints
    n,  distr,  _ = plt.hist(xpts)
    plt.savefig('hist.png')

    # evaluate the histogram and make seperate arrays for the different objects
    objectarray = []
    temp = []
    for i in range(len(n)):
        if n[i] > 0:
            temp.append(distr[i])
            temp.append(distr[i+1])
        else:
                if len(temp)!=0:
                    objectarray.append(temp)
                temp = []
    objectarray.append(temp)

    objects = []

    # determine the objects with the previous calculated arrays
    for i in range(len(objectarray)):
        y_values = []
        min_x = int(np.amin(objectarray[i]))
        max_x = int(np.amax(objectarray[i]))

        for j in range(len(xpts)):
            if xpts[j] > min_x and xpts[j] < max_x:
                y_values.append(ypts[j])

        min_y = int(np.amin(y_values))
        max_y = int(np.amax(y_values))
        x = min_x
        y = min_y
        w = max_x - min_x
        h = max_y - min_y

        depth_mean = round(get_depth_mean(depth, x, y, w, h), 3)

        object = DetectedObject(x, y, w, h, depth_mean)
        objects.append(object)

    return objects

def get_depth_mean(depth, x, y, w, h):

    depth_values = []
    for i in range(y, y+h):
        for j in range(x, x+w):
            depth_value = depth[i][j]
            if depth_value > 0:
                depth_values.append(depth_value)
    return np.mean(depth_values)

def get_correction(d, a, hfov,  img_x):
    """
    Returns the estimatated correction in pixels between the color and depth image
    """

    width = 2 * d*math.tan((hfov/2)*math.pi/180) # in meters
    one_meter = img_x / width
    return int(a*one_meter)




def get_dimensions(d, w, h, hfov, vfov,  img_x, img_y):

    width = 2 * d*math.tan((hfov/2)*math.pi/180)
    height = 2 * d*math.tan((vfov/2)*math.pi/180)

    real_x = (w*width) / img_x
    real_y = (h*height) / img_y
    return real_x,  real_y

def distance_between_objects(obj1, obj2, hfov, img_x):
    """
    Returns the estimatated distance between two objects
    """

    distance = obj2.x - (obj1.x+obj1.w)
    d = (obj1.d+obj2.d)/2
    width = 2 * d*math.tan((hfov/2)*math.pi/180)
    return (distance*width) / img_x

def draw_bounding_box(objects,color):
    """
    Draws a bounding box for each object on the color image
    """

    for i in range(len(objects)):
        x, y, w, h, d = objects[i].get_attributes()
        print(x, y, w, h, d)
        corr = get_correction(d, a, hfov, x)
        cv2.rectangle(color, (x-corr, y), (x+w-corr, y+h), (0, 255, 0), 4)

        try:
            real_x,  real_y = get_dimensions(d, w, h, hfov, vfov, 640, 480)
            real_x = round(real_x, 3)
            real_y = round(real_y, 3)

        except:
            real_x,  real_y = 'ERROR'

        cv2.putText(color, 'depth = ' + str(d) + 'm', (30, i*60 + 30) ,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(color, 'width = ' + str(real_x)+ 'm', (30, i*60 + 45) ,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(color, 'height = ' + str(real_y)+ 'm', (30, i*60 + 60) ,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if(i < len(objects)-1):
            ## distance between left and right object
            distance = round(distance_between_objects(objects[i], objects[i+1], hfov, 640), 3)
            if distance > l:
                textcolor =  (0, 255, 0)
            else:
                textcolor =  (0, 0, 255)

            cv2.putText(color, 'distance between objects = ' + str(distance) + 'm',
                                (320, i*60 + 70) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1)

def is_endpoint(color):
    """
    Determines with the taken picture if the endpoint has reached or not
    The endpoint is a white cirkel and the method returns true if the endpoint is reached
    """

    img = cv2.cvtColor(color,  cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(img,(5,5),0)

    lower_range = np.array([175, 175, 175], dtype=np.uint8)
    upper_range = np.array([255, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(blur, lower_range, upper_range)
    res = cv2.bitwise_and(img,img, mask= mask)

    bilateral_filtered_image = cv2.bilateralFilter(res, 5, 175, 175)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

    _, contours, _= cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
    	approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    	area = cv2.contourArea(contour)
    	if ((len(approx) > 8) & (area > 10000) & (area < 30000)):
    		contour_list.append(contour)

    if not len(contour_list)==0:
    	return True
    else:
    	return False

def take_picture(threshold1, threshold2):
    """
    Takes a picture of the scene and returns:
    * if the endpoint is reached or not
    * the objects between the thresholds
    * the dimensions of the picture
    """

    # start the service - also available as context manager
    serv = pyrs.Service()

    # create a device from device id and streams of interest
    dev = serv.Device()

    # retrieve 60 frames of data as "warm up"
    for i in range(60):
        dev.wait_for_frames()

    print("Taking a picture")

    color = dev.color
    color = cv2.cvtColor(color,  cv2.COLOR_RGB2BGR)
    depth = dev.depth*dev.depth_scale

    img_y,  img_x = depth.shape

    endpoint = is_endpoint(color)
    objects = []

    if not endpoint:
        try:
            objects = get_objects(color, depth, threshold1, threshold2)
            draw_bounding_box(objects,color)
        except:
            cv2.putText(color, 'no objects detected between ' + str(threshold1) + 'm and '
                + str(threshold2) + 'm', (30, 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(color, 'Endpoint reached!', (30, 30) ,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite('picture.png',color)

    ## stop device and service
    dev.stop()
    serv.stop()

    return endpoint, objects




#################################   PATH PLANNING  ##############################################


def arm_and_takeoff_nogps(aTargetAltitude):
    """
    Arms and takes off drone without GPS connection
    """

    print("Taking of to an altitude of ", str(aTargetAltitude))

    print("Basic pre-arm checks")

    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED_NOGPS mode
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        vehicle.armed = True
        time.sleep(1)

    print("Taking off!")

    thrust = DEFAULT_TAKEOFF_THRUST
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        print(" Altitude: %f  Desired: %f" %
              (current_altitude, aTargetAltitude))
        if current_altitude >= aTargetAltitude*0.95: # Trigger just below target alt.
            print("Reached target altitude")
            break
        elif current_altitude >= aTargetAltitude*0.6:
            thrust = SMOOTH_TAKEOFF_THRUST
        set_attitude(thrust = thrust)
        time.sleep(0.2)


def set_attitude(roll_angle = 0.0, pitch_angle = 0.0, yaw_rate = 0.0, thrust = 0.5, duration = 0):
    """
    This is similar to the movement of the controller by the user,
    contrary that this is done by code
    """

    # Thrust >  0.5: Ascend
    # Thrust == 0.5: Hold the altitude
    # Thrust <  0.5: Descend
    msg = vehicle.message_factory.set_attitude_target_encode(
        0, # time_boot_ms
        1, # Target system
        1, # Target component
        0b00000000, # Type mask: bit 1 is LSB
        to_quaternion(roll_angle, pitch_angle), # Quaternion
        0, # Body roll rate in radian
        0, # Body pitch rate in radian
        math.radians(yaw_rate), # Body yaw rate in radian
        thrust  # Thrust
    )
    vehicle.send_mavlink(msg)

    start = time.time()
    while time.time() - start < duration:
        vehicle.send_mavlink(msg)
        time.sleep(0.1)

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

def land_and_disarm():

    print("Setting LAND mode")
    vehicle.mode = VehicleMode("LAND")
    vehicle.flush()
    # Disarm vehicle
    print("Disarming motors")
    vehicle.armed = False
    time.sleep(1)

def display_basic_vehicle_info(vehicle):

    print("Basic vehicle information:")
    print(" Attitude: ", vehicle.attitude)
    print(" Type: ", vehicle._vehicle_type)
    print(" Armed: ", vehicle.armed)
    print(" Is Armable?: ", vehicle.is_armable)
    print(" System status: ", vehicle.system_status.state)
    print(" GPS: ", vehicle.gps_0)
    print(" Battery: ", vehicle.battery)
    print(" Alt: ", vehicle.location.global_relative_frame.alt)


"""
The move functions move_forward, move_backward, move_left, move_right
and yaw are temporally incorrect implemented.
In a future work they have to be completely and correctly implemented.
"""

def move_forward(distance):

    print("Moving forward")
    set_attitude(pitch_angle = -5, thrust = 0.5, duration = 3)

def move_backward(distance):

    print("Moving backward")
    set_attitude(pitch_angle = 5, thrust = 0.5, duration = 3)

def move_left(distance):

    print("Moving left")
    set_attitude(roll_angle = -5, thrust = 0.5, duration = 3)

def move_right(distance):

    print("Moving right")
    set_attitude(roll_angle = 5, thrust = 0.5, duration = 3)



def yaw(degrees):
    # positive yaw means right, negative yaw means left
    print("Yawing vehicle")
    set_attitude(yaw_angle = degrees, thrust = 0.5, duration = 3)



def turn_to_endpoint(previous_direction, w_real, d, d_traveled):
    """
    Turns the drone to the endpoint when an obstacle was avoided.
    Parameter previous_direction can be left(0) or right(1)
    """
    theta = math.atan((D_ENDPOINT-d_traveled)/w_real)
    phi = np.pi/2 - theta

    if previous_direction==0:
        # drone yaws to the right to the direction of the endpoint
        yaw(phi)
    else:
        # drone yaws to the left to the direction of the endpoint
        yaw(-phi)
        phi = -phi

    return phi

def avoid_objects(objectnr, objects, d_traveled, phi):
"""
Overlaying function for the movement of the drone when detecting one or more get_objects.
Returns d_traveled wich is the distance that the drone has traveled from the beginning until now.
"""

    if objectnr==0:
        move_forward(THRESHOLD2 - THRESHOLD1)
        d_traveled = d_traveled + THRESHOLD2 - THRESHOLD1

    elif objectnr==1:
        x, y, w, h, d = objects[0].get_attributes()
        width = 2*d*math.tan((hfov/2)*math.pi/180)
        w_real = (w*width) / img_x
        previous_direction = None
        mean = (x+(x+w))/2

        if mean >= img_x/2:
            move_left(w_real)
            previous_direction = 0
        else:
            move_right(w_real)
            previous_direction = 1

        move_forward(d)
        d_traveled = d_traveled + d*np.cos(phi)
        phi = turn_to_endpoint(previous_direction, w_real, d, d_traveled)


    # more than one object detected
    else:
        obj1 = None
        obj2 = None
        distance = 0
        # determine the 2 objects who have the greatest distance between,
        # if it's not possible to pass, then move left/right
        for i in range(len(objects)):
            distance_temp = round(distance_between_objects(objects[i],objects[i+1],hfov,640),3)
            if distance_temp > l and distance < distance_temp:
                obj1 = objects[i]
                obj2 = objects[i+1]
                distance = distance_temp

        # distance greater than zero so possibility for the drone to pass between the objects
        # objects with greatest distance between are determined
        if distance > 0:
            x1, y1, w1, h1, d1 = obj1.get_attributes()
            x2, y2, w2, h2, d2 = obj2.get_attributes()
            mean_x = (x2-(x1+w1))/2
            d = min(d1,d2)
            width = 2*d*math.tan((hfov/2)*math.pi/180)

            # right corner left object is greater than half image so objects are in right side
            if x1+w1 > img_x/2:
                w_real = ((mean_x-img_x/2)*width) / img_x
                previous_direction = 1
                move_right(w_real)

            # right corner right object is smaller than half image so objects are in left side
            elif x2 < img_x/2:
                w_real = ((img_x/2-mean_x)*width) / img_x
                previous_direction = 0
                move_left(w_real)

            # assume that half the image is between the objects
            else:
                distance1 = img_x/2 - (x1+w1)
                distance2 = x2 - img_x/2
                difference = np.absolute(distance2 - distance1)
                w_real = ((difference)*width) / img_x

                if distance1 > distance2:
                    previous_direction = 0
                    move_left(w_real)
                elif distance1 < distance2:
                    previous_direction = 1
                    move_right(w_real)
                # distance1 = distance2 so the drone swerfs in the middle of the objects
                else:
                    pass

        # drone can't pass between the objects
        # select extreme left and extreme right objects, determine to move left or right
        else:
            x1, y1, w1, h1, d1 = objects[0].get_attributes()
            x2, y2, w2, h2, d2 = objects[len(objects)].get_attributes()
            previous_direction = None

            mean_x = (x1+(x2+w2))/2

            d_values = []
            for i in range(len(objects)):
                d_values.append(objects[i].d)

            d = min(d_values)

            width = (d1+d2)*math.tan((hfov/2)*math.pi/180)
            w_real = ((w1+w2)*width) / img_x
            if mean_x >= img_x/2:
                move_left(w_real)
                previous_direction = 0
            else:
                move_right(w_real)
                previous_direction = 1

        move_forward(d)
        d_traveled = d_traveled + d*np.cos(phi)
        phi = turn_to_endpoint(previous_direction, w_real, d, d_traveled)


    return d_traveled, phi

#################################   MAIN ########################################################

def main():

    ## Connect the Vehicle
    print('Connecting to vehicle on: %s' % connection_string)
    vehicle = connect(connection_string, wait_ready=True)

    # Display the basic information of the vehicle
    display_basic_vehicle_info(vehicle)

    # Take off 1m in GUIDED_NOGPS mode.
    arm_and_takeoff_nogps(1)

    # Hold the position for 3 seconds.
    print("Holding position for 3 seconds")
    set_attitude(duration = 3)

    try:

        endpoint, objects = take_picture(THRESHOLD1, THRESHOLD2)

        d_traveled = 0
        phi_sum = 0
        while not endpoint:
            objectnr = len(objects)
            print(str(objectnr) + " object(s) detected")
            d_traveled, phi = avoid_objects(objectnr, objects, d_traveled, phi_sum)
            endpoint, objects = take_picture(THRESHOLD1, THRESHOLD2)
            phi_sum = phi_sum + phi

    except:
        pass

    land_and_disarm()

    # Close vehicle object before exiting script
    print("Close vehicle object")
    vehicle.close()

if __name__ == "__main__":
    main()
