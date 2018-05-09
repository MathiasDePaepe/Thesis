import psutil
# setup logging
import logging
logging.basicConfig(level = logging.INFO)

## import the package
import pyrealsense as pyrs
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import sys
import time


def calculate(t1, t2):
    # from psutil.cpu_percent()
    # see: https://github.com/giampaolo/psutil/blob/master/psutil/__init__.py
    t1_all = sum(t1)
    t1_busy = t1_all - t1.idle
    t2_all = sum(t2)
    t2_busy = t2_all - t2.idle
    if t2_busy <= t1_busy:
        return 0.0
    busy_delta = t2_busy - t1_busy
    all_delta = t2_all - t1_all
    busy_perc = (busy_delta / all_delta) * 100
    return round(busy_perc, 1)

print("Start CPU calculation")
cpu_time_a = (time.time(), psutil.cpu_times())

print("-" * 25)

## parameters

hfov = 54.4
vfov = 42
a = 0.0584
l = 0.71        # length of drone with 10cm safety space each side

class Object:
    def __init__(self, x, y, w, h, d):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.d = d

    def get_attributes(self):
        return self.x, self.y, self.w, self.h, self.d

def get_objects(color, depth, threshold1, threshold2):

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    surf = cv2.xfeatures2d.SURF_create(500)

    # find and draw the keypoints
    kp = surf.detect(blur,None)

    pts = [p.pt for p in kp]
    xpts = []
    ypts = []

    ## evaluate the keypoints and only save the keypoints who are between the given threshold
    depth_values = []
    for i in range(0,len(pts)):
        xco = int(pts[i][0])
        yco = int(pts[i][1])
        depth_value = depth[yco][xco]
        if depth_value >= float(threshold1) and depth_value <= float(threshold2):
            xpts.append(xco)
            ypts.append(yco)
            depth_values.append(depth_value)

    ## make histogram of x coordinates of the saved keypoints
    n,  distr,  _ = plt.hist(xpts)

    ## evaluate the histogram and make seperate arrays for the different objects
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
    ## determine the objects with the previous calculated arrays
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

        object = Object(x, y, w, h, depth_mean)
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

## returns the estimatated correction in pixels between the color and depth image
def get_correction(d, a, hfov,  img_x):
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
    distance = obj2.x - (obj1.x+obj1.w)
    d = (obj1.d+obj2.d)/2
    width = 2 * d*math.tan((hfov/2)*math.pi/180)
    return (distance*width) / img_x

def is_endpoint(color):
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


def main():
    ## start the service - also available as context manager
    serv = pyrs.Service()

    ## create a device from device id and streams of interest
    dev = serv.Device()

    ## retrieve 60 frames of data as "warm up"
    for i in range(60):
        dev.wait_for_frames()

    color = dev.color
    depth = dev.depth*dev.depth_scale

    img_y,  img_x = depth.shape

    if len(sys.argv)<3:
        threshold1 = 0.6
        threshold2 = 2
    else:
        threshold1 = sys.argv[1]
        threshold2 = sys.argv[2]

    try:
        endpoint = is_endpoint(color)

        if not endpoint:
            objects = get_objects(color, depth, threshold1, threshold2)

            for i in range(len(objects)):
                x, y, w, h, d = objects[i].get_attributes()
                print(x, y, w, h, d)
                try:
                    real_x,  real_y = get_dimensions(d, w, h, 70, 43, 640, 480)
                    real_x = round(real_x, 3)
                    real_y = round(real_y, 3)

                except:
                    real_x,  real_y = 'ERROR'

                print('depth = ' + str(d) + 'm')
                print('width = ' + str(real_x)+ 'm')
                print('height = ' + str(real_y)+ 'm')

                if(i < len(objects)-1):
                    ## distance between left and right object
                    distance = round(distance_between_objects(objects[i], objects[i+1], 70, 640), 3)
                    print('distance between objects = ' + str(distance) + 'm')
        else:
            print('Endpoint reached!')

    except:
        print('no objects detected between ' + str(threshold1) + 'm and '+ str(threshold2) + 'm')

    ## stop device and service
    dev.stop()
    serv.stop()

if __name__ == "__main__":
    main()

print("-" * 25)

cpu_time_b = (time.time(), psutil.cpu_times())
print('CPU used in', cpu_time_b[0] - cpu_time_a[0], 'seconds: ',calculate(cpu_time_a[1], cpu_time_b[1]))
print(psutil.virtual_memory())
