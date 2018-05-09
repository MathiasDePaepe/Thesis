## setup logging
import logging
logging.basicConfig(level = logging.INFO)

## import the package
import pyrealsense as pyrs
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

import time
start_time = time.time()

## parameters

hfov = 54.4
vfov = 42
a = 0.0584
l = 0.71

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
        if depth_value >= float(threshold1)
							and depth_value <= float(threshold2):
            xpts.append(xco)
            ypts.append(yco)
            depth_values.append(depth_value)

    ## make histogram of x coordinates of the saved keypoints
    n,  distr,  _ = plt.hist(xpts)
    plt.savefig('hist.png')

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

    real_w = (w*width) / img_x
    real_h = (h*height) / img_y
    return real_w,  real_h

def distance_between_objects(obj1, obj2, hfov, img_x):
    distance = obj2.x - (obj1.x+obj1.w)
    d = (obj1.d+obj2.d)/2
    width = 2 * d*math.tan((hfov/2)*math.pi/180)
    return (distance*width) / img_x

def main():
    serv = pyrs.Service()
    dev = serv.Device()
    for i in range(60):
        dev.wait_for_frames()

    color = dev.color
    color = cv2.cvtColor(color,  cv2.COLOR_RGB2BGR)
    depth = dev.depth*dev.depth_scale

    img_y,  img_x = depth.shape

    if len(sys.argv)<3:
        print('InputError: python file.py threshold1 threshold2')
        sys.exit()
    else:
        threshold1 = sys.argv[1]
        threshold2 = sys.argv[2]

    try:

        objects = get_objects(color, depth, threshold1, threshold2)

        for i in range(len(objects)):
            x, y, w, h, d = objects[i].get_attributes()
            print(x, y, w, h, d)
            corr = get_correction(d, a, hfov, x)
            cv2.rectangle(color, (x-corr, y), (x+w-corr, y+h), (0, 255, 0), 4)

            try:
                real_w,  real_h = get_dimensions(d, w, h, 70, 43, 640, 480)
                real_w = round(real_w, 3)
                real_h = round(real_h, 3)

            except:
                real_x,  real_y = 'ERROR'

            cv2.putText(color, 'depth = ' + str(d) + 'm', (30, i*60 + 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(color, 'width = ' + str(real_w)+ 'm', (30, i*60 + 45) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(color, 'height = ' + str(real_h)+ 'm', (30, i*60 + 60) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if(i < len(objects)-1):
                ## distance between left and right object
                distance = round(distance_between_objects(objects[i], objects[i+1], 70, 640), 3)
                if distance > l:
                    textcolor =  (0, 255, 0)
                else:
                    textcolor =  (0, 0, 255)

                cv2.putText(color, 'distance between objects = ' + str(distance) + 'm', (320, i*60 + 70) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1)

    except:
        cv2.putText(color, 'no objects detected between ' + str(threshold1) + 'm and '+ str(threshold2) + 'm', (30, 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite('obj_SURF.png',color)

    ## stop device and service
    dev.stop()
    serv.stop()

if __name__ == "__main__":
    main()

print(time.time() - start_time,  "sec")
