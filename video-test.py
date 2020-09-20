from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None



def YOLO():

    global metaMain, netMain, altNames
    configPath = "./yolov3_gs.cfg"
    weightPath = "./yolov3_gs_last.weights"
    metaPath = "./bisindo/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    usewebcam = False
    skipframe = 1
    efpies = 8 #for default
    if(usewebcam):
        #cap = cv2.VideoCapture(-1)  #webcam
        cap = cv2.VideoCapture("http://192.168.100.25:8080/video") #android webcam
        vidend = -1 #no end
    else:
        vidname = "wax"
        cap = cv2.VideoCapture("tests/input-framed/framed-" + vidname + ".avi")
        vidend = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        efpies = cap.get(cv2.CAP_PROP_FPS)
        print("No. of Frame : " + str(vidend))
        print("FPS : " + str(efpies))
    cap.set(3, 640)
    cap.set(4, 480)
    out = cv2.VideoWriter(
        "fout-" + vidname + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), efpies,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),1)
    fcount = 0
    streak = 0
    streak_limit = 7
    lastchar = ''
    last_streak_char = ''
    detected_char = []
    while(cap.isOpened()):
        ret, frame_read = cap.read()
        if(fcount == vidend):
            print("video ends")
            break
        if(ret and fcount%skipframe == 0):
            prev_time = time.time()
            #frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_read,#frame_rgb to frame_read
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            darknet.copy_image_from_bytes(darknet_image,frame_gray.tobytes()) #frame_resized to frame_gray

            #detections = []
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.15)
            if(len(detections) > 0):
                char = str(detections[0][0].decode())
                #print("detect " + char)
                print(char)
            else:
                char = ''
                #print("no detection")
                print('0')
            image = cvDrawBoxes(detections, frame_resized)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if(lastchar == char):
                #print("same")
                streak += 1
                if(streak == streak_limit and char != last_streak_char):
                    last_streak_char = char
                    if(char != ''):
                        #print("char added")
                        detected_char.append(char)
            else:
                streak = 1
            lastchar = char
            out.write(image)
            cv2.imshow('Demo', image)
            print((time.time()-prev_time))
            #input("frame " + str(fcount+1) + ": Press Enter to continue...")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("video ends")
                break
        fcount += 1
    cap.release()
    out.release()

    print(detected_char)

if __name__ == "__main__":
    YOLO()
