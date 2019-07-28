import math
import pickle
from darkflow.yolo_res import *
from monodepth import *
import os
import matplotlib.pyplot as plt
from helper import *
import numpy as np
from constants import PERSPECTIVE_FILE_NAME, UNWARPED_SIZE, ORIGINAL_SIZE
import argparse
import logging



parser = argparse.ArgumentParser(description='VA-ABS setup options')

parser.add_argument('--input', type=str,help='input file for video', default='./input_video/project_video_1.mp4')
parser.add_argument('--write', type=bool,help='output_video', default=False)
args = parser.parse_args()

logging.basicConfig(filename="log.log", format='%(asctime)s %(message)s', filemode='w')
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 

os.system('clear')

#load perspective data
with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
    perspective_data = pickle.load(f)

perspective_transform = perspective_data["perspective_transform"]
pixels_per_meter = perspective_data['pixels_per_meter']
orig_points = perspective_data["orig_points"]

def process_image(img, boxes, car_finder, labels):
    car_finder.find_cars(img, boxes, labels)
    return cf, car_finder.get_details()

#create carFinder Object
cf = CarFinder(64, hist_bins=128, small_size=20, orientations=12, pix_per_cell=8, cell_per_block=1,
                               transform_matrix=perspective_transform, warped_size=UNWARPED_SIZE,
                               pix_per_meter=pixels_per_meter)
                               


cap = read_VideoCapture(args.input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
if args.write:
    i = 0
    while(True):
        if(os.path.exists('./output_videos/output_'+str(i)+'.mp4')):
            i += 1
        else:
            break
    out = cv2.VideoWriter('./output_videos/output_'+str(i)+'.mp4',  cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while(cap.isOpened()):
    ret, img = cap.read()
    result = get_Result(img)
    boxes = []
    labels = []
    for r in result:
        print(r)
        boxes.append(np.array([r['topleft']['x'],r['topleft']['y'],r['bottomright']['x'],r['bottomright']['y']]))
        labels.append(r['label'])
        
    cf, details = process_image(img, boxes, cf, labels)
    intensity = get_intensity(img,details)
    #print(details)
    braking_signal = get_signal(details, intensity)
    msg = " "
    print('Braking Signal: ',braking_signal)
    if braking_signal == 0:
        msg = "Slow Down"
    if braking_signal == 1:
        msg = "ABS Activated"
    print(msg)
    im_final = cf.draw_cars(img)
    cv2.putText(im_final,"{}".format(msg),(50,50),cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 1, 1))
    cv2.imshow('final output: ', im_final)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    if args.write:
        out.write(im_final)
