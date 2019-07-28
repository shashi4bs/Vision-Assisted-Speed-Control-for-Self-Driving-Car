import cv2
from .darkflow.net.build import TFNet
import matplotlib.pyplot as plt

import tensorflow as tf
import os

cdir = os.getcwd()
cfgdir = os.path.join(cdir,'darkflow/cfg')
bindir = os.path.join(cdir,'darkflow/bin')
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
olddir = os.getcwd()
with tf.Session(config=config) as sess:
    options = {
        'model': os.path.join(cfgdir,'yolo-tiny.cfg'),
        'load': os.path.join(bindir,'yolo-tiny.weights'),
        'threshold': 0.3,
        'gpu': 1
    }
    os.chdir(os.path.join(cdir,'darkflow'))
    tfnet = TFNet(options)
    os.chdir(olddir)
#img = cv2.imread('./traffic.jpeg', cv2.IMREAD_COLOR)


def read_VideoCapture(path):
    return cv2.VideoCapture(path)
    


def get_Result(img, show_op=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    result = tfnet.return_predict(img)
    #print(result)
    if(show_op):
        for r in result:
            t1 = (r['topleft']['x'],r['topleft']['y'])
            br = (r['bottomright']['x'],r['bottomright']['y'])
            label = r['label']
            img = cv2.rectangle(img, t1, br,(0,255,0),5)
            img = cv2.putText(img, label, t1, cv2.FONT_ITALIC, 0.8, (255,255,255), 2)

        cv2.imshow('img',img)
    return result

