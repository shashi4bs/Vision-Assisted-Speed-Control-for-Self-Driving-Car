import cv2
from monodepth import get_Depth_Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

class DigitalFilter:
    def __init__(self, vector, b, a, intensity=None):
        self.len = len(vector)
        self.b = b.reshape(-1, 1)
        self.a = a.reshape(-1, 1)
        self.input_history = np.tile(vector.astype(np.float64), (len(self.b), 1))
        self.output_history = np.tile(vector.astype(np.float64), (len(self.a), 1))
        self.old_output = np.copy(self.output_history[0])

    def output(self):
        return self.output_history[0]

    def speed(self):
        return self.output_history[0] - self.output_history[1]

    def new_point(self, vector):
        self.input_history = np.roll(self.input_history, 1, axis=0)
        self.old_output = np.copy(self.output_history[0])
        self.output_history = np.roll(self.output_history, 1, axis=0)
        self.input_history[0] = vector
        self.output_history[0] = (np.matmul(self.b.T, self.input_history) - np.matmul(self.a[1:].T, self.output_history[1:]))/self.a[0]
        return self.output()

    def skip_one(self):
        self.new_point(self.output())


def area(bbox):
    return float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))


class Car:
    def __init__(self, bounding_box, first=False, warped_size=None, transform_matrix=None, pix_per_meter=None, label=None):
        self.warped_size = warped_size
        self.transform_matrix = transform_matrix
        self.pix_per_meter = pix_per_meter
        self.has_position = self.warped_size is not None \
                            and self.transform_matrix is not None \
                            and self.pix_per_meter is not None

        self.bbox = bounding_box
        self.filtered_bbox = DigitalFilter(bounding_box, 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.position = DigitalFilter(self.calculate_position(bounding_box), 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.found = True
        self.num_lost = 0
        self.num_found = 0
        self.display = first
        self.fps = 25
        self.label=label

    def calculate_position(self, bbox):
        if (self.has_position):
            pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
            dst = cv2.perspectiveTransform(pos, self.transform_matrix).reshape(-1, 1)
            #print('  ',(self.warped_size[1]-dst[1]))
            return np.array((self.warped_size[1]-dst[1])/self.pix_per_meter[1])
        else:
            return np.array([0])

    def get_window(self):
        return self.filtered_bbox.output()

    def one_found(self):
        self.num_lost = 0
        if not self.display:
            self.num_found += 1
            if self.num_found > 5:
                self.display = True

    def one_lost(self):
        self.num_found = 0
        self.num_lost += 1
        if self.num_lost > 2:
            self.found = False
        
    def update_car(self, bboxes):
        current_window = self.filtered_bbox.output()
        intersection = np.zeros(4, dtype = np.float32)
        for idx, bbox in enumerate(bboxes):
            intersection[0:2] = np.maximum(current_window[0:2], bbox[0:2])
            intersection[2:4] = np.minimum(current_window[2:4], bbox[2:4])
            if (area(bbox)>0) and area(current_window) and ((area(intersection)/area(current_window)>0.7) or (area(intersection)/area(bbox)>0.7)):
                self.one_found()
                self.filtered_bbox.new_point(bbox)
                self.position.new_point(self.calculate_position(bbox))
                bboxes.pop(idx)
                return

        self.one_lost()
        self.filtered_bbox.skip_one()
        self.position.skip_one()

    def get_details(self):
        window = self.bbox
        pos = self.position.output()[0]
        speed = self.position.speed()[0]*self.fps*3.6
        label = self.label
        return (window, pos, speed,label)
        
    def draw(self, img, color=(255, 0, 0), thickness=2):
        if self.display:
            window = self.filtered_bbox.output().astype(np.int32)
            cv2.rectangle(img, (window[0], window[1]), (window[2], window[3]), color, thickness)
            if self.has_position:
                cv2.putText(img, "RPos: {:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[1]-5)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))
                cv2.putText(img, "RPos: {:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[1]-5)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))
                cv2.putText(img, "RVel: {:6.2f}km/h".format(self.position.speed()[0]*self.fps*3.6), (int(window[0]), int(window[3]+20)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))
                cv2.putText(img, "RVel: {:6.2f}km/h".format(self.position.speed()[0]*self.fps*3.6), (int(window[0]), int(window[3]+20)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))

class CarFinder:
    def __init__(self, size, hist_bins, small_size, orientations=12, pix_per_cell=8, cell_per_block=2,
                 hist_range=None, scaler=None, window_sizes=None, window_rois=None,
                 warped_size=None, transform_matrix=None, pix_per_meter=None):
        self.size = size
        self.small_size = small_size
        self.hist_bins = hist_bins
        self.hist_range = (0, 256)
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.orientations = orientations
        self.scaler = scaler
        self.num_cells = self.size//self.pix_per_cell
        self.num_blocks = self.num_cells - (self.cell_per_block-1)
        if hist_range is not None:
            self.hist_range = hist_range

        self.window_sizes = window_sizes
        self.window_rois = window_rois
        self.cars = []
        self.first = True
        self.warped_size = warped_size
        self.transformation_matrix = transform_matrix
        self.pix_per_meter = pix_per_meter
        self.labels = ['car', 'truck', 'bicycle', 'person']

    def car_find_roi(self, img):
        result = get_Result(img)
        box = []
        label = []
        for r in result:
            if r['label'] in self.labels:
                box.append([r['topleft']['x'], r['topleft']['y'],r['bottomright']['x'], r['bottomright']['y']])
                label.append(r['label'])
        return box, label

    def find_cars(self, img, boxes, labels):
        car_windows = []
        #car_windows, label = self.car_find_roi(img)
        #print('CarWindow', car_windows)
       
        bboxes_temp = []
        for w in car_windows:
            bboxes_temp.append(np.array(w))
        '''
        #additional code for box processing----
        bboxes = []
        for bbox in bboxes_temp:
            car_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2], :]
            he = bbox[3]-bbox[1]
            medi = np.median(car_img[-he//8:-1], axis=[0,1])
            print(medi)
            near = cv2.inRange(car_img, medi - np.array([35, 35, 35]),medi+np.array([35, 35, 35]))
            if near is not None:
                cc = np.sum(near, axis=1)/255 > int(0.8*near.shape[1])
                eee = len(cc)-1
                while eee >= 0 and cc[eee]:
                   eee -= 1
                bbox[3] = bbox[1]+eee
            bboxes.append(bbox)
        print(bboxes)
        '''
        #bboxes = bboxes_temp
        bboxes = boxes
        for car in self.cars:
            car.update_car(bboxes)
        for bbox,l in zip(bboxes,labels):
            self.cars.append(Car(bbox, self.first, self.warped_size, self.transformation_matrix, self.pix_per_meter, label=l))

        tmp_cars = []
        for car in self.cars:
            if car.found:
                tmp_cars.append(car)
        self.cars = tmp_cars
        self.first = False
    
    def get_details(self):
        res = []
        for car in self.cars:
            res.append(car.get_details())
        return res 

    def draw_cars(self, img):
        i2 = np.copy(img)
        for car in self.cars:
            car.draw(i2)
        return i2


def get_pix(image, result, draw_Box=False):
    intensity_map = {}
    for r in result:
        t1 = (r['topleft']['x'],r['topleft']['y'])
        br = (r['bottomright']['x'],r['bottomright']['y'])
        label = r['label']
        img = cv2.rectangle(image, t1, br,(0,255,0),5)
        img = cv2.putText(image, label, t1, cv2.FONT_ITALIC, 0.8, (255,255,255), 2)
    return image, intensity_map

def get_pixel_value(car_img):
    i, j, k = car_img.shape
    img = np.array(car_img)
    try:
        print('pixel_mean',np.mean(img[:,0,0]))
        print('pixel_max', max(img[:,0,0]))
        return max(img[:,0,0])
    except:
        return 0
    
def get_intensity(img, details):
    depth_image = get_Depth_Image(img)
    intensity = []
    for d in details:
        window, position, rspeed, label = d
        car_img = img[window[1]:window[3], window[0]:window[2],:]
        intensity.append(get_pixel_value(car_img))    
        print('window: ',window,'position:',position,'rspeed:',rspeed)  
    return intensity    
        
def get_signal(details, intensity):
    for d, i in zip(details, intensity):
        print(d,i)
        if i>=230 and d[2]>=60:
            return 1
        elif i>=190 and d[2]>=60:
            return 0
    return -1
