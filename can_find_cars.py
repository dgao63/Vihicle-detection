import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import math


class Vehicle():
    def __init__(self):
        # number of images to keep tracing
        self.n = 2
        self.new_detected = False
        self.locked = False
        self.merged = False
        self.lost = False
        self.confidence = None
        self.color_pool = []
        self.color_pool.append((255,0,0))
        self.color_pool.append((0,0,255))
        self.color_pool.append((0,255,0))
        self.color_pool.append((0,255,255))
        self.color_pool.append((0,100,255))
        self.color_pool.append((0,255,100))
        self.color = self.color_pool[np.random.randint(6)]
        # was the line detected in the last iteration?
        self.center_x = None
        self.recent_center_x = []
        self.mean_center_x = None
        self.std_center_x = None
        self.center_y = None
        self.recent_center_y = []
        self.mean_center_y = None
        self.std_center_y = None
        self.recent_window_length = [] 
        self.recent_window_height = []
        self.mean_window_length = None
        self.std_window_length = None
        self.draw_window_length = None
        self.mean_window_height = None
        self.std_window_height = None
        self.draw_window_height = None

    def adjust_length(self, win_0, win_1):
        x_min = win_0[0]
        x_max = win_1[0]
        y_min = win_0[1]
        y_max = win_1[1]
        length = x_max - x_min
        if len(self.recent_window_length) < self.n:
            self.recent_window_length.append(length)
            return max(self.recent_window_length)
        self.mean_window_length = np.mean(self.recent_window_length)
        self.std_window_length = np.std(self.recent_window_length)
        self.draw_window_length = self.mean_window_length
        #print("mean:", self.mean_window_length)
        #print("std:", self.std_window_length)
        threshold = max(self.std_window_length, 60)
        if abs(length - self.mean_window_length) < threshold:
            self.recent_window_length.append(length)
            if len(self.recent_window_length) > self.n:
                del(self.recent_window_length[0])
            self.mean_window_length = np.mean(self.recent_window_length)
            return self.mean_window_length
        else:
            return self.mean_window_length

    def adjust_height(self, win_0, win_1):
        x_min = win_0[0]
        x_max = win_1[0]
        y_min = win_0[1]
        y_max = win_1[1]
        height = y_max - y_min
        if len(self.recent_window_height) < self.n:
            self.recent_window_height.append(height)
            return max(self.recent_window_height)
        self.mean_window_height = np.mean(self.recent_window_height)
        self.std_window_height = np.std(self.recent_window_height)
        self.draw_window_height = self.mean_window_height
        threshold = max(self.std_window_height, 50)
        if abs(height - self.mean_window_height) < threshold:
            self.recent_window_height.append(height)
            if len(self.recent_window_height) > self.n:
                del(self.recent_window_height[0])
            self.mean_window_height = np.mean(self.recent_window_height)
            return self.mean_window_height
        else:
            return self.mean_window_height

    def adjust_center_x(self, bbox_center_x):
        if len(self.recent_center_x) < self.n:
            self.recent_center_x.append(bbox_center_x)
            return bbox_center_x
        self.mean_center_x = np.mean(self.recent_center_x)
        self.std_center_x = np.std(self.recent_center_x)
        threshold = max(self.std_center_x, self.mean_window_length*0.1)
        print("car center_x threshold:", threshold)
        if abs(bbox_center_x - self.mean_center_x) < threshold:
            self.recent_center_x.append(bbox_center_x)
            if len(self.recent_center_x) > self.n:
                del(self.recent_center_x[0])
            self.mean_center_x = np.mean(self.recent_center_x)
            return bbox_center_x
        else:
            return self.recent_center_x[-1]

    def adjust_center_y(self, bbox_center_y):
        if len(self.recent_center_y) < self.n:
            self.recent_center_y.append(bbox_center_y)
            return bbox_center_y
        self.mean_center_y = np.mean(self.recent_center_y)
        self.std_center_y = np.std(self.recent_center_y)
        #threshold = max(self.std_center_y, self.mean_window_height*0.1)
        threshold = max(self.std_center_y, 8)
        if abs(bbox_center_y - self.mean_center_y) < threshold:
            self.recent_center_y.append(bbox_center_y)
            if len(self.recent_center_y) > self.n:
                del(self.recent_center_y[0])
            self.mean_center_y = np.mean(self.recent_center_y)
            return bbox_center_y
        else:
            return self.recent_center_y[-1]


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        image_features = []
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            image_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            image_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            image_features.append(hog_features)
        features.append(np.concatenate(image_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5), scale=(1,1)):
    window_list = []
    scale_list = np.linspace(scale[0], scale[1], num=4)
    for scale in scale_list:
        x = xy_window[0]/scale
        y = xy_window[1]/scale
        xy_window = (x, y)
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((int(startx), int(starty)), (int(endx), int(endy))))
        # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def single_img_features(img, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #print(np.max(spatial_features))
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #print(np.max(hist_features))
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #print(np.max(hog_features))
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel='ALL', spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        print(np.max(features))
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            #cv2.rectangle(img,tuple(window[0]),tuple(window[1]),(0,0,255),2) 
            #plt.imshow(img)
            #plt.show()
    #8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap



def CarPerspectiveMatrix():
    src_p1 = (253,697) #left down
    src_p2 = (610,435) #left up
    src_p3 = (670,431) #right up
    src_p4 = (1061,690) #right down
    src = np.float32([src_p1,src_p2,src_p3,src_p4])
    dst_p1 = (500,697)
    dst_p2 = (500,0)
    dst_p3 = (750,0)
    dst_p4 = (750,690)
    dst = np.float32([dst_p1,dst_p2,dst_p3,dst_p4])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def find_root_dist_orient(root):
    print(root)
    M_car, Minv_car = CarPerspectiveMatrix()
    ym = 3.0/52 # meters per pixel in y dimension
    xm = 3.7/250 # meters per pixel in x dimension
    root_center = cv2.perspectiveTransform(np.array([np.array([[700,720]], dtype=np.float32)]), M_car)
    root =        cv2.perspectiveTransform(np.array([np.array([list(root)], dtype=np.float32)]), M_car)
    print(root)
    x0 = root_center[0][0][0]
    y0 = root_center[0][0][1]
    x1 = root[0][0][0]
    y1 = root[0][0][1]
    x_dis = abs(x1-x0)*xm
    y_dis = abs(y1-y0)*ym
    orientation = math.atan2(y_dis, x_dis)/math.pi*180
    distance = math.sqrt(x_dis*x_dis+y_dis*y_dis)
    return orientation, distance


def initial_draw_box(img, labels, cars, clf):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        #print("car_number:", car_number)
        new_detected_car = True
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bbox_center = ( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2 )
        print("bbox center:", bbox_center)
        if len(cars) == 0:
            new_car = Vehicle()
            new_car.new_detected = True
            new_car.center_x = bbox_center[0]
            new_car.center_y = bbox_center[1]
            new_car.recent_window_length.append(bbox[1][0]-bbox[0][0])
            new_car.recent_window_height.append(bbox[1][1]-bbox[0][1])
            cars.append(new_car)
            print("new car detected")
            cv2.rectangle(img, bbox[0], bbox[1], new_car.color, 4)
            orientation, distance = find_root_dist_orient(bbox[1])
            car_img = cv2.resize(img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0],:], (64,64))
            car_img_feature = single_img_features(car_img)
            proba = clf.predict_proba(car_img_feature)
            car_pos = len(cars)
            cv2.putText(img, "car_" + "{:.0f}".format(car_pos)+"  conf: "+"{:.2f}".format(np.max(proba[0])), (bbox[0][0],bbox[0][1]-28), font, 0.8, new_car.color, 2)
            cv2.putText(img, "dist: "+"{:.1f}".format(distance)+"m "+"orient: "+"{:.0f}".format(orientation)+"deg", (bbox[0][0],bbox[0][1]-6), font, 0.8, new_car.color, 2)
            continue
        for car in cars:
            print("car_list center:", [car.center_x, car.center_y])
            if abs(bbox_center[0] - car.center_x) < 150 and abs(bbox_center[1] - car.center_y) < 100:
                new_detected_car = False
        if new_detected_car == True:
            new_car = Vehicle()
            new_car.new_detected = True
            new_car.center_x = bbox_center[0]
            new_car.center_y = bbox_center[1]
            new_car.recent_window_length.append(bbox[1][0]-bbox[0][0])
            new_car.recent_window_height.append(bbox[1][1]-bbox[0][1])
            cars.append(new_car)
            print("new car detected")
            cv2.rectangle(img, bbox[0], bbox[1], new_car.color, 4)
            orientation, distance = find_root_dist_orient(bbox[1])
            car_img = cv2.resize(img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0],:], (64,64))
            car_img_feature = single_img_features(car_img)
            proba = clf.predict_proba(car_img_feature)
            car_pos = len(cars)
            cv2.putText(img, "car_" + "{:.0f}".format(car_pos)+"  conf: "+"{:.2f}".format(np.max(proba[0])), (bbox[0][0],bbox[0][1]-28), font, 0.8, new_car.color, 2)
            cv2.putText(img, "dist: "+"{:.1f}".format(distance)+"m "+"orient: "+"{:.0f}".format(orientation)+"deg", (bbox[0][0],bbox[0][1]-6), font, 0.8, new_car.color, 2)
            continue
 
    # Return the image
    return img, cars, bbox[1]

def car_draw_box(img, labels, cars, pos, confidence, clf):
    car = cars[pos]
    font = cv2.FONT_HERSHEY_SIMPLEX
    dist_list = []
    bbox_list = []
    bbox_center_list = []
    if labels[1] == 0:
        cars[pos].lost = True
        '''
        cv2.rectangle(img, (np.int(car.center_x-car.recent_window_length[-1]*0.5), np.int(car.center_y-car.recent_window_height[-1]*0.5)), 
            (np.int(car.center_x+car.recent_window_length[-1]*0.5), np.int(car.center_y+car.recent_window_height[-1]*0.5)), car.color, 4)
        car_pos = pos + 1
        cv2.putText(img, "car_" + "{:.0f}".format(car_pos), (np.int(car.center_x-car.recent_window_length[-1]*0.5), np.int(car.center_y-car.recent_window_height[-1]*0.5)-28), font, 0.8, car.color, 2)
        cv2.putText(img, "conf:" + "{:.2f}".format(confidence), (np.int(car.center_x-car.recent_window_length[-1]*0.5), np.int(car.center_y-car.recent_window_height[-1]*0.5)-6), font, 0.8, car.color, 2)
        '''
        return img, cars, (None, None)

    print("number of labels[1]:", labels[1])
    split = 0
    for car_number in range(1, labels[1]+1):
        #print("car_number:", car_number)
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bbox_center = ( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2 )
        print("bbox center:", bbox_center)
        print("car center:", [car.center_x, car.center_y])
        if bbox_center[0] > car.center_x - (car.recent_window_length[-1]*0.5) and bbox_center[0] < car.center_x + (car.recent_window_length[-1]*0.5):
            split = split + 1
    car_split = False
    if split > 1:
        car_split = True
    if car_split == True:
        cars[pos].lost = True
        for car_number in range(1, labels[1]+1): #should skip the vehicle that has been boxed at the begining of this function, might do it later
            #print("car_number:", car_number)
            #new_detected_car = True
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            bbox_center = ( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2 )
            print("bbox center:", bbox_center)
            new_car = Vehicle()
            #new_car.new_detected = True ##inside individual draw, do not set
            new_car.center_x = bbox_center[0]
            new_car.center_y = bbox_center[1]
            new_car.recent_window_length.append(bbox[1][0]-bbox[0][0])
            new_car.recent_window_height.append(bbox[1][1]-bbox[0][1])
            cars.append(new_car)
            print("merged car split")
            cv2.rectangle(img, bbox[0], bbox[1], new_car.color, 4)
            orientation, distance = find_root_dist_orient(bbox[1])
            car_img = cv2.resize(img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0],:], (64,64))
            car_img_feature = single_img_features(car_img)
            proba = clf.predict_proba(car_img_feature)
            car_pos = len(cars)
            print("draw split car num:", car_number-1)
            cv2.putText(img, "car_" + "{:.0f}".format(car_pos)+"  conf: "+"{:.2f}".format(np.max(proba[0])), (bbox[0][0],bbox[0][1]-28), font, 0.8, new_car.color, 2)
            cv2.putText(img, "dist: "+"{:.1f}".format(distance)+"m "+"orient: "+"{:.0f}".format(orientation)+"deg", (bbox[0][0],bbox[0][1]-6), font, 0.8, new_car.color, 2)
        return img, cars, bbox[1]

    nonzero = (labels[0] == 1).nonzero()
    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    # Draw the box on the image
    bbox_center = ( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2 )    
    #cars[pos].center_x = car.adjust_center_x(bbox_center[0])
    cars[pos].center_y = car.adjust_center_y(bbox_center[1])
    cars[pos].center_x = bbox_center[0]
    #cars[pos].center_y = bbox_center[1]
    cars[pos].new_detected = False
    cars[pos].locked = True
    draw_length = cars[pos].adjust_length(bbox[0], bbox[1])
    draw_height = cars[pos].adjust_height(bbox[0], bbox[1])
    print("center_x of the box:", cars[pos].center_x)
    print("length to draw:", draw_length)
    print("locked car window:", bbox)
    locked_win_0 = (np.int(cars[pos].center_x-draw_length*0.5), np.int(cars[pos].center_y-draw_height*0.5))
    locked_win_1 = (np.int(cars[pos].center_x+draw_length*0.5), np.int(cars[pos].center_y+draw_height*0.5))
    print("adjusted locked car window:", [locked_win_0, locked_win_1])
    cv2.rectangle(img, locked_win_0, locked_win_1, cars[pos].color, 4)
    orientation, distance = find_root_dist_orient(locked_win_1)
    car_pos = pos + 1
    if cars[pos].merged == True:
        cv2.putText(img, "merged car_" + "{:.0f}".format(car_pos)+"  conf: "+"{:.2f}".format(confidence), (locked_win_0[0],locked_win_0[1]-28), font, 0.8, cars[pos].color, 2)
        cv2.putText(img, "dist: "+"{:.1f}".format(distance)+"m "+"orient: "+"{:.0f}".format(orientation)+"deg", (locked_win_0[0],locked_win_0[1]-6), font, 0.8, cars[pos].color, 2)
    else:
        cv2.putText(img, "car_" + "{:.0f}".format(car_pos)+"  conf: "+"{:.2f}".format(confidence), (locked_win_0[0],locked_win_0[1]-28), font, 0.8, cars[pos].color, 2)
        cv2.putText(img, "dist: "+"{:.1f}".format(distance)+"m "+"orient: "+"{:.0f}".format(orientation)+"deg", (locked_win_0[0],locked_win_0[1]-6), font, 0.8, cars[pos].color, 2)
    # Return the image
    return img, cars, locked_win_1

def new_draw_box(img, labels, cars, clf):
    print("find new possible cars")
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        #print("car_number:", car_number)
        new_detected_car = True
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bbox_center = ( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2 )
        print("bbox center:", bbox_center)
        for car in cars:
            print("car_list center:", [car.center_x, car.center_y])
            if abs(bbox_center[0] - car.center_x) < 150 and abs(bbox_center[1] - car.center_y) < 100:
                new_detected_car = False
        if new_detected_car == True:
            new_car = Vehicle()
            new_car.new_detected = True
            new_car.center_x = bbox_center[0]
            new_car.center_y = bbox_center[1]
            new_car.recent_window_length.append(bbox[1][0]-bbox[0][0])
            new_car.recent_window_height.append(bbox[1][1]-bbox[0][1])
            new_car.new = True
            cars.append(new_car)
            print("new car detected")
            cv2.rectangle(img, bbox[0], bbox[1], new_car.color, 4)
            orientation, distance = find_root_dist_orient(bbox[1])
            car_img = cv2.resize(img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0],:], (64,64))
            car_img_feature = single_img_features(car_img)
            proba = clf.predict_proba(car_img_feature)
            car_pos = len(cars)
            cv2.putText(img, "car_" + "{:.0f}".format(car_pos)+"  conf: "+"{:.2f}".format(np.max(proba[0])), (bbox[0][0],bbox[0][1]-28), font, 0.8, new_car.color, 2)
            cv2.putText(img, "dist: "+"{:.1f}".format(distance)+"m "+"orient: "+"{:.0f}".format(orientation)+"deg", (bbox[0][0],bbox[0][1]-6), font, 0.8, new_car.color, 2)
            continue

    split_list = np.zeros(len(cars))
    for car_number in range(1, labels[1]+1):
        #print("car_number:", car_number)
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bbox_center = ( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2 )
        print("bbox center:", bbox_center)
        for i in range(len(cars)):
            car = cars[i]
            print("car_list center:", [car.center_x, car.center_y])
            if bbox_center[0] > car.center_x - car.recent_window_length[-1]*0.5 and bbox_center[0] < car.center_x + car.recent_window_length[-1]*0.5:
                split_list[i] = split_list[i] + 1
    car_split = False
    for j in range(len(split_list)):
        if split_list[j] > 1:
            car_split = True
            split_pos = j #if deal with multi-cars, should do a list, maybe later
    if car_split == True:
        cars[split_pos].lost = True
        for car_number in range(1, labels[1]+1): #should skip the vehicle that has been boxed at the begining of this function, might do it later
            #print("car_number:", car_number)
            #new_detected_car = True
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            bbox_center = ( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2 )
            print("bbox center:", bbox_center)
            new_car = Vehicle()
            new_car.new_detected = True
            new_car.center_x = bbox_center[0]
            new_car.center_y = bbox_center[1]
            new_car.recent_window_length.append(bbox[1][0]-bbox[0][0])
            new_car.recent_window_height.append(bbox[1][1]-bbox[0][1])
            new_car.new = True
            cars.append(new_car)
            print("merged car split")
            cv2.rectangle(img, bbox[0], bbox[1], new_car.color, 4)
            orientation, distance = find_root_dist_orient(bbox[1])
            car_img = cv2.resize(img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0],:], (64,64))
            car_img_feature = single_img_features(car_img)
            proba = clf.predict_proba(car_img_feature)
            car_pos = len(cars)
            ccv2.putText(img, "car_" + "{:.0f}".format(car_pos)+"  conf: "+"{:.2f}".format(np.max(proba[0])), (bbox[0][0],bbox[0][1]-28), font, 0.8, new_car.color, 2)
            cv2.putText(img, "dist: "+"{:.1f}".format(distance)+"m "+"orient: "+"{:.0f}".format(orientation)+"deg", (bbox[0][0],bbox[0][1]-6), font, 0.8, new_car.color, 2)

    # Return the image
    return img, cars, bbox[1]






