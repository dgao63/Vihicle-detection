import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from can_find_cars import *
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
import pickle
print("modules loaded")

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Read in cars and notcars
car_images = glob.glob('split_data/split_data/vehicles/train/vehicle/*.png')
notcar_images = glob.glob('split_data/split_data/non-vehicles/train/*.png')
X_train_img = []
for fname in car_images:
        image = plt.imread(fname)
        X_train_img.append(image)
for fname in notcar_images:
        image = plt.imread(fname)
        X_train_img.append(image)

X_train = extract_features(X_train_img, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

num_car = len(car_images)
num_notcar = len(notcar_images)
y_train = np.hstack((np.ones(num_car), np.zeros(num_notcar)))
print("X_train size:", len(X_train))
print("y_train size:", len(y_train))

#####################

car_images = glob.glob('split_data/split_data/vehicles/validation/vehicle/*.png')
notcar_images = glob.glob('split_data/split_data/non-vehicles/validation/*.png')
X_vali_img = []
for fname in car_images:
        image = plt.imread(fname)
        X_vali_img.append(image)
for fname in notcar_images:
        image = plt.imread(fname)
        X_vali_img.append(image)

X_validation = extract_features(X_vali_img, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

num_car = len(car_images)
num_notcar = len(notcar_images)
y_validation = np.hstack((np.ones(num_car), np.zeros(num_notcar)))
print("X_validation size:", len(X_validation))
print("y_validation size:", len(y_validation))

#######################

car_images = glob.glob('split_data/split_data/vehicles/test/vehicle/*.png')
notcar_images = glob.glob('split_data/split_data/non-vehicles/test/*.png')
X_test_img = []
for fname in car_images:
        image = plt.imread(fname)
        X_test_img.append(image)
for fname in notcar_images:
        image = plt.imread(fname)
        X_test_img.append(image)

X_test = extract_features(X_test_img, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

num_car = len(car_images)
num_notcar = len(notcar_images)
y_test = np.hstack((np.ones(num_car), np.zeros(num_notcar)))
print("X_test size:", len(X_test))
print("y_test size:", len(y_test))



data_todump = {"X_train": X_train, "y_train": y_train, "X_validation":X_validation, "y_validation":y_validation, "X_test":X_test, "y_test":y_test}
pickle.dump(data_todump, open("split_data.p", 'wb'))
print("done")








