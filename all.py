# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from can_see_line import *
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from can_find_cars import *
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
import pickle
print("modules loaded!")

mtx, dist = calibration(nx=9, ny=6)
M, Minv = PerspectiveMatrix()
M_car, Minv_car = CarPerspectiveMatrix()
image_size = (1280, 720)
image_count = 0


left_line = Line()
right_line = Line()

left_arrow = plt.imread("./left_arrow.jpg")
right_arrow = plt.imread("./right_arrow.jpg")
straight_arrow = plt.imread("./straight.jpg")

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
y_start_stop = [300, None] # Min and max in y to search in slide_window()
xy_overlap=(0.75, 0.75)
font = cv2.FONT_HERSHEY_SIMPLEX
threshold_new = 4
threshold_locked = 4
threshold_merged = 4
ystart = 400
ystop = 720
image_count = 0
need_test = 500000
need_plot = 7002626
start_num = 700
plot_period = 200000
scale_img = [1.2, 1.6, 1.8, 2.5]
scale_car = [1.2, 1.4, 2.5]
car_list = []



model_scaler = pickle.load( open( "model_scaler.p", "rb" ))
X_scaler = model_scaler["scaler"]
clf = model_scaler["clf"]


def find_cars(img, ystart, ystop, scale_img, scale_car, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    global car_list
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255.0
    on_windows = []
    car_windows = []
    car_confidence = []
    xstart = 600
    for i in range(3):
        car_windows.append([])
    
    if (image_count > start_num and image_count % 50 == 0) or image_count == start_num: 
        #print("")
        print("scan whole image")
        for scale in scale_img:
            if scale == 1.2:
                ystop = 430
            if scale == 1.6 or scale == 1.4:
                ystop = 560
            if scale == 1.8:
                ystop = 720
            if scale == 2.5 :
                ystop = 720
            print("scale:", scale)
            img_tosearch = img[ystart:ystop,xstart:,:]
            ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
                
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // pix_per_cell)-1
            nyblocks = (ch1.shape[0] // pix_per_cell)-1 
            nfeat_per_block = orient*cell_per_block**2
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // pix_per_cell)-1 
            cells_per_step = 1 # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step
            
            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

            
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    #print(np.max(hog_features))
                    assert math.isnan(np.max(hog_features)) is False, "hog_features has nan"

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                  
                    # Get color features
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    #print(np.max(spatial_features))
                    assert math.isnan(np.max(spatial_features)) is False, "spatial_features has nan"
                    hist_features = color_hist(subimg, nbins=hist_bins)
                    #print(np.max(hist_features))
                    assert math.isnan(np.max(hist_features)) is False, "hist_features has nan"

                    # Scale features and make a prediction
                    test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                    #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                    test_prediction = svc.predict(test_features)
                    
                    if test_prediction == 1:
                        proba = svc.predict_proba(test_features)
                        print("condifence:", proba)
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        on_windows.append( ( (xstart+xbox_left, ytop_draw+ystart),(xstart+xbox_left+win_draw,ytop_draw+win_draw+ystart) ) )
                        cv2.rectangle(draw_img,(xstart+xbox_left, ytop_draw+ystart),(xstart+xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                        #if image_count > need_plot:
                        #    plt.imshow(draw_img)
                        #    plt.show() 

    for i in range(len(car_list)):
        if car_list[i].lost == True:
            continue
        car = car_list[i]
        car_proba = []
        print("total num in carlist:", len(car_list))
        print("position in car list:", i)
        print("scan car surroundings")
        for scale in scale_car:
            print("scale:", scale)
            if len(car.recent_window_length) == 0 or len(car.recent_window_height) == 0:
                x_start = np.int(car.center_x - 200)
                x_stop = np.int(car.center_x + 200)
                y_start = np.int(car.center_y - 60)
                y_stop = np.int(car.center_y + 60)
            else:
                #print("use mean value to scan car area")
                if car.recent_window_length[-1]  > 200: ##### maybe remove the ratio!!!
                    x_range = 200
                elif car.recent_window_length[-1]  < 50:
                    x_range = 50
                else:
                    x_range = car.recent_window_length[-1] 
                x_start = np.int(car.center_x - x_range)
                x_stop = np.int(car.center_x + x_range)
                y_start = np.int(car.center_y - car.recent_window_height[-1])
                y_stop = np.int(car.center_y + car.recent_window_height[-1])
            img_tosearch = img[ystart:ystop,x_start:x_stop,:]
            ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
                
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // pix_per_cell)-1
            nyblocks = (ch1.shape[0] // pix_per_cell)-1 
            nfeat_per_block = orient*cell_per_block**2
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // pix_per_cell)-1 
            cells_per_step = 1  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step
            
            # Compute individual channel HOG features for the entire image
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

            
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    #print(np.max(hog_features))
                    assert math.isnan(np.max(hog_features)) is False, "hog_features has nan"

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                  
                    # Get color features
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    #print(np.max(spatial_features))
                    assert math.isnan(np.max(spatial_features)) is False, "spatial_features has nan"
                    hist_features = color_hist(subimg, nbins=hist_bins)
                    #print(np.max(hist_features))
                    assert math.isnan(np.max(hist_features)) is False, "hist_features has nan"

                    # Scale features and make a prediction
                    test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                    #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                    test_prediction = svc.predict(test_features)
                    
                    if test_prediction == 1:
                        pp = np.max(svc.predict_proba(test_features)[0])
                        if pp > 0.9:
                            car_proba.append(pp)
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        #on_windows.append( ( (xbox_left+x_start, ytop_draw+ystart),(xbox_left+win_draw+x_start,ytop_draw+win_draw+ystart) ) )
                        car_windows[i].append( ( (xbox_left+x_start, ytop_draw+ystart),(xbox_left+win_draw+x_start,ytop_draw+win_draw+ystart) ) )
                        cv2.rectangle(draw_img,(xbox_left+x_start, ytop_draw+ystart),(xbox_left+win_draw+x_start,ytop_draw+win_draw+ystart),(0,0,255),6) 
                        #if image_count > need_plot:
                        #    plt.imshow(draw_img)
                        #    plt.show()
        car_confidence.append(np.mean(car_proba))
    if image_count > need_plot or image_count % plot_period == 0:
        plt.imshow(draw_img)
        plt.show()                    
    return on_windows, car_windows, car_confidence


def process_image(image):
    global car_list
    img = np.copy(image)
    global image_count
    image_count = image_count + 1
    print()
    print(image_count)
    left_line.detected = False
    right_line.detected = False
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    ksize=3

    '''
    if image_count > need_plot:
        plt.imshow(gray, cmap='gray')
        plt.show()
    '''

    sx_binary = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(undist, sobel_kernel=ksize, thresh=(30, 100))
    dir_binary = dir_thresh(undist, sobel_kernel=ksize, thresh=(0.7, np.pi/2))
    s_binary = s_color_thresh(undist, thresh=(120, 255))
    l_binary = l_color_thresh(undist, thresh=(120, 255))


    combined_binary = np.zeros_like(sx_binary)
    combined_binary[ ( (s_binary == 1) & (l_binary == 1) ) | (sx_binary == 1) ] = 1

    '''
    if image_count > need_plot:
        plt.imshow(combined_binary, cmap='gray')
        plt.show()
    '''

    binary_warped = cv2.warpPerspective(combined_binary, M, image_size, flags=cv2.INTER_LINEAR)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    if image_count == 1:
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        leftx_base = left_line.next_x_base
        rightx_base = right_line.next_x_base

    if image_count == need_test:
        print("current leftx_base: ", leftx_base)
        print("current rightx_base: ", rightx_base)

    if image_count == need_test:
        print("leftx_base: ", leftx_base)
        print("rightx_base: ", rightx_base)
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        '''
        if image_count > need_plot:
            print("window number: ", window)
            print("good_left_inds number: ", len(good_left_inds))
            print("good_right_inds number: ", len(good_right_inds))
        '''

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_shift = np.int(np.mean(nonzerox[good_left_inds])) - leftx_current
            if abs(leftx_shift) < 40:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            elif leftx_shift > 40:
                leftx_current = leftx_current + 40
            else:
                leftx_current = leftx_current - 40
        if len(good_right_inds) > minpix:        
            rightx_shift = np.int(np.mean(nonzerox[good_right_inds])) - rightx_current
            if abs(rightx_shift) < 40:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            elif rightx_shift > 40:
                rightx_current = rightx_current + 40
            else:
                rightx_current = rightx_current - 40

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    if (len(left_lane_inds) > 2000) and (len(right_lane_inds) > 2000):
        left_line.detected = True
        right_line.detected = True

    if (left_line.detected == False) or (right_line.detected == False):
        left_fit = left_line.recent_fit_param[-1]
        right_fit = right_line.recent_fit_param[-1]
        left_fitx = left_line.recent_xfitted[-1]
        right_fitx = right_line.recent_xfitted[-1]
    else:
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_fit = left_line.fit_param_adjust(left_fit, image_count)
        right_fit = right_line.fit_param_adjust(right_fit, image_count)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        
        left_fitx = left_line.x_adjust(left_fitx, image_count)
        right_fitx = right_line.x_adjust(right_fitx, image_count)

        left_line.next_x_base = int(np.mean(left_fitx[int(len(left_fitx)*0.6):int(len(left_fitx)*0.8)]))
        right_line.next_x_base = int(np.mean(right_fitx[int(len(right_fitx)*0.6):int(len(right_fitx)*0.8)]))

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        '''
        if image_count > need_plot:
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
        '''

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    left_curverad = left_line.curverad_adjust(left_curverad, image_count)
    right_curverad = right_line.curverad_adjust(right_curverad, image_count)

    left_x_bottem = left_fit[0]*719**2 + left_fit[1]*719 + left_fit[2]
    left_x_top = left_fit[2]
    right_x_bottem = right_fit[0]*719**2 + right_fit[1]*719 + right_fit[2]
    right_x_top = right_fit[2]
    line_mid = (right_x_bottem + left_x_bottem)/2
    offset = xm_per_pix*(line_mid - 660) #camera not in the very middle

    # calculate direction and steering
    if left_x_top > left_x_bottem and right_x_top > right_x_bottem:
        left_line.direction = 1 #right
    elif left_x_top < left_x_bottem and right_x_top < right_x_bottem:
        left_line.direction = -1 #left
    else:
        left_line.direction = 100 #bad 
    if left_curverad > 1500 and right_curverad > 1500:
        left_line.direction = 0

    a = [300, 600, 800, 1000, 2000]
    b = [12, 10, 8, 6, 0]
    fit = np.polyfit(a, b, 2)
    curedad = (left_curverad + right_curverad) * 0.5 
    if curedad > 2000:
        curedad = 2000
    if left_line.direction == 1 or left_line.direction == -1:
        steering = left_line.direction * ( abs(fit[0]*curedad**2+fit[1]*curedad+fit[2]) )
        steering = left_line.steering_adjust(steering, image_count)
    elif left_line.direction == 0:
        steering = 0
        #steering = left_line.steering_adjust(steering, image_count)
    elif left_line.direction == 100:
        steering = 0

    print("steering: ", int(steering), "%")

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 140))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    small_left_arrow = cv2.resize(left_arrow, (60,90))
    small_right_arrow = cv2.resize(right_arrow, (60,90))
    small_straight_arrow = cv2.resize(straight_arrow, (60,90))
    if left_line.direction == -1:
        newwarp[500:500+small_left_arrow.shape[0], 620:620+small_left_arrow.shape[1]] = small_left_arrow
    if left_line.direction == 1:
        newwarp[500:500+small_right_arrow.shape[0], 620:620+small_right_arrow.shape[1]] = small_right_arrow
    if left_line.direction == 0 or left_line.direction == 100:
        newwarp[500:500+small_straight_arrow.shape[0], 620:620+small_straight_arrow.shape[1]] = small_straight_arrow
    #plt.imshow(newwarp)
    #plt.show()
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)


    small_size = (320, 180)
    
    index = combined_binary.nonzero()
    a = combined_binary[:,:,np.newaxis]
    aa = np.dstack((a,a,a))
    aa[index[0], index[1]] = [255, 255, 255]
    small_gray = cv2.resize(aa, small_size)

    small_out_img = cv2.resize(out_img, small_size)
    fig, ax = plt.subplots(1, 1)  # create figure & 1 axis
    ax.plot(histogram)
    fig.savefig('./fig.png')
    plt.close(fig) 
    png = Image.open('./fig.png')
    png.load() 
    background = Image.new("RGB", (800,600), (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    background.save('fig.jpg', 'JPEG', quality=80)
    histo = plt.imread('./fig.jpg')
    histo = histo[64:539,102:670]
    small_histo = cv2.resize(histo, (320, 120))

    #result[50:50+small_gray.shape[0], 50:50+small_gray.shape[1]] = small_gray
    #result[240:240+small_histo.shape[0], 50:50+small_histo.shape[1]] = small_histo
    result[100:100+small_out_img.shape[0], 50:50+small_out_img.shape[1]] = small_out_img
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "lanes in bird's eye view", (50,80), font, 1, (255,255,255), 2)
    if left_curverad < 3000:
        cv2.putText(result, "left curvature: " + "{:.2f}".format(left_curverad)+"m", (50,330), font, 1, (255,255,255), 2)
    else:
        cv2.putText(result, "left curvature: > 3km", (50,330), font, 1, (255,255,255), 2)
    if right_curverad < 3000:
        cv2.putText(result, "right curvature: " + "{:.2f}".format(right_curverad)+"m", (50,370), font, 1, (255,255,255), 2)
    else:
        cv2.putText(result, "right curvature: > 3km", (50,370), font, 1, (255,255,255), 2)
    cv2.putText(result, "offset from left: " + "{:.2f}".format(offset)+"m", (50,410), font, 1, (255,255,255), 2)
    #cv2.putText(result, "steering to the right: " + "{:.0f}".format(steering)+"%", (800,230), font, 1, (255,255,255), 2)
    if steering != 0:
        cv2.putText(result, "{:.0f}".format(abs(steering))+"%", (620,490), font, 1, (255,255,255), 2)
    else:
        cv2.putText(result, "{:.0f}".format(abs(steering))+"%", (630,490), font, 1, (255,255,255), 2)
    #plt.imshow(result)
    #plt.show()





    print("lane detection done!")
    print("vehicle detection starts here!")

    bbox, car_bbox, car_conf = find_cars(img, ystart, ystop, scale_img, scale_car, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins) 
    car_roots = []
    #print("number of labels:", labels[1])
    if image_count == start_num:
        heat = np.zeros_like(img[:,:,0]).astype(np.float32)   
        heat = add_heat(heat,bbox)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,threshold_new)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        result, car_list, root = initial_draw_box(result, labels, car_list, clf)
        car_roots.append(root)
    elif image_count > start_num and image_count % 50 == 0:
        heat = np.zeros_like(img[:,:,0]).astype(np.float32)   
        heat = add_heat(heat,bbox)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,threshold_new)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        if labels[1] > len(car_list):
            print("label number is larger than car_list, go to new_draw_box")
            result, car_list, root = new_draw_box(result, labels, car_list, clf)
            car_roots.append(root)
    for i in range(len(car_list)):
        if car_list[i].new_detected == True:
            car_list[i].new_detected = False
            continue
        if car_list[i].lost == True:
            continue
        print("draw box for locked individual car num:", i)
        win = car_bbox[i]
        car_heat = np.zeros_like(img[:,:,0]).astype(np.float32)
        car_heat = add_heat(car_heat,win)
        if car_list[i].merged == True:
            car_heat = apply_threshold(car_heat, threshold_merged)
        else:
            car_heat = apply_threshold(car_heat, threshold_locked)
        car_heatmap = np.clip(car_heat, 0, 255)
        car_labels = label(car_heatmap)
        result, car_list, root = car_draw_box(result, car_labels, car_list, i, car_conf[i], clf)
        car_roots.append(root)

    print("num of car roots:", len(car_roots))
    for i in range(len(car_roots)):
        print(i)
        root = car_roots[i]
        if root[0] != None and root[1] != None:
            orientation, distance = find_root_dist_orient(root)
            print("orientation:", orientation)
            print("distance:", distance)
            if car_list[i].merged == True:
                cv2.putText(result, "merged car_"+"{:.0f}".format(i+1)+" distance: "+"{:.1f}".format(distance)+"m", (800,120+80*i), font, 1, (255,255,255), 2)
                cv2.putText(result, "merged car_"+"{:.0f}".format(i+1)+" orientation: "+"{:.0f}".format(orientation)+'deg', (800,160+80*i), font, 1, (255,255,255), 2)
            else:
                cv2.putText(result, "car_"+"{:.0f}".format(i+1)+" distance: "+"{:.1f}".format(distance)+"m", (800,120+80*i), font, 1, (255,255,255), 2)
                cv2.putText(result, "car_"+"{:.0f}".format(i+1)+" orientation: "+"{:.0f}".format(orientation)+'deg', (800,160+80*i), font, 1, (255,255,255), 2)


    if len(car_list) > 1:
        for i in range(len(car_list)-1): #workaround for this video
            car_1 = car_list[i]
            car_2 = car_list[i+1]
            if car_1.recent_window_length[-1] != None and car_2.recent_window_length[-1] != None:
                if ((car_1.center_x - car_1.recent_window_length[-1]*0.5) - (car_2.center_x - car_2.recent_window_length[-1]*0.5))>0:
                    #car_2.draw_window_length = car_2.draw_window_length + car_1.draw_window_length
                    #car_2.mean_window_length = car_2.recent_window_length[-1] + car_1.recent_window_length[-1]
                    #for j in range(len(car_2.recent_window_length)):
                    #    car_2.recent_window_length[j] = car_2.recent_window_length[j] + car_1.recent_window_length[-1]
                    car_list[i+1].merged = True
                    print("!!!!!!!!!!!!merged car found!!!!!!!!!!!!!!")
                    del(car_list[i])

    
    cv2.putText(result, "vihecles in bird's eye view", (800,80), font, 1, (255,255,255), 2)

    '''
    if image_count % 20 == 0:
        for car in car_list:
            car.draw_window_length = car.mean_window_length
            car.draw_window_height = car.mean_window_height
    '''

    #cv2.putText(result, "image id: " + str(image_count), (800,50), font, 1, (255,255,255), 2)

    if image_count > need_plot or image_count % plot_period == 0:
        plt.imshow(result)
        plt.show()
        '''
        fig = plt.figure(figsize=(24,12))
        plt.subplot(121)
        plt.imshow(result)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.show()
        '''
    
    return result



output = './final_11.mp4'
clip3 = VideoFileClip('./project_video.mp4')
#output = './P4/hard/challenge_video.mp4'
#clip3 = VideoFileClip('./P4/challenge_video.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(output, audio=False)
