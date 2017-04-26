import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # number of images to keep tracing
        self.n = 10
        # was the line detected in the last iteration?
        self.detected = False  
        self.line_lost_succ = 0
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.meanx = None 
        # polyfit param in last n iterations  
        self.recent_fit_param = []
        #polynomial coefficients averaged over the last n iterations
        self.mean_fit_param = None  
        self.fit_param_std = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.curverad = None 
        self.recent_curverad = []
        self.mean_curverad = None
        self.curverad_std = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        self.next_x_base = None
        self.direction = None # right:1 left:-1
        self.steering = None
        self.recent_steering = []
        self.mean_steering = None
        self.steering_std = None

    def fit_param_adjust(self, fit_param, m):
        assert self.detected == True
        if m <= self.n:
            self.recent_fit_param.append(fit_param)
            return fit_param
        self.mean_fit_param = np.mean(self.recent_fit_param, axis=0)
        self.fit_param_std = np.std(self.recent_fit_param, axis=0)
        #print("new fit param: ", fit_param)
        #print("mean_fit_param: ", self.mean_fit_param)
        #print("fit_param_std: ", self.fit_param_std)
        error_scale = 3
        if (abs(fit_param[0] - self.mean_fit_param[0]) < error_scale*self.fit_param_std[0]) and (abs(fit_param[1] - self.mean_fit_param[1]) < error_scale*self.fit_param_std[1]) and (abs(fit_param[2] - self.mean_fit_param[2]) < 10*self.fit_param_std[2]):
            self.recent_fit_param.append(fit_param)
            if len(self.recent_fit_param) > self.n:
                del(self.recent_fit_param[0])
            return fit_param
        else: # maybe do not append bad fitx? What if trending towards bad?
            self.recent_fit_param.append(fit_param)
            if len(self.recent_fit_param) > self.n:
                del(self.recent_fit_param[0])
            return self.recent_fit_param[-2]

    def x_adjust(self, fitx, m):
        assert self.detected == True
        if m <= self.n:
            self.recent_xfitted.append(fitx)
            return fitx
        self.meanx = int(np.mean(self.recent_xfitted))
        mean_fitx = int(np.mean(fitx))
        #print("recent x mean: ", self.meanx)
        #print("fitx mean: ", mean_fitx)
        if abs(mean_fitx - self.meanx) < 20:
            self.recent_xfitted.append(fitx)
            if len(self.recent_xfitted) > self.n:
                del(self.recent_xfitted[0])
            return fitx
        else: # maybe do not append bad fitx? What if trending towards bad?
            self.recent_xfitted.append(fitx)
            if len(self.recent_xfitted) > self.n:
                del(self.recent_xfitted[0])
            return self.recent_xfitted[-2]

    def curverad_adjust(self, curverad, m):
        if m <= self.n:
            self.recent_curverad.append(curverad)
            return curverad
        self.mean_curverad = np.mean(self.recent_curverad, axis=0)
        self.curverad_std = np.std(self.recent_curverad, axis=0)
        #print("new fit param: ", fit_param)
        #print("mean_fit_param: ", self.mean_fit_param)
        #print("fit_param_std: ", self.fit_param_std)
        error_scale = 2
        if ( (curverad - self.mean_curverad) < (error_scale * self.curverad_std) ):
            self.recent_curverad.append(curverad)
            if len(self.recent_curverad) > self.n:
                del(self.recent_curverad[0])
            return curverad
        else: # maybe do not append bad fitx? What if trending towards bad?
            result_curverad = self.recent_curverad[-1]
            self.recent_curverad.append(curverad)
            if len(self.recent_fit_param) > self.n:
                del(self.recent_fit_param[0])
            return result_curverad

    def steering_adjust(self, steering, m):
        steering_n = 5
        if len(self.recent_steering) <= (steering_n):
            self.recent_steering.append(steering)
            return steering
        if steering == 0:
            self.recent_steering.append(steering)
            if len(self.recent_steering) > (steering_n):
                del(self.recent_steering[0])
            return steering            
        self.mean_steering = np.mean(self.recent_steering, axis=0)
        self.steering_std = np.std(self.recent_steering, axis=0)
        #print("new fit param: ", fit_param)
        #print("mean_fit_param: ", self.mean_fit_param)
        #print("fit_param_std: ", self.fit_param_std)
        error_scale = 2
        if ( (steering - self.mean_steering) < (error_scale * self.steering_std) ):
            self.recent_steering.append(steering)
            if len(self.recent_steering) > (steering_n):
                del(self.recent_steering[0])
            return steering
        else: # maybe do not append bad fitx? What if trending towards bad?
            result_steering = self.recent_steering[-1]
            self.recent_steering.append(steering)
            if len(self.recent_steering) > (steering_n):
                del(self.recent_steering[0])
            return result_steering

def calibration(nx=9, ny=6):
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = plt.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)

    chessboard_image = plt.imread('./camera_cal/calibration3.jpg')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, chessboard_image.shape[0:2],None,None)
    return mtx, dist

def PerspectiveMatrix():
    src_p1 = (253,697) #left down
    src_p2 = (585,456) #left up
    src_p3 = (700,456) #right up
    src_p4 = (1061,690) #right down
    src = np.float32([src_p1,src_p2,src_p3,src_p4])
    dst_p1 = (303,697)
    dst_p2 = (303,0)
    dst_p3 = (1011,0)
    dst_p4 = (1011,690)
    dst = np.float32([dst_p1,dst_p2,dst_p3,dst_p4])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20, 100)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    #thresh_min = 20
    #thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    img = sxbinary
    grad_binary = np.copy(img) # Remove this line
    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(30, 100)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return mag_binary

def dir_thresh(img, sobel_kernel=3, thresh=(0.7, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def s_color_thresh(undist, thresh=(170, 255)):
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary

def l_color_thresh(undist, thresh=(100, 255)):
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh[0]) & (l_channel <= thresh[1])] = 1
    return l_binary







