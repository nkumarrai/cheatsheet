import cv2, os, sys
import numpy as np

http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html#py-table-of-content-imgproc

#Playing with Images
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html#py-display-image

#Playing with videos
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video

# Read in input images
input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

#Find HSV value of a color
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print hsv_green

#Mask - Threshold the HSV image to get only blue colors (color values are blue colors)
mask = cv2.inRange(hsv, np.array([110,50,50]), np.array([130,255,255]))
#Apply Mask - Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)

#Resize - there is no anti-aliasing flag in opencv. Use PILImage for that.
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

#Translation - create transformation matrix and call warpAffine.
M = np.float32([[1,0,100],[0,1,50]]) 
dst = cv2.warpAffine(img,M,(cols,rows)) #there are options to cover the boundary pixels.
#Rotation - create transformation matrix and call warpAffine.
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1) #(center, rotation, scale)
dst = cv2.warpAffine(img,M,(cols,rows))
#Affine transform - parallel lines will still be parallel. Need three points.
#Parallelogram.
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
#Perspective transform - straight lines will be straight lines. Need 4 points.
#Any random figure with straight lines still as the straight lines.
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))

#Thresholding
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#Adaptive thresholding - (src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, 
	cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	cv2.THRESH_BINARY,11,2)
#Thresholding - Otsu’s binarization - for bimodal images (with two peaks in the histogram)
# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering - BEST result.. filter the noise with gaussian filter first.
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#2D convolution
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
#Blurring - averaging filter or use cv2.boxFilter()
blur = cv2.blur(img,(5,5))
#Gaussian blurring - it blurs the edges too
blur = cv2.GaussianBlur(img,(5,5),0)
#Median blurring - remove salt n pepper noise
median = cv2.medianBlur(img,5)
#Bilateral blurring - takes two gaussian filters - the second one is pixel difference to make sure the edges aren't blurred.
blur = cv2.bilateralFilter(img,9,75,75)

#Morphological transformation
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
#Opening - Erosion followed by Dilation
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#Closing - Dilation followed by Erosion
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#Morphological Gradient - difference between dilation and erosion of an image.
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
#Top hat - difference between input image and Opening of the image.
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
#Black hat - difference between the closing of the input image and input image.
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
#Get a custom kernel
#Rectangular Kernel
cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

#Gradient
#Sobel and Scharr filter 
#Sobel operators is a joint Gausssian smoothing plus differentiation operation, so it is more resistant to noise. 
#You can specify the direction of derivatives to be taken, vertical or horizontal (by the arguments, yorder and xorder respectively). 
#You can also specify the size of kernel by the argument ksize. If ksize = -1, a 3x3 Scharr filter is used which gives better results 
#than 3x3 Sobel filter. Please see the docs for kernels used.
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#Laplacian
laplacian = cv2.Laplacian(img,cv2.CV_64F)
#In our last example, output datatype is cv2.CV_8U or np.uint8. But there is a slight problem with that. 
#Black-to-White transition is taken as Positive slope (it has a positive value) while White-to-Black transition is taken 
#as a Negative slope (It has negative value). So when you convert data to np.uint8, all negative slopes are made zero. In simple words, you miss that edge.
#If you want to detect both edges, better option is to keep the output datatype to some higher forms, like cv2.CV_16S, cv2.CV_64F etc, 
#take its absolute value and then convert back to cv2.CV_8U. Below code demonstrates this procedure for a horizontal Sobel filter and difference in results.
# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

#Canny Edge Detection - (image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) - 1) Noise reduction - gaussian (5x5), 2)Finding intensity
# gradient, 3) Non-maximum Suppression using the thresholds.
#First argument is our input image. Second and third arguments are our minVal and maxVal respectively. 
#Third argument is aperture_size. It is the size of Sobel kernel used for find image gradients.
edges = cv2.Canny(img,100,200)

#Image Pyramids
#like face, we are not sure at what size the object will be present in the image. In that case, we will need to create a set of images with different 
#resolution and search for object in all the images. 
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html#py-pyramids

#Contours 
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html#table-of-content-contours
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Draw contours
cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.drawContours(img, contours, 3, (0,255,0), 3)
cnt = contours[4]
cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
#Image moments - use contours to find different properties of the image
#Centroid, area, arc length
cnt = contours[0]
M = cv2.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt,True)
#Get a convex hull
hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
#Check convexity
k = cv2.isContourConvex(cnt)
#Bouding rectangle (straight) - uses a set of points - can be contours or something else
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#Bouding rectangle (rotated) - uses a set of points - can be contours or something else
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)
#Minimum enclosing circle
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(img,center,radius,(0,255,0),2)
#Fitting an ellipse
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img,ellipse,(0,255,0),2)
#Fitting a line
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
#Contour Properties
#Learn to find different properties of contours like Solidity, Mean Intensity etc.
#Contours : More Functions
#Learn to find convexity defects, pointPolygonTest, match different shapes etc.
#Contours Hierarchy
#Learn about Contour Hierarchy

#Histograms
img = cv2.imread('home.jpg')
color = ('b','g','r')
# for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
#with mask - create a mask using np.array() and setting of some pixels to zero
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
#Histogram equalization
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
#Adaptive histogram equalization
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#py-histogram-equalization
#2D Histograms .. plotted for two features hue and saturation. Earlier it was just for grayscale color values.
img = cv2.imread('home.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# channels = [0,1] because we need to process both H and S plane.
# bins = [180,256] 180 for H plane and 256 for S plane.
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist,interpolation = 'nearest')
plt.show()
#Histogram Backprojection - use histogram to find all instances of an object in the image
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_backprojection/py_histogram_backprojection.html#histogram-backprojection
...cv2.calcBackProject()...

#Fourier transform
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html#fourier-transform

#Matching template
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#py-template-matching

#Hough line - detect lines
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html#hough-circles

#Hough circles - detect circles
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html#hough-circles

#Image Segmentation with Watershed Algorithm
#Learn to segment images with watershed segmentation

#Interactive Foreground Extraction using GrabCut Algorithm
#Learn to extract foreground with GrabCut algorithm

#Feature Detection and Description
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html#py-table-of-content-feature2d
#Feature Detection - Look for the regions in images which have maximum variation when moved (by a small amount) in all regions around it.
#Feature Description - generate a vector or some representation of the region for CV to work.
#Harris Corner Detection - (rotation invariant but not scale invariant) interesting explanation of eigen vectors. 
#How they translate into "flat surfaces", "edges" or "corners"?
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html#harris-corners
#Other corner detector
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html#shi-tomasi

#SIFT (Scale-Invariant Feature Transform) 
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html#sift-intro
http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)
img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)
sift = cv2.SIFT()
kp, des = sift.detectAndCompute(gray,None)

#FAST Feature Detector
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
# Print all default params
print "Threshold: ", fast.getInt('threshold')	
print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
print "neighborhood: ", fast.getInt('type')
print "Total Keypoints with nonmaxSuppression: ", len(kp)
cv2.imwrite('fast_true.png',img2)
# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)
print "Total Keypoints without nonmaxSuppression: ", len(kp)
img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
cv2.imwrite('fast_false.png',img3)

#BRIEF (Binary Robust Independent Elementary Features)
img = cv2.imread('simple.jpg',0)
# Initiate STAR detector
star = cv2.FeatureDetector_create("STAR")
# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

#ORB (Oriented FAST and Rotated BRIEF)
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html#orb

#Feature Matching
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

#Feature Matching + Homography to find Objects
http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#py-feature-homography