# FFT , IFFT transform , numpy absolute

dft = np.fft.fft2(np.float32(im),newsize)
return np.fft.fftshift(dft)
f_ishift = np.fft.ifftshift(shift)
img_back = np.fft.ifft2(f_ishift)
return np.abs(img_back)
abs_sobel64f = np.absolute(sobelx64f)

# Add a new axis
im = im[:,:,np.newaxis]

# Compare the type
type(im[0,0,0]) == np.dtype(np.float32)
type(im[0,0,0]) == np.dtype(np.float16)

# Change the type
im = np.array(im*255).astype('uint8')

# Create zeros and ones
out = np.zeros(im.shape)
out = np.ones(im.shape)

# zeros like
m = np.zeros_like(A, dtype='float32')

# Cumulative sum, masked_equal, fill with zeros????????????????
cdf = np.cumsum(hist_frame)
cdf_m = np.ma.masked_equal(cdf,0)
cdf = np.ma.filled(cdf_m,0).astype('uint8')

# numpy clip
img_out = np.clip(img_out, 0, 255)

# Concatenate or stack horizontally
ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))

# numpy log
magnitude_spectrum = 20*np.log(np.abs(fshift))

# numpy add or subtract
LA = np.subtract(np.array(gpA[i-1]).astype('float32'), np.array(cv2.pyrUp(np.array(gpA[i]).astype('float32')).astype('float32')))
ls_ = cv2.add(np.array(ls_).astype('float32'), np.array(LS[i]).astype('float32'))

# type uint8
green = np.uint8([[[0,255,0 ]]])

# create a range
ask = cv2.inRange(hsv, np.array([110,50,50]), np.array([130,255,255]))

# what's int0 ?????????????????
box = np.int0(box)

# Matrix numpy dot - matrix multiplication
X = np.dot(K,X)

# Reshape the array from 1D to 2D back to 1D
src_pts = np.float32(pts1).reshape(-1,1,2)

# Create mask with multiple conditions
maskr = np.all([img[:,:,0] > 240, img[:,:,1] < 20, img[:,:,2] < 20], axis=0)
maskb = np.all([img[:,:,0] < 20, img[:,:,1] < 20, img[:,:,2] > 240], axis=0

# Numpy arrange , create an array
idxs = np.arange(len(train_filepaths))

# Numpy shuffle
np.random.shuffle(idxs)