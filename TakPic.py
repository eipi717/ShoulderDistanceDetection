from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

tetst_ = "/Users/nicholas717/Downloads/1.jpeg"
img = cv2.imread(tetst_)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure()

#plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
#ascent = img.ascent()
#result = ndimage.sobel(img)
result = gaussian_filter(img, sigma=5)
ax1.imshow(img)
cv2.imshow("Hello", result)
#ax2.imshow(result)
plt.show()