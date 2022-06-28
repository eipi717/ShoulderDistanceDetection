import cv2
import numpy as np

img = cv2.imread("/Users/nicholas717/Downloads/darkImage/16504506.jpg", 0)

Histeq = cv2.equalizeHist(img)

clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(5, 5))
clahe = clahe.apply(img)

res = np.hstack((Histeq, img, clahe))
cv2.imshow("test", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
