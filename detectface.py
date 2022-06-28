# Detect faces and crop it to new file [face{i}] for i = 1, 2, 3, ...

import cv2
import imutils
from selfmath import dist

def Detection(img):
    dist2ref = [0]

    # Read the input image
    img = cv2.imread(img)
    #img = imutils.resize(img, width = 900)

    # Create a reference point
    # To obtain the face distance, in ratio
    ref = ((img.shape[1] // 2), (img.shape[0] // 2))
    cv2.circle(img=img, center=ref, radius=3, color=(0, 0, 255), thickness=5)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('/Users/nicholas717/opt/anaconda3/pkgs/opencv-4.5.5-py38hc2a0b3f_0/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    i = 1

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x-5, y-5), (x + w + 5, y + h + 2), (0, 0, 255), 2)
        face_mid = ((x + x + w + 5) // 2, (y + y + h + 2) // 2)
        dist2ref.append(dist(face_mid, ref))

        # Crop the face from the original image
        faces = img[y-20:y + h + 10, x:x + w + 10]
        cv2.imwrite('face' + str(i) + '.jpg', faces)
        i += 1

    # Display the output
    cv2.imwrite('detcted.jpg', img)
    #img = imutils.resize(img, width=600)
    #cv2.imshow('img', img)
    #cv2.waitKey()
    return dist2ref
