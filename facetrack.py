import cv2
import dlib
import imutils

def landmarks(img):
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("/Users/nicholas717/Downloads/shape_predictor_68_face_landmarks.dat")

# read the image
#img = cv2.imread("/Users/nicholas717/Downloads/3.jpg")

# Convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point


    # Create landmark object
        landmarks = predictor(image=gray, box=face)

    # Loop through all the points
        for n in range(6, 11):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

        # Draw a circle
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

# show the image
#img = imutils.resize(img, width=800)
    cv2.imshow(winname="Face_landmarks", mat=img)

# Delay between every fram
    cv2.waitKey(delay=0)

# Close all windows
    cv2.destroyAllWindows()

