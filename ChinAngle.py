import cv2
import dlib
import numpy as np
from selfmath import GetSlope, LinesInAngle, MidPt, dist
import imutils
from detectface import Detection

def DetectChin(img, p, refdis):

    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("/Users/nicholas717/Downloads/Work/code/shape_predictor_68_face_landmarks_GTX.dat")

    # read the image
    img = cv2.imread(str(img))
    a, b, _ = img.shape
    img = imutils.resize(img, width = 350, height=350)

    # Convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        # Create landmark object
        landmarks = predictor(image=gray, box=face)


        # Loop through all the points
        X = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            X.append((x, y))

            # Find the mid point of two eyebrows
            if n is 22:
                mid_pt = MidPt(X[21], X[22])
                cv2.circle(img=img, center=(int(mid_pt[0]), int(mid_pt[1])), radius=3, color=(0, 255, 255), thickness=1)
                line = np.array([np.array(mid_pt), np.array(X[8])], dtype=int)

                #cv2.drawContours(img, [line], 0, (0, 255, 0), 2)

                distance = dist(mid_pt, X[8])

            # Obtain the angle of right / left face to the nose
            elif n is 33:
                x14 = np.array(X[14])
                x15 = np.array(X[15])
                x33 = np.array(X[33])
                x01 = np.array(X[1])
                x02 = np.array(X[2])

                m1433 = GetSlope(x14, x33)
                m1533 = GetSlope(x15, x33)
                m0133 = GetSlope(x01, x33)
                m0233 = GetSlope(x02, x33)

                """
                cv2.drawContours(img, [np.array([x01, x33])], 0, (255, 255, 255), 2)
                cv2.drawContours(img, [np.array([x02, x33])], 0, (255, 255, 255), 2)
                cv2.drawContours(img, [np.array([x14, x33])], 0, (255, 255, 255), 2)
                cv2.drawContours(img, [np.array([x15, x33])], 0, (255, 255, 255), 2)
                """

                # Draw face width
                x00 = np.array(X[0])
                x16 = np.array(X[16])
                cv2.drawContours(img, [np.array([x00, x16])], 0, (0, 0, 100), 2)

                angle_left = LinesInAngle(m1433, m1533)
                angle_right = LinesInAngle(m0133, m0233)
                print("Left angle: ", angle_left)
                print("Right angle: ", angle_right)
                print("Distance to ref point is: ", refdis)


            # Draw a circle
            #cv2.putText(img, str(n), (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)

    # Find the angle of the chin
    # By the slope of two lines on left and right
        m1 = GetSlope(X[6], X[8])
        m2 = GetSlope(X[8], X[10])
        face_width = dist(x00, x16)
        angle = np.abs(LinesInAngle(m1, m2))
        print("Ori angle is: ", angle)
        print("Face width is: ", face_width)

    if p == True:
        # Put the angle on the image
        #cv2.putText(img, str(angle), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # show the image
        cv2.imshow("Hello", img)

        # Export the image to test_num.jpg
        #cv2.imwrite('test' + str(j) + '.jpg', img)


    # Delay between every fram
    cv2.waitKey(delay=0)

    # Close all windows
    cv2.destroyAllWindows()
    return

if __name__ == "__main__" :

    img_ori = "/Users/nicholas717/Downloads/Work/code/2.webp"
    dist2ref = Detection(img_ori)

    A = []
    for i in range(1, 50):
        try:
            print(i)
            a = DetectChin("/Users/nicholas717/PycharmProjects/PythonProject/face" + str(i) + ".jpg", True, dist2ref[i])
            A.append((i, a))
        except:
            print("No", i)
            continue
