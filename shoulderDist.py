import mediapipe as mp
import cv2
import numpy as np
import selfmath
import imutils
import os

def shoulder(img_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        img = cv2.imread(img_path)

        # Resize the cropped image, in aspect ratio
        img = imutils.resize(image=img, width=500)
        print(img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

            # Render detections
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Get the image shape, in the form of (width, height)
        img_height, img_width, _ = img.shape
        img_wh = np.array((img_width, img_height))

        # Convert from normalized coordinates to pixel coordinates
        # By multiplying the image shape
        shoulder_right = np.array((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)) * img_wh
        shoulder_left = np.array((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)) * img_wh

        cv2.circle(img=img, center=(round((shoulder_right[0])), round(shoulder_right[1])), radius=15, color=(0, 0, 255), thickness=3)
        cv2.circle(img=img, center=(round((shoulder_left[0])), round(shoulder_left[1])), radius=15, color=(0, 0, 255), thickness=3)

        # Print the pixel coordinates of left and right shoulders
        print("LEFT shoulder: ", tuple(shoulder_left))
        print("RIGHT shoulder: ", tuple(shoulder_right))

        # Print the Distance between two shoulders
        shoulderDist = selfmath.dist(shoulder_right, shoulder_left)
        print("Dist: ", round(shoulderDist, 2))

        # Display the cropped image
        cv2.imshow('Hello', img)

        # Delay between every fram
        cv2.waitKey(delay=0)

        # Close all windows
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":

    # Test the image in test_image/ dir
    for filename in os.listdir('test_image'):
        # Ignore example.png and .DS_Store
        if filename not in ('example.png', '.DS_Store'):
            try:
                shoulder("/Users/nicholas717/PycharmProjects/PythonProject/test_image/" + str(filename))
            except:
                # Print error message if the detection fail
                print(filename, "Not work!")
