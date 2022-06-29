import mediapipe as mp
import cv2
import numpy as np
import selfmath
import imutils

def shoulder(img_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        img = cv2.imread(img_path)
        print(img.shape)
        #img = imutils.resize(image=img, width=500, height=200)
        #img = cv2.resize(img, (220, 550))
        print("After", img.shape)
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


        print("LEFT shoulder: ", tuple(shoulder_left))
        print("RIGHT shoulder: ", tuple(shoulder_right))

        print("Dist: ", round(selfmath.dist(shoulder_right, shoulder_left), 2))
        shoulderDist = selfmath.dist(shoulder_right, shoulder_left)

        cv2.imshow('Hello', img)

        # Delay between every fram
        cv2.waitKey(delay=0)

        # Close all windows
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    img_path = "/Users/nicholas717/Downloads/Work/code/2.jpg"

    for i in range(100):
        try:
            shoulder("/Users/nicholas717/PycharmProjects/PythonProject/test_image/body" + str(i) + ".jpg")
        except:
            print(i, " Not work!")
