# TODO: check if the object is distorted first
#  **** length of the left and right shoulders are different based on the face, ****

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

        # Find the mid-point of the face
        right_eye_inner = np.array((landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y)) * img_wh
        left_eye_inner = np.array((landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y)) * img_wh
        head_mid = selfmath.MidPt(right_eye_inner, left_eye_inner)
        head_mid = [int(x) for x in head_mid]

        # Find the mid-point of the lower part of the body
        left_foot_index = np.array((landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y)) * img_wh
        right_foot_index = np.array((landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)) * img_wh
        foot_mid = selfmath.MidPt(left_foot_index, right_foot_index)
        foot_mid = [int(x) for x in foot_mid]

        # Find the distortion of the shoulders
        # by comparing the distance between head_mid and each shoulder
        dist_leftShoulder2head = selfmath.dist(shoulder_left, head_mid)
        dist_rightShoulder2head = selfmath.dist(shoulder_right, head_mid)

        # Draw line for indication
        cv2.line(img, head_mid, (head_mid[0], foot_mid[1]), (0, 255, 0), 3)

        # Calculate the object height
        # Consider the vertical direction only so the line is straight
        body_dist_vert = np.abs(head_mid[1] - foot_mid[1])
        print("Body height: ", body_dist_vert)

        # Print the pixel coordinates of left and right shoulders
        #
        #print("LEFT shoulder: ", tuple(shoulder_left))
        #print("RIGHT shoulder: ", tuple(shoulder_right))
        #

        # Print the Distance between two shoulders
        shoulderDist = round(selfmath.dist(shoulder_right, shoulder_left), 2)
        print(f"ShoulderDist: {shoulderDist}")

        # Find and print the ratio between shoulderDistance and object height
        ratio_percentage = round(((shoulderDist / body_dist_vert) * 100), 2)
        print(f"Ratio of shoulder size to Body height: {ratio_percentage} \n"
              f"Shoulder distance is {ratio_percentage}% of the object height")
        print(f"Distance between face and left shoulder: {dist_leftShoulder2head}\n"
              f"Distance between face and right shoulder: {dist_rightShoulder2head}")

        if (np.abs(dist_leftShoulder2head - dist_rightShoulder2head) > 50):
            print("The body turned!")
            if (dist_rightShoulder2head > dist_leftShoulder2head):
                print("The shoulder turned right! ")
            elif (dist_leftShoulder2head > dist_rightShoulder2head):
                print("The shoulder turned left! ")

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
                print(f'{filename} not working!')