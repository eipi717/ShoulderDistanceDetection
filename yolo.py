# Use YOLO algorithm to detect person and crop it to new file [body{i}] for i = 1, 2, 3, ...

import numpy as np
import time
import cv2
import imutils

# input image
INPUT_FILE='/Users/nicholas717/Downloads/Work/code/report_shoulder/gp1.jpg'

# labels file's location
LABELS_FILE='/Users/nicholas717/PycharmProjects/PythonProject/data/coco.names'

# yolo.cfg file's location
CONFIG_FILE='/Users/nicholas717/PycharmProjects/PythonProject/cfg/yolov3.cfg'

# yolo.weights's location
WEIGHTS_FILE='/Users/nicholas717/PycharmProjects/PythonProject/yolov3.weights'

# threshold value
CONFIDENCE_THRESHOLD=0.3


LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

image = cv2.imread(INPUT_FILE)
#image = cv2.resize(image, (800, 600))
#image = imutils.resize(image, height = image.shape[0] // 2)
(H, W) = image.shape[:2]
print("img dim: ", (H, W))

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > CONFIDENCE_THRESHOLD:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
	CONFIDENCE_THRESHOLD)

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		color = [int(c) for c in COLORS[classIDs[i]]]

		# Create bounding box, for the half of body
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		body = image[y//2: (y+h)//2, x: x+w]
		cv2.imwrite('./test/body' + str(i) + '.jpg', body)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

# show the output image
cv2.imwrite("./test/example.png", image)