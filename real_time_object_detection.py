# To run the script, execute the following commands
# workon cv
# python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# import tree so we can write our data into an xml file
import xml.etree.cElementTree as ET

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True,  help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pre-trained model")
ap.add_argument("-c", "--probability", type = float, default = 0.2, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

unique_label = input("Enter object to keep track of: ")
print("Tracking " + str(unique_label))
unique_counter = 0
track_end = ""
track_start = ""
# initialize the list of class labels MobileNet SSD was trained to detect
# and generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the xml file format
root = ET.Element("root")
detections_element = ET.SubElement(root, "detections")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
start_time = time.time()
# loop over the frames from the video stream
while True:
	# resize the video stream window at a maximum width of 500 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	# Binary Large Object = BLOB
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and get the detections
	net.setInput(blob)
	detections = net.forward()

	# unique_label = "person"
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the probability of the prediction
		probability = detections[0, 0, i, 2]

		# filter out weak detections by ensuring that probability is greater than the min probability
		if probability > args["probability"]:
			# extract the index of the class label from the
			# 'detections', then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], probability * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			object_label = format(CLASSES[idx])
			if(object_label == unique_label):
				print(unique_label) # display detection on terminal
				if(unique_counter == 0):
					track_start = int(time.time() - start_time)
					print("Object located at " + str(track_start) + " seconds")
					unique_counter = 1
				else:
					if(unique_counter == 1):
						unique_counter == 2
						track_end = int(time.time() - start_time)
						print("still on screen @ " + str(track_end) + " seconds")
			# calculate start time - current time to keep a seconds timer
			elapsed_time = int(time.time() - start_time)
			# print(int(elapsed_time))
			# Keep each pair labels detected + probability of the label + total frames per second
			ET.SubElement(detections_element, "object", name="detection: confidence").text = label
			ET.SubElement(detections_element, "time", name="seconds").text = str(elapsed_time) + " sec"
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#add total fps and total elapsed time to xml doc
session_info = ET.SubElement(root, "info")
ET.SubElement(session_info, "total_time").text = "Session duration: " + format(fps.elapsed()) + " seconds"
ET.SubElement(session_info, "fps").text = "Average " + format(fps.fps()) + " frames/sec"
# generate xml document
tree = ET.ElementTree(root)
tree.write("session.xml")

# cleanup
cv2.destroyAllWindows()
vs.stop()

# display for how much time object of interest was visible
period = track_end - track_start
print(unique_label + " was on screen for a total of " + str(period) + " seconds (" + str(track_start) + "-" + str(track_end) + " sec)")
