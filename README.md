## Real-time object detection with deep learning and OpenCV
#### - project developed by Alina Balaur

## Table of Contents

- Real-time object detection with deep learning and OpenCV
   - Introduction
   - Ubuntu 16+: How to install OpenCV
      - #1. Install OpenCV dependencies.
      - #2. Download OpenCV
      - #3. Setup your python environment (2.7 or 3+)
         - Creating your Python environment.
         - Verifying that you are in the “cv” virtual environment
         - Install NumPy into your Python virtual environment
      - #4. Configuring and compiling OpenCV
      - #5. Finish your OpenCV installation
         - For Python 2.7:
         - For Python 3.5:
      - #6. Testing your OpenCV installation
   - Object detection in video with deep learning and OpenCV
         - Real-time deep learning object detection results
   - Bibliography & sources


### Introduction

In this guide we will walk through all the steps needed to set up our machine so we can then apply real-time object detection using **deep learning** and **OpenCV** to work with video streams and video files. In order to do that we will use the **VideoStream** class that comes with the **imutils** package**.**

Before we dive into the coding part, let’s first get our machine ready for implementation. For the purpose of this project, I ended up using Ubuntu as it was much more “error-less” to set up all the necessary packages. You can either use a dedicated Ubuntu installation or simply an Ubuntu Vmware one. I personally used both and they work just fine. Once you have an Ubuntu machine ready for use, you can proceed to setting up OpenCV.

### Ubuntu 16+: How to install OpenCV

Ubuntu 16+ ships with with both **python 2.7** and **python 3.5** installed. You can access them by typing the python or python3 commands accordingly in your terminal. The default python version used is the 2.7 one. However, we are going to set up OpenCV for both versions in case one prefers to work with python 2.7.

#### #1. Install OpenCV dependencies.

All of the following steps will be accomplished by using our terminal. To start, open up your command line and update the ​ **apt-get** package manager in order to refresh and upgrade your pre-installed packages/libraries:

```
$​ sudo apt-get update
$​ sudo apt-get upgrade
```
Next, install some **developer tools**:

```
$​ sudo apt-get install build-essential cmake pkg-config
```
The pkg-config package will (very likely) be already installed on your system, but be sure to include it in the above apt-get command just in case. The **cmake** program is used to automatically configure our OpenCV build.

OpenCV is an **_image processing_** and **_computer vision_** library. Therefore, OpenCV needs to be able to load various image file formats from disk such as JPEG, PNG, TIFF etc. In order to load these images from our disk, OpenCV actually calls other image I/O libraries that facilitate the loading and decoding process. We will install the necessary ones below:

```
$​ sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev
libpng12-dev
```
Now, to install packages used to process video streams and access frames from cameras, run:

```
$​ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
libv4l-dev
$​ sudo apt-get install libxvidcore-dev libx264-dev
```
OpenCV ships with a very limited set of GUI tools. These GUI tools allow us to display an image to our screen (cv2.imshow), wait for/record key presses (cv2.waitKey), track mouse events (cv2.setMouseCallback), and create simple GUI elements such as trackbars etc. Nothing too fancy here, these are just simple tools that allow you to debug your code and build very simple applications.

Internally, the name of the module that handles OpenCV GUI operations is ​ highgui ​. The highgui ​module relies on the GTK library, which you should install using the following command:

```
$​ sudo apt-get install libgtk-3-dev
```
Next, we install libraries that are used to optimize various functionalities inside OpenCV,
such as matrix operations:

```
$​ sudo apt-get install libatlas-base-dev gfortran
```

We’ll wrap up step #1 by installing the Python development headers and libraries for both Python 2.7 and Python 3.5 (this way you have both):

```
$​ sudo apt-get install python2.7-dev python3.5-dev
```

**_Note_** _: If you do not install the Python development headers and static library, you’ll run into issues during Step #4 where we run cmake to configure the OpenCV build. If these headers are not installed, then the cmake command will be unable to automatically determine the proper values of the Python interpreter and Python libraries. In short, the output of this section will look “empty” and you will not be able to build the Python bindings._


#### #2. Download OpenCV


The most recent version of OpenCV at this time is 3.4.1, which we will download a .zip of and unzip using the commands:

```
$​ cd ~
$​ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.1.zip
$​ unzip opencv.zip
```
When new versions of OpenCV are released you can check the official OpenCV GitHub and download the latest release by simply changing the version number of the .zip. We will need the opencv_contrib repository as well:

```
$​ wget -O opencv_contrib.zip
https://github.com/Itseez/opencv_contrib/archive/3.4.1.zip
$​ unzip opencv_contrib.zip
```
**_Note_** _: Both your_ **_opencv_** _and_ **_opencv_contrib_** _versions should be the same (in this case, 3.4.1). If the versions numbers do not match up, you could very easily run into compile time errors (or worse, runtime errors that are near impossible to debug)._


#### #3. Setup your python environment (2.7 or 3+)


We are now ready to start configuring our Python development environment for the build. The first step is to install ​pip​, a Python package manager:

```
$​ cd ~
$​ wget https://bootstrap.pypa.io/get-pip.py
$​ sudo python get-pip.py
```

Now take notice; we are going to install virtualenv and virtualenvwrapper. These are two Python packages that allow us to create separate python environments for
each project that we might be working on.

```
$​ sudo pip install virtualenv virtualenvwrapper
$​ sudo rm -rf ~/get-pip.py ~/.cache/pip
```

Once we have virtualenv and virtualenvwrapper installed, we need to update our ~/.bashrc file to include the following lines at the bottom of the file:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```

The ~/.bashrc file is simply a **shell script** that Bash runs whenever you launch a new terminal. You normally use this file to set various configurations. In this case, we are setting an environment variable called WORKON_HOME to point to the directory where our Python virtual environments live. We then load any necessary configurations from virtualenvwrapper.

To update your ~/.bashrc file simply use a standard text editor.

```
$​ nano ~/.bashrc
```
Save and close once you added the lines at the bottom of the file and reload the changes to bash:

```
source ~/.bashrc
```
Now that we have installed virtualenv and virtualenvwrapper, the next step is to actually **create** the **Python virtual environment** - we do this using the mkvirtualenv command.

But before executing this command, you need to make a choice: Do you want to use Python 2.7 or Python 3? Your choice will determine which command you run in the following section.


##### Creating your Python environment.


If you decide to use Python 2.7, use the following command to create a Python 2 virtual environment:

```
$​ mkvirtualenv cv -p python
Otherwise, use this command to create a Python 3 virtual environment:
$​ mkvirtualenv cv -p python
```

Regardless of which Python command you decide to use, the end result is that we have created a Python virtual environment named ​ **cv** (short for “computer vision”). You can of course name this virtual environment whatever you like (and create as many Python virtual environments as you want).


##### Verifying that you are in the “cv” virtual environment


Whenever you open up a new terminal, you’ll need to use the workon command to re-access your **cv** virtual environment:

```
$​ workon cv
```

To validate that you are in the cv virtual environment, simply examine your command line - if you see the text (cv) preceding your prompt, then you are in the cv virtual
environment:


##### Install NumPy into your Python virtual environment


The final step before we compile OpenCV is to install **NumPy**, a Python package used for **numerical processing**. To install NumPy, ensure you are in the **cv** virtual environment (otherwise NumPy will be installed into the _system version of Python rather than the cv environment_ ). From there execute the following command:

```
$​ pip install numpy
```


#### #4. Configuring and compiling OpenCV


At this point, all of our necessary prerequisites have been installed - we are now ready to compile OpenCV! But before we do that, double-check that you are in the cv virtual environment by examining your prompt and if not, use the workon command:

```
$​ workon cv
```

After ensuring you are in the cv virtual environment, we can setup and configure our build using Cmake:

```
$​ cd ~/opencv-3.4.1/
$​ mkdir build
$​ cd build
$​ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.4.1/modules \
-D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
-D BUILD_EXAMPLES=ON ..
```

The above commands change the directory to ~/opencv-3.4.1, which is where you downloaded and unarchived the previous .zip files.

**_Note_** _: If you are getting an error related to stdlib.h: No such file or directory during either the cmake or make phase of this tutorial you’ll also need to include the following option to CMake: -D ENABLE_PRECOMPILED_HEADERS=OFF. In this case you should delete your build directory, re-create it, and then re-run CMake with the above option included. This will resolve the stdlib.h error. _

Inside this directory we create a sub-directory named build and switch into it. The build directory is where the actual compile is going to take place. Finally, we execute cmake to configure our build. Assuming your CMake command exited without any errors, you can now compile OpenCV:

```
$​ make -j
```

The -j switch controls the number of processes to be used when compiling OpenCV - you’ll want to set this value to the number of processors/cores on your machine. In my case, I have a quad-core processor, so I set -j. The last step is to actually install OpenCV 3+ on Ubuntu:

```
$​ sudo make install
$​ sudo ldconfig
```

#### #5. Finish your OpenCV installation

##### For Python 2.7:

After running sudo make install , your Python 2.7 bindings for OpenCV 3+ should now be located in **/usr/local/lib/python-2.7/site-packages/**. You can verify this using the **ls** command:

```
$​ ls -l /usr/local/lib/python2.7/site-packages/
```
The final step is to sym-link our OpenCV **cv2.so** bindings into our **cv** virtual environment for Python 2.7:

```
$​ ​cd​ ~/.virtualenvs/cv/lib/python2.7/site-packages/
$​ ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
```

##### For Python 3.5:

After running sudo make install, your OpenCV + Python 3 bindings should be located in **/usr/local/lib/python3.5/site-packages/**. Again, you can verify this using the ls command:

```
$​ ls -l /usr/local/lib/python3.5/site-packages/
```

For some reason, when compiling OpenCV with Python  3  support, the output ​ **cv2.so** filename is different. The actual filename might vary for you, but it should look something similar to **cv2.cpython-35m-x86_64-linux-gnu.so**. All you need to do is rename the file:

```
$​ cd /usr/local/lib/python3.5/site-packages/
$​ sudo mv cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
```

After renaming **cv2.cpython-35m-x86_64-linux-gnu.so** to **simply cv2.so**, we can sym-link our OpenCV bindings into the **cv** virtual environment for Python 3.5:

```
$​ cd ~/.virtualenvs/cv/lib/python3.5/site-packages/
$​ ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so
```

#### #6. Testing your OpenCV installation

To verify that your installation is working:

1. Open up a new terminal.
2. Execute the workon command to access the **cv** Python virtual environment.
3. Attempt to import the Python + OpenCV bindings.

Here is how to perform these steps:

Once OpenCV has been installed, you can delete both the opencv-3.4.1 and opencv_contrib-3.4.1 directories (along with their associated .zip files):

```
$​ cd ~
$​ rm -rf opencv-3.1.0 opencv_contrib-3.1.0 opencv.zip
opencv_contrib.zip
```
However, be careful when running this command! You should first make sure you have properly installed OpenCV on your system otherwise, you’ll need to restart the entire compile process!


### Object detection in video with deep learning and OpenCV

To build our deep learning-based real-time object detector with OpenCV we’ll need to:

1. Access our webcam/video stream in an efficient manner and
2. apply object detection to each frame.

To see how this is done, we open up a new file, name it **real_time_object_detection.py** and insert the following code:

```
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
```

We begin by importing our necessary packages. We will need **imutils** installed and OpenCV 3.3+. 
Next, we’ll parse our command line arguments:

```
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True,  help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pre-trained model")
ap.add_argument("-c", "--probability", type = float, default = 0.2, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

What are these?

● --prototxt: The path to the Caffe prototxt file (I’m using MobileNet as it is more
lightweight and simpler).
● --model: The path to the pre-trained model.
● --probability: The minimum probability threshold to filter weak detections. The
default is 20%.

We then initialize a class list and a color set:

```
# initialize the list of class labels MobileNet SSD was trained to detect
# and generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))
```

Here we initialize CLASS labels and corresponding random COLORS. Now, let’s load our model and set up our video stream:

```
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
```

We load our serialized model, providing the references to our prototxt and model files. Next, let’s initialize our video stream (this can be from a video file or a camera). First we start the VideoStream, then we wait for the camera to warm up, and finally we start the frames per second counter. The VideoStream and FPS classes are part of the imutils package.

Now, let’s loop over each and every frame (for speed purposes, you could skip frames):

```
# loop over the frames from the video stream
while True:
	# resize the video stream window at a maximum width of 500 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=500)

	# grab the frame dimensions and convert it to a blob
	# Binary Large Object = BLOB
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and get the detections
	net.setInput(blob)
	detections = net.forward()
```

First, we read a frame from the stream, followed by resizing it.

Since we will need the width and height later, we should grab these now on. This is followed by converting the frame to a blob with the dnn module. We set the blob as the input to our neural network and feed the input through the net which will give us our detections.

At this point, we have detected objects in the input frame. It is now time to look at the confidence values and determine if we should draw a box + label surrounding the object:

```

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the probability of the prediction
		probability = detections[0, 0, i, 2]

		# filter out weak detections by ensuring that probability is
		# greater than the min probability
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

```

We start by looping over our detections , keeping in mind that multiple objects can be detected in a single image. We also apply a check to the confidence (i.e., probability) associated with each detection. If the confidence is high enough (above our set threshold), then we’ll display the prediction in the terminal as well as draw the
prediction on the image with text and a colored bounding box.

Let’s break it down line-by-line:

1. **Looping** through our detections, first we extract the **confidence value**.
2. If the **confidence** is above our **minimum threshold**, we extract the class label index and compute the bounding box coordinates around the detected object.
3. Then, we extract the **(x, y)-coordinates** of the box which we will will use for drawing a rectangle and displaying the label text.
4. We build a text **label** containing the **CLASS** name and the **confidence.**
5. We also **draw** a colored rectangle around the object using our class color and previously extracted (x, y)-coordinates.

In general, we want the label to be displayed above the rectangle, but if there isn’t room, we’ll display it just below the top of the rectangle.

Finally, we overlay the colored text onto the frame using the y-value that we just calculated.

The remaining steps in the frame capture loop involve:

1. displaying the frame,
2. checking for a quit key, and
3. updating our frames per second counter

```
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()
```

The above code block is pretty self-explanatory - first we display the frame. Then we capture a key press while checking if the “q” key (for “quit”) is pressed, at which point we break out of the frame capture loop. Finally, we update our FPS counter.

If we break out of the loop (“q” key press or end of the video stream), we have some tasks to take care of:

```
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()
```

When we’ve exited the loop, we stop the FPS counter and print information about the frames/second to our terminal.
We close the open window followed by stopping the video stream.

##### Real-time deep learning object detection results

To see our object detector in action, open up a terminal and execute the following command:

```
$​ python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

For those of you that use python 2.7 execute the command:

```
$​ python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

Provided that OpenCV can access your webcam you should see the output video frame with any detected objects. I have included a demonstration video in the project folder with the results of this application.

All in all, the end result is a deep learning-based object detector application that can process approximately 6-8 FPS (depending on the speed of our system, of course).

Further speed improvements can be obtained by:

➔ Applying skip frames.
➔ Swapping different variations of MobileNet (that are faster, but less accurate).


### Bibliography & sources

❏ https://docs.opencv.org/3.4.1/
❏ https://docs.opencv.org/3.4.1/d9/df8/tutorial_root.html
❏ https://github.com/jrosebr1/imutils
❏ https://github.com/jrosebr1/imutils/issues/
❏ https://www.shaileshjha.com/how-to-install-ubuntu-16-04-1-lts-and-vmware-tools-in-vmware-workstation-12-pro/
❏ https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html
❏ https://realpython.com/tutorials/computer-vision/
❏ http://www.swarthmore.edu/NatSci/mzucker1/opencv-2.4.10-docs/doc/tutorials/tutorials.html
❏ https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/

