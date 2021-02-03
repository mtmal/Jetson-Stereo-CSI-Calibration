# Jetson-Stereo-CSI-Calibration

## General Info
A tool to perform stereo CSI camera calibration tested on Jetson Nano. This application uses the [Stereo-CSI-Camera](https://github.com/mtmal/Jetson-Stereo-CSI-Camera) which is loaded as git submodule via cmake. The code has been tested using Waveshare's [IMX219-83 Stereo Camera](https://www.waveshare.com/wiki/IMX219-83_Stereo_Camera) and Nvidia Jetson Nano with JetPack 4.4.1.

## Requirements
Same as for [Stereo-CSI-Camera](https://github.com/mtmal/Jetson-Stereo-CSI-Camera/blob/main/README.md#requirements) as it needs to compile.

## Setup
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j 3
$ ./CSI_Camera_Calibration -h
```
## Usage
This is a typical application for stereo camera calibration. I used default OpenCV checkerboard displayed on mobile (Samsung A70 which offers fairly good size with individual checkerboard square size of 8 mm). It comes with some caveats but they are covered in [Discussion](#discussion). <br>
The following keys are used:
* Space - acquires image(s) for the current step
* Enter - finishes the current step and proceeds to the next one
* Escape - exits the current step and application

## Calibration Approach
1. To get the best calibration result, the tool acquires the highest resolution from cameras (mode 0 - hard coded). Any calibration parameters can be later scaled to lower resolution, but if you were to scale up, you could potentially loose some information. For convenience, images are resized to 640x480 for display only.
2. Because images are at their highest resolution (8 MPx in case of IMX219) and OpenCV does not provide CUDA-enabled tools for checkerboard detection, it is not possible to analyse each frame in a timely manner to provide a live detection feedback. Therefore,you just need to make sure you can see the checkerboard and hit space. Image will be saved and analysed after the acquisition stage is finished.
3. When capturing images for stereo camera calibration, individual left and right camera images will be also used for single camera calibration. Because those images will have checker board around their centre of inside (i.e. right edge of the left camera and left edge of the right camera), the focus during individual camera acquisition step should be put on the other edges.
4. When all images are acquired, the tool will begin the search for checkerboard. Please note that with default settings this may take a while. If checker board is detected and sub-pixel refinment is performed with quite large window (33). I found it working quite well in my setup (more on that in [Discussion](#discussion)).
5. The next step is the actual camera calibration. RMS is one of the indicators of how good the calibration is. For single camera it should be around 0.3-0.5. Sometimes there might be one image which has high error and could make the entire calibration wrong. To mitigate that, after each calibration I analyse RMS from individual images/pairs. In the case of single camera, all images which RMS is above 0.5 (hard-coded) will be removed and the calibration will be repeated. It is do-while loop so there might be few iterations. In case of stereo calibration the threshold is 2.0 as typically it is bigger than for single camera. Another check is the epipolar error. I calculate it twice, with default values as in any OpenCV example, and with estimated new camera matrix and rectification rotation. Now, I would expect the second test to provide lower value (we provide estimated rectification information), yet it is higher. I am not sure why and I would welcome any explanation (I might be doing something wrong!)
6. After calibration is done, results are saved in XML files and loaded by CSI_Camera. You can then inspect rectified images in a live view.

## Discussion
I have not tried using checkerboard printed on a rigit paper/board yet. It is important that the board do not bend, hence I decided to use a mobile. However, I found that corners on images, after zoomin in, where not exact! Basically, you could see that black squares do not corder with their corners. It is all fine when visually inspecting checkerboard on mobile's screen, but not on images taken by IMX219. It might be artefact of taking images of a screen, or just a quality of the camera. This is the reason I used large window (33) for sub-pixel refinment, as it was able to better estimate the corner position. Too large window, however, was misaligning the corner point so it all boils down to finding the sweet spot.
