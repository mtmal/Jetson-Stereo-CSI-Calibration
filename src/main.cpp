////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2021 Mateusz Malinowski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#include <regex>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <CSI_StereoCamera.h>
#include "Calibration.h"

/** The name of the window for left camera image. */
static const std::string LEFT_WINDOW_NAME  = "Left Cam Image";
/** The name of the window for right camera image. */
static const std::string RIGHT_WINDOW_NAME = "Right Cam Image";
/** The size to which images are resized only for display (calibration is done on dull size). */
static const cv::Size DISPLAY_SIZE(640, 480);
/** ASCII code for enter key. */
static const int ENTER_KEY = 13;
/** ASCII code for escape key. */
static const int ESCAPE_KEY = 27;
/** ASCII code for space key. */
static const int SPACE_KEY = 32;
/** Suffix used for left camera images. */
static const std::string LEFT_IMAGE_SUFFIX  = "_" + LEFT_CALIB_FILE  + ".png";
/** Suffix used for right camera images. */
static const std::string RIGHT_IMAGE_SUFFIX = "_" + RIGHT_CALIB_FILE + ".png";
/** Working folder where results of on-line calibration should be stored.  */
static std::string FOLDER_MAIN;
/** Foder where stereo pairs should be saved. */
static std::string FOLDER_STEREO;


// to restart CSI camera in system use command:
// $sudo systemctl restart nvargus-daemon

/**
 * Prints help information about this application.
 *  @param name the name of the executable.
 */
void printHelp(const char* name)
{
	printf("Usage: %s [options] \n", name);
	printf("    -h, --help     -> prints this information \n");
	printf("    -o, --offline  -> sets the path to the folder with previously captured images, default: capture images from camera. \n");
	printf("    -c, --cols     -> sets the number of columns (width) of the checkerboard, default: 6 \n");
	printf("    -r, --rows     -> sets the number of rows (height) of the checkerboard, default: 9 \n");
	printf("    -w, --window   -> sets the size of window for sub-pixel refinement, default: 33 \n");
	printf("    -s, --square   -> sets the checkerboard's square size in the same units as baseline, default: 0.008 \n");
	printf("    -b, --baseline -> sets the stereo camera baseline in X axis in the same units as the square, default: -0.06 \n");
	printf("\nExample: %s -o <path_to_images> \n\n", name);
	printf("NOTE: if the application that uses nvargus to control cameras was killed without releasing the cameras,"
			" execute the following:\n\n"
			"$ sudo systemctl restart nvargus-daemon \n\n");
}

/**
 *  @return the time of the system in seconds.
 */
double getTime()
{
    struct timespec timeStruct;
    clock_gettime(CLOCK_REALTIME, &timeStruct);
    return static_cast<double>(timeStruct.tv_sec) + static_cast<double>(timeStruct.tv_nsec / 1000000) / 1000;
}

/**
 * Updates global variables for folders with a new parent folder. For off-line calibration.
 *  @param main the new folder where sub-folders with files holding intrinsics and extrinsic are located.
 */
void populateFoldersNames(const std::string& main)
{
    FOLDER_MAIN   = main;
    FOLDER_STEREO = FOLDER_MAIN + "/" + STEREO_CALIB_FILE + "/";
}

/**
 * Analyses a single image to determine if the checker board can be found or not.
 *  @param calib the reference the calibration class.
 *  @param file the path to the image which needs to be loaded from a file.
 *  @param resultFolder the folder where where image with overlaid checkerboard corners should be saved (only if detected).
 *  @param[in/out] data a data holding all information related to a single camera calibration.
 *  @return true if the checkerboard was detected in the image located under @p file.
 */
bool analyseImg(const Calibration& calib, const std::string& file, const std::string& resultFolder, SingleCamDataStruct& data)
{
    bool retVal;
    data.mColImg = cv::imread(file); // we want coloured image as well
    cv::cvtColor(data.mColImg, data.mGreyImg, cv::COLOR_BGR2GRAY);
    data.mRawPoints.clear();
    retVal = calib.findChessCorners(data);
    if (retVal)
    {
        cv::imwrite(resultFolder + file.substr(file.find_last_of("/") + 1), data.mColImg);
    }
    else
    {
        printf("Failed to detect pattern in %s \n", file.c_str());
    }
    return retVal;
}

/**
 * Takes all stereo pairs from stereo folder and analyses them for a presence of a checker board.
 * It narrows the search to only left camera images and then replaces the suffix to build the path
 * to the right camera image. Then, both images are analysed. If checkerboard is detected on any
 * image, that image is added to relevant single camera calibration process. If checkerboard is
 * detected on both images, then the pair is added to stereo image calibration.
 *  @param calib the reference the calibration class.
 *  @param stereo a data holding all information related to a stereo camera calibration.
 */
void stereoCamFind(const Calibration& calib, StereoCamDataStruct& stereo)
{
    int counter = 0;
    bool rCamFlag, lCamFlag;
    std::string folderWithResults(FOLDER_STEREO + "checkerboard/");
    std::vector<std::string> results;

    mkdir(folderWithResults.c_str(), 0777);

    cv::glob(FOLDER_STEREO + "*" + LEFT_IMAGE_SUFFIX, results);
    for (const std::string& file : results)
    {
        lCamFlag = analyseImg(calib, file, folderWithResults, stereo.mLCam);
        rCamFlag = analyseImg(calib, std::regex_replace(file,
        		std::regex(LEFT_IMAGE_SUFFIX), RIGHT_IMAGE_SUFFIX), folderWithResults, stereo.mRCam);
        if (lCamFlag)
        {
            stereo.mLCam.mSingleImgPoints.push_back(stereo.mLCam.mRawPoints);
        }
        if (rCamFlag)
        {
            stereo.mRCam.mSingleImgPoints.push_back(stereo.mRCam.mRawPoints);
        }
        if (lCamFlag && rCamFlag)
        {
            printf("Image #%d \n", ++counter);
            stereo.mLCam.mStereoImgPoints.push_back(stereo.mLCam.mRawPoints);
            stereo.mRCam.mStereoImgPoints.push_back(stereo.mRCam.mRawPoints);
        }
    }
}

/**
 * Displays images in on-line calibration.
 *  @param stereoData the data structure with the latest images.
 */
void displayImages(StereoCamDataStruct& stereoData)
{
    cv::resize(stereoData.mLCam.mColImg, stereoData.mLCam.mDispImg, DISPLAY_SIZE);
    cv::resize(stereoData.mRCam.mColImg, stereoData.mRCam.mDispImg, DISPLAY_SIZE);
    cv::imshow(LEFT_WINDOW_NAME,  stereoData.mLCam.mDispImg);
    cv::imshow(RIGHT_WINDOW_NAME, stereoData.mRCam.mDispImg);
}

/**
 * Saves the images to a file.
 */
void saveImage(StereoCamDataStruct& stereoData)
{
    std::string timestamp = std::to_string(getTime());
    cv::imwrite(FOLDER_STEREO + timestamp + LEFT_IMAGE_SUFFIX,  stereoData.mLCam.mColImg);
    cv::imwrite(FOLDER_STEREO + timestamp + RIGHT_IMAGE_SUFFIX, stereoData.mRCam.mColImg);
}

char* parseInputs(int argc, char** argv, Calibration& calib, StereoCamDataStruct& stereoData)
{
	char* offlinePath = nullptr;
    for (int i = 1; i < argc; ++i)
    {
        if ((0 == strcmp(argv[i], "--offline")) || (0 == strcmp(argv[i], "-o")))
        {
        	offlinePath = argv[i + 1];
        }
        else if ((0 == strcmp(argv[i], "--cols")) || (0 == strcmp(argv[i], "-c")))
        {
        	calib.setBoardWidth(atoi(argv[i + 1]));
        }
        else if ((0 == strcmp(argv[i], "--rows")) || (0 == strcmp(argv[i], "-r")))
        {
        	calib.setBoardHeight(atoi(argv[i + 1]));
        }
        else if ((0 == strcmp(argv[i], "--window")) || (0 == strcmp(argv[i], "-w")))
        {
        	calib.setWindowSize(atoi(argv[i + 1]));
        }
        else if ((0 == strcmp(argv[i], "--square")) || (0 == strcmp(argv[i], "-s")))
        {
        	calib.setSquareSize(static_cast<float>(atof(argv[i + 1])));
        }
        else if ((0 == strcmp(argv[i], "--baseline")) || (0 == strcmp(argv[i], "-b")))
        {
        	stereoData.mTranslation.at<double>(0) = atof(argv[i + 1]);
        }
        else if ((0 == strcmp(argv[i], "--help")) || (0 == strcmp(argv[i], "-h")))
        {
        	printHelp(argv[0]);
        	exit(0);
        }
		else
		{
			/* nothing to do in here */
		}
    }
    return offlinePath;
}

int main(int argc, char** argv)
{
	/** Image size for CSI cameras. */
    cv::Size imageSize(640, 480);
    /** The mode in which CSI cameras should operate. */
    unsigned int mode = 0;
    /** The frequency at which CSI cameras should acquire images. */
    unsigned int framerate = CSI_Camera::getSizeForMode(mode, imageSize);
    /** The ASCII code of key pressed by the user. */
    int key = 0;
    /** The error code returned by this application. */
    int retVal = 0;
    /** The CSI stereo camera wrapper. */
    CSI_StereoCamera stereoCamera(imageSize);
    /** The structure that holds all information for stereo camera calibration. */
    StereoCamDataStruct stereoData(imageSize, DISPLAY_SIZE, CSI_Camera::FOCAL_LENGTH_M, CSI_Camera::SENSOR_WIDTH_M, -0.06);
    /** Class which provides all functionality for stereo camera calibration. */
    Calibration calib(imageSize, cv::Size(6, 9), 33, 0.008f);
    /* Parse any user inputs. */
    char* offlinePath = parseInputs(argc, argv, calib, stereoData);

    if (nullptr == offlinePath)
    {
    	/* We will be acquiring images of calibration board, therefore we need to create all folders first. */
        populateFoldersNames("./" + std::to_string(getTime()));
        mkdir(FOLDER_MAIN.c_str(),   0777);
        mkdir(FOLDER_STEREO.c_str(), 0777);

        cv::namedWindow(LEFT_WINDOW_NAME,  cv::WINDOW_AUTOSIZE);
        cv::namedWindow(RIGHT_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
        if (stereoCamera.startCamera(framerate, mode))
        {
        	/* The main loop which controls acquiring images from CSI stereo camera. */
            while (ESCAPE_KEY != key)
            {
                /** We need to take raw images for calibration. */
                if (stereoCamera.getRawImages(stereoData.mLCam.mColImg, stereoData.mRCam.mColImg))
                {
                	/* Display current images. */
                    displayImages(stereoData);
                    key = cv::waitKey(0) & 0xff;
                    /** If the user pressed space bar, save images to the file. We will analyse them later.
                     * Otherwise it takes too much time and may cause lags in display. Remember: we are doing
                     * the calibration in the highest possible resolution to get the best possible results. */
                    if (SPACE_KEY == key)
                    {
                        puts("Saving image");
                        saveImage(stereoData);
                    }
                }
                else
                {
                    puts("Failed to capture raw images");
                    key = cv::waitKey(30) & 0xff;
                }
            }
        }
        else
        {
            puts("Failed to open camera.");
            retVal = -1;
        }
    }
    else
    {
    	/* We want to load previously captured images so we need to update paths to folders accordingly. */
        populateFoldersNames(offlinePath);
        cv::namedWindow(LEFT_WINDOW_NAME,  cv::WINDOW_AUTOSIZE);
        cv::namedWindow(RIGHT_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    }

    /* Now that images were captured, or we supplied folder with existing images, we load images first and detect
     * checker board. At this stage we build our lists of image points. */
    puts("Identifying checkerboard.");
    stereoCamFind(calib, stereoData);

    /** If we failed to detect any image points, we simply skip calibration. Throw error and exit. */
    if (stereoData.mLCam.mSingleImgPoints.empty() || stereoData.mLCam.mStereoImgPoints.empty() ||
        stereoData.mRCam.mSingleImgPoints.empty() || stereoData.mRCam.mStereoImgPoints.empty())
    {
        puts("Failed to capture image points.");
        retVal = -1;
    }
    else
    {
        /* Now run the calibration on individual cameras and then stereo using already-calibrated intrinsics. */
        puts("calibrating left camera");
        calib.calibrateSingleCamera(FOLDER_MAIN, stereoData.mLCam);
        puts("calibrating right camera");
        calib.calibrateSingleCamera(FOLDER_MAIN, stereoData.mRCam);
        puts("calibrating stereo camera");
        calib.calibrateStereoCamera(FOLDER_MAIN, stereoData);

        /* Load the calibration information to CSI cameras. */
        if (stereoCamera.loadCalibration(FOLDER_MAIN))
        {
        	/* If we haven't started camera yet, do it now. */
        	if (!stereoCamera.isInitialised())
        	{
        		stereoCamera.startCamera(framerate, mode);
        	}

            key = 0;
            /* Now we perform visual inspection of calibration. Rectified greyscale images are displayed. */
            while (stereoCamera.isInitialised() && ESCAPE_KEY != key)
            {
                key = cv::waitKey(30) & 0xff;
                if (stereoCamera.getRectified(false, stereoData.mLCam.mGreyImg, stereoData.mRCam.mGreyImg))
                {
                    cv::resize(stereoData.mLCam.mGreyImg, stereoData.mLCam.mDispImg, DISPLAY_SIZE);
                    cv::resize(stereoData.mRCam.mGreyImg, stereoData.mRCam.mDispImg, DISPLAY_SIZE);
                    cv::imshow(LEFT_WINDOW_NAME,  stereoData.mLCam.mDispImg);
                    cv::imshow(RIGHT_WINDOW_NAME, stereoData.mRCam.mDispImg);
                }
            }
        }
        else
        {
            printf("Failed to load calibration data from folder: %s \n", FOLDER_MAIN.c_str());
        }
    }
    
    /** At the end, destroy all windows and exit cleanly. */
    cv::destroyAllWindows();
    return retVal;
}
