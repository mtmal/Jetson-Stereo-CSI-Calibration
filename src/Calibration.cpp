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

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include "Calibration.h"

namespace
{
/** The RMS threshold for a single camera calibration. */
static const double RMS_THRESHOLD = 0.5;

/**
 * Calculates lines using image points from stereo calibration.
 *  @param data the data structure for individual camera.
 *  @param imgPoints the image points from stereo pairs.
 *  @param F the fundamental matrix.
 *  @param whichCam an index of the camera for which epilines are to be calculated.
 *  @param[out] lines computed lines.
 */
void calculateLines(const SingleCamDataStruct& data, const cv::Mat& imgPoints, const cv::Mat& F, const int whichCam,
                    std::vector<cv::Vec3f>& lines)
{
    undistortPoints(imgPoints, imgPoints, data.mCameraMatrix, data.mDist, data.mRectification, data.mNewCameraMatrix);
    computeCorrespondEpilines(imgPoints, whichCam, F, lines);
}

/**
 * Checks the quality of stereo calibration.
 *  @param stereo the data with information about stereo calibration.
 */
void calibrationQualityCheck(const StereoCamDataStruct& stereo)
{
    long unsigned int npoints = 0;
    long unsigned int i, j;
    double err = 0.0;
    double tempError;
    std::vector<cv::Vec3f> lines[2];

    // CALIBRATION QUALITY CHECK - as from OpenCV sample
    for (i = 0; i < stereo.mLCam.mStereoImgPoints.size(); ++i)
    {
    	std::vector<cv::Point2f> lImgPoints(stereo.mLCam.mStereoImgPoints[i].size());
    	std::vector<cv::Point2f> rImgPoints(stereo.mRCam.mStereoImgPoints[i].size());

    	std::copy(stereo.mLCam.mStereoImgPoints[i].begin(), stereo.mLCam.mStereoImgPoints[i].end(), lImgPoints.begin());
    	std::copy(stereo.mRCam.mStereoImgPoints[i].begin(), stereo.mRCam.mStereoImgPoints[i].end(), rImgPoints.begin());

    	tempError = 0.0;
        calculateLines(stereo.mLCam, cv::Mat(lImgPoints), stereo.mFundamental, 1, lines[0]);
        calculateLines(stereo.mRCam, cv::Mat(rImgPoints), stereo.mFundamental, 2, lines[1]);

        for (j = 0; j < stereo.mLCam.mStereoImgPoints[i].size(); ++j)
        {
        	tempError += fabs(lImgPoints[j].x * lines[1][j][0] + lImgPoints[j].y * lines[1][j][1] + lines[1][j][2]) +
						 fabs(rImgPoints[j].x * lines[0][j][0] + rImgPoints[j].y * lines[0][j][1] + lines[0][j][2]);
        }
        printf("Average error in frame %lu: %f \n", i, tempError / stereo.mLCam.mStereoImgPoints[i].size());
        err += tempError;
        npoints += stereo.mLCam.mStereoImgPoints[i].size();
    }
    printf("average epipolar err = %f \n", err / npoints);
}
} /* end of the anonymous namespace */

Calibration::Calibration(const cv::Size& imageSize,const cv::Size& boardSize, const int windowSize, const float squareSize)
: mImageSize(imageSize), mBoardSize(boardSize), mWindowSize(windowSize, windowSize), mSquareSize(squareSize)
{
}

Calibration::~Calibration()
{
}

bool Calibration::findChessCorners(SingleCamDataStruct& data) const
{
	static const cv::Size DEFAULT_SIZE(-1, -1);
	static const cv::TermCriteria DEFAULT_CRITERIA_CORNERS(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 1e-4);

    bool found = findChessboardCorners(data.mGreyImg, mBoardSize, data.mRawPoints,
    		cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_ADAPTIVE_THRESH);
    if (found)
    {
        cv::cornerSubPix(data.mGreyImg, data.mRawPoints, mWindowSize, DEFAULT_SIZE, DEFAULT_CRITERIA_CORNERS);
        cv::drawChessboardCorners(data.mColImg, mBoardSize, cv::Mat(data.mRawPoints), found);
    }
    return found;
}

void Calibration::calibrateSingleCamera(const std::string& folder, SingleCamDataStruct& data) const
{
	bool test = true;
    double rms;
    long int i;
    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    cv::FileStorage fs(folder + "/" + data.mName + CALIB_FILE_EXTENSION, cv::FileStorage::WRITE);

    calcChessboardCorners(objectPoints[0]);
    while (test && !data.mSingleImgPoints.empty())
    {
        objectPoints[0][mBoardSize.width - 1].x = objectPoints[0][0].x + mSquareSize * (mBoardSize.width - 1);
        objectPoints.resize(data.mSingleImgPoints.size(), objectPoints[0]);
    	test = false;
		rms = calibrateCamera(objectPoints, data.mSingleImgPoints, mImageSize, data.mCameraMatrix,
							  data.mDist, rvecs, tvecs, data.mStdDevIntrinsics, data.mStdDevExtrinsics, data.mPerViewErrors,
							  cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_RATIONAL_MODEL);
		printf("RMS error reported by calibrateCamera: %g\n", rms);

        if (rms > RMS_THRESHOLD)
        {
		    // we check now RMS for individual images. If any exceeds threshold, remove it and repeat calibration.
		    for (i = data.mSingleImgPoints.size() - 1; i >= 0; --i)
		    {
			    if (data.mPerViewErrors.at<double>(i) > RMS_THRESHOLD)
			    {
				    data.mSingleImgPoints.erase(data.mSingleImgPoints.begin() + i);
				    test = true;
			    }
		    }
        }
    }

    data.mCameraMatrix.copyTo(data.mNewCameraMatrix);

    std::cout << CAMERA_MATRIX << ": " << data.mCameraMatrix << "\n";
    std::cout << DISTORTION << ": " << data.mDist << "\n";
    std::cout << "Per View Errors: " << data.mPerViewErrors << "\n";
    if (fs.isOpened())
    {
        fs << CAMERA_MATRIX << data.mCameraMatrix;
        fs << DISTORTION 	<< data.mDist;
        fs.release();
    }
    else
    {
        std::cout << "Failed to open file: " << folder << "/" << data.mName << CALIB_FILE_EXTENSION << "\n";
    }
}

void Calibration::calibrateStereoCamera(const std::string& folder, StereoCamDataStruct& stereo) const
{
	static const cv::TermCriteria DEFAULT_CRITERIA_STEREO(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-5);

	bool test = true;
    double rms;
    long int i;
    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    cv::FileStorage fs(folder + "/" + STEREO_CALIB_FILE + CALIB_FILE_EXTENSION, cv::FileStorage::WRITE);

    calcChessboardCorners(objectPoints[0]);
    puts("Running stereoCalibrate");
    while (test && !stereo.mLCam.mStereoImgPoints.empty() && !stereo.mRCam.mStereoImgPoints.empty())
    {
        objectPoints.resize(stereo.mLCam.mStereoImgPoints.size(), objectPoints[0]);
    	test = false;
		rms = stereoCalibrate(objectPoints, stereo.mLCam.mStereoImgPoints, stereo.mRCam.mStereoImgPoints,
						stereo.mLCam.mCameraMatrix, stereo.mLCam.mDist,
						stereo.mRCam.mCameraMatrix, stereo.mRCam.mDist,
						mImageSize, stereo.mRotation, stereo.mTranslation, stereo.mEssential, stereo.mFundamental,
						stereo.mPerViewErrors,
						cv::CALIB_FIX_INTRINSIC + cv::CALIB_USE_EXTRINSIC_GUESS + cv::CALIB_RATIONAL_MODEL,
						DEFAULT_CRITERIA_STEREO);
		printf("RMS error reported by stereoCalibrate: %g\n", rms);

        if (rms > RMS_THRESHOLD * 4)
        {
		    // we check now RMS for individual images. If any exceeds threshold, remove the pair and repeat calibration.
		    for (i = stereo.mLCam.mStereoImgPoints.size() - 1; i >= 0; --i)
		    {
			    // stereo calibration often gives higher RMS which is OK, but needs to be reflected in the test
			    if (stereo.mPerViewErrors.at<double>(i, 0) > RMS_THRESHOLD * 4 ||
				    stereo.mPerViewErrors.at<double>(i, 1) > RMS_THRESHOLD * 4)
			    {
				    stereo.mLCam.mStereoImgPoints.erase(stereo.mLCam.mStereoImgPoints.begin() + i);
				    stereo.mRCam.mStereoImgPoints.erase(stereo.mRCam.mStereoImgPoints.begin() + i);
				    test = true;
			    }
		    }
        }
    }

    std::cout << "Rotation Matrix: " << stereo.mRotation << "\n";
    std::cout << "Translation matrix: " << stereo.mTranslation << "\n";
    std::cout << "Per view errors: " << stereo.mPerViewErrors << "\n";

    puts("Checking calibration quality 1");
    calibrationQualityCheck(stereo);

    puts("Running stereoRectify");
    stereoRectify(stereo.mLCam.mCameraMatrix, stereo.mLCam.mDist,
                  stereo.mRCam.mCameraMatrix, stereo.mRCam.mDist,
                  mImageSize, stereo.mRotation, stereo.mTranslation, 
                  stereo.mLCam.mRectification, stereo.mRCam.mRectification, 
                  stereo.mLCam.mNewCameraMatrix, stereo.mRCam.mNewCameraMatrix, stereo.mReProjection,
                  cv::CALIB_ZERO_DISPARITY, 0);

    std::cout << "Left:  new camera matrix: " << stereo.mLCam.mNewCameraMatrix << "\n";
    std::cout << "Left:  new dist matrix: " << stereo.mLCam.mDist << "\n";
    std::cout << "Right: new camera matrix: " << stereo.mRCam.mNewCameraMatrix << "\n";
    std::cout << "Right: new dist matrix: " << stereo.mRCam.mDist << "\n";

    if (fs.isOpened())
    {
        fs << ROTATION             << stereo.mRotation
           << TRANSLATION          << stereo.mTranslation
           << RECTIFICATION_LEFT   << stereo.mLCam.mRectification
           << RECTIFICATION_RIGHT  << stereo.mRCam.mRectification
           << NEW_CAM_MATRIX_LEFT  << stereo.mLCam.mNewCameraMatrix
           << NEW_CAM_MATRIX_RIGHT << stereo.mRCam.mNewCameraMatrix
           << DISPARITY_TO_DEPTH   << stereo.mReProjection;
        fs.release();
    }
    else
    {
        std::cout << "Failed to open file: " << folder << "/" << STEREO_CALIB_FILE << CALIB_FILE_EXTENSION << "\n";
    }

    puts("Checking calibration quality 2");
    // that should give better results as we use new camera matrix and rectification rotation, but it does not. why?
    calibrationQualityCheck(stereo);
}

void Calibration::calcChessboardCorners(std::vector<cv::Point3f>& corners) const
{
    int i, j;
    corners.resize(0);
    for (i = 0; i < mBoardSize.height; ++i)
    {
        for (j = 0; j < mBoardSize.width; ++j)
        {
            corners.push_back(cv::Point3f(j * mSquareSize, i * mSquareSize, 0));
        }
    }
}
