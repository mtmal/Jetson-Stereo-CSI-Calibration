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

#include <cmath>
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "Calibration.h"

namespace
{
/** The RMS threshold for a single camera calibration. */
static const double RMS_THRESHOLD = 0.5;

/**
 * Calculates lines using image points from stereo calibration.
 *  @param data a data structure for individual camera.
 *  @param imgPoints the image points from stereo pairs.
 *  @param F the fundamental matrix.
 *  @param whichCam an index of the camera for which epilines are to be calculated.
 *  @param[out] lines computed lines.
 */
void calculateLines(const SingleCamDataStruct& data, const std::vector<cv::Point2f>& imgPoints, const cv::Mat& F, const int whichCam,
                    std::vector<cv::Vec3f>& lines, std::vector<cv::Point2f>& undistPt)
{
    undistortPoints(imgPoints, undistPt, data.mCameraMatrix, data.mDist, data.mRectification, data.mNewCameraMatrix);
    computeCorrespondEpilines(cv::Mat(undistPt), whichCam, F, lines);
}

/**
 * Calculates a distance between a point and a line.
 *  @param pt a 2D point.
 *  @param vec a line with A, B, and C components.
 *  @return a distance of the point to the line.
 */
double distance(const cv::Point2f& pt, const cv::Vec3f& vec)
{
    return fabs(vec[0] * pt.x + vec[1] * pt.y + vec[2]) / sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
}

/**
 * Puts all image points, which are stored in a per-image vector, into a single vector.
 *  @param data a data structure for individual camera.
 *  @param[out] imgPoints a list of all image points.
 */
void flattenImagePoints(const SingleCamDataStruct& data, std::vector<cv::Point2f>& imgPoints)
{
    long unsigned int i;
    std::vector<cv::Point2f> temp;

    imgPoints.clear();
    for (i = 0; i < data.mStereoImgPoints.size(); ++i)
    {
        temp.clear();
        undistortPoints(data.mStereoImgPoints[i], temp, data.mCameraMatrix, data.mDist, data.mRectification, data.mNewCameraMatrix);
        imgPoints.insert(imgPoints.end(), temp.begin(), temp.end());
    }
}

/**
 * Finds a fundamental matrix using undistorted image points of a stereo pair.
 *  @param stereo a data structure holding all stereo camera image points.
 *  @return fundamental matrix
 */
cv::Mat recomputeFundamentalMat(const StereoCamDataStruct& stereo)
{
	std::vector<cv::Point2f> lImgPoints;
	std::vector<cv::Point2f> rImgPoints;

    flattenImagePoints(stereo.mLCam, lImgPoints);
    flattenImagePoints(stereo.mRCam, rImgPoints);

    return cv::findFundamentalMat(lImgPoints, rImgPoints, cv::FM_RANSAC, 3, 0.99, 1000);
}

/**
 * Checks epipolar lines of a stereo camera using calibration outputs.
 *  @param stereo a data structure holding all stereo camera image points.
 *  @param folder a path to a folder where images with epilines should be saved.
 *  @param size the size of rectified images. 
 */
void checkEpipolarLines(const StereoCamDataStruct& stereo, const std::string& folder, const cv::Size& size)
{
    long unsigned int npoints = 0;
    long unsigned int i, j;
    std::vector<cv::Point2f> pt1, pt2;
    std::vector<cv::Vec3f> lines1, lines2;
    cv::Mat lImgRect, rImgRect, combined;
    cv::Mat lMap1, lMap2, rMap1, rMap2;
    std::string file;
    double err = 0.0;
    double tempError;
    // Oddly, fundamental matrix provided by stereoCalibrate works well when R and P matrices are not prvided to undistortPoints.
    // A workaround is to recompute fundamental matrix using undistorted points, where R and P are provided.
    cv::Mat F = recomputeFundamentalMat(stereo);

    cv::initUndistortRectifyMap(stereo.mLCam.mCameraMatrix, stereo.mLCam.mDist, stereo.mLCam.mRectification, 
        stereo.mLCam.mNewCameraMatrix, size, CV_32FC1, lMap1, lMap2);
    cv::initUndistortRectifyMap(stereo.mRCam.mCameraMatrix, stereo.mRCam.mDist, stereo.mRCam.mRectification, 
        stereo.mRCam.mNewCameraMatrix, size, CV_32FC1, rMap1, rMap2);

    for (i = 0; i < stereo.mLCam.mStereoImgPoints.size(); ++i)
    {
    	tempError = 0.0;
        pt1.clear();
        pt2.clear();
        lines1.clear();
        lines2.clear();

        file = stereo.mLCam.mImgPathsStereo[i];

        calculateLines(stereo.mLCam, stereo.mLCam.mStereoImgPoints[i], F, 1, lines1, pt1);
        calculateLines(stereo.mRCam, stereo.mRCam.mStereoImgPoints[i], F, 2, lines2, pt2);

        cv::remap(stereo.mLCam.mImagesStereo[i], lImgRect, lMap1, lMap2, cv::INTER_LINEAR);
        cv::remap(stereo.mRCam.mImagesStereo[i], rImgRect, rMap1, rMap2, cv::INTER_LINEAR);

        cv::cvtColor(lImgRect, lImgRect, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rImgRect, rImgRect, cv::COLOR_GRAY2BGR);
        
        for (j = 0; j < stereo.mLCam.mStereoImgPoints[i].size(); ++j)
        {
            cv::line(lImgRect, cv::Point(0, -lines2[j][2] / lines2[j][1]),
                     cv::Point(lImgRect.cols, -(lines2[j][2] + lines2[j][0] * lImgRect.cols) / lines2[j][1]),
                     cv::Scalar(255, 0, 0));
            cv::circle(lImgRect, pt1[j], 10, cv::Scalar(0, 255, 0), -1);
            cv::line(rImgRect, cv::Point(0, -lines1[j][2] / lines1[j][1]),
                     cv::Point(rImgRect.cols, -(lines1[j][2] + lines1[j][0] * rImgRect.cols) / lines1[j][1]),
                     cv::Scalar(0, 255, 0));
            cv::circle(rImgRect, pt2[j], 10, cv::Scalar(255, 0, 0), -1);

            tempError += distance(pt1[j], lines2[j]) + distance(pt2[j], lines1[j]);
        }
        printf("Average error in frame %lu: %f \n", i, tempError / stereo.mLCam.mStereoImgPoints[i].size());
        err += tempError;
        npoints += stereo.mLCam.mStereoImgPoints[i].size();
        cv::hconcat(lImgRect, rImgRect, combined);
        cv::imwrite(folder + file.substr(file.find_last_of("/") + 1), combined);
    }
    printf("Average epipolar err = %f \n", err / npoints);
}

} /* end of the anonymous namespace */

Calibration::Calibration(const cv::Size& imageSize,const cv::Size& boardSize, const int windowSize, const float squareSize)
: mImageSize(imageSize), mBoardSize(boardSize), mWindowSize(windowSize, windowSize), mSquareSize(squareSize), mMarkerSize(-1.0f),
  mCalibrationFlags(/*cv::CALIB_RATIONAL_MODEL*/), mRefinedStrategy(false), mDictionary(nullptr), mCharucoboard(nullptr), mBoard(nullptr)
{
}

Calibration::~Calibration()
{
}

void Calibration::initialise()
{
    /* ChArUco codes are used */
    if (isChArUco())
    {
        if (mMarkerSize < 0)
        {
            mMarkerSize = mSquareSize * 0.8f;
        }
        mCharucoboard = cv::aruco::CharucoBoard::create(mBoardSize.width, mBoardSize.height, mSquareSize, mMarkerSize, mDictionary);
        mBoard = mCharucoboard.staticCast<cv::aruco::Board>();
    }
}

bool Calibration::findCorners(SingleCamDataStruct& data) const
{
    data.mRawPoints.clear();
    data.mChArUcoIndices.clear();
    return isChArUco() ? findChArUcoCorners(data) : findChessCorners(data);
}

bool Calibration::findChessCorners(SingleCamDataStruct& data) const
{
	static const cv::Size DEFAULT_SIZE(-1, -1);
	static const cv::TermCriteria DEFAULT_CRITERIA_CORNERS(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 1e-4);

    bool found = findChessboardCorners(data.mGreyImg, mBoardSize, data.mRawPoints,
    		cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_ADAPTIVE_THRESH);
    if (found)
    {
        cv::cornerSubPix(data.mGreyImg, data.mRawPoints, mWindowSize, DEFAULT_SIZE, DEFAULT_CRITERIA_CORNERS);
        cv::drawChessboardCorners(data.mColImg, mBoardSize, cv::Mat(data.mRawPoints), found);
        data.mSingleObjectPoints.push_back(std::vector<cv::Point3f>());
        calcChessboardCorners(data.mSingleObjectPoints.back());
        data.mSingleImgPoints.push_back(data.mRawPoints);
    }
    return found;
}

bool Calibration::findChArUcoCorners(SingleCamDataStruct& data) const
{
    static const cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
    std::vector<std::vector<cv::Point2f> > corners;
    std::vector<std::vector<cv::Point2f> > rejected;
    cv::Mat currentCharucoCorners;
    cv::Mat currentCharucoIds;

    // detect markers
    cv::aruco::detectMarkers(data.mGreyImg, mDictionary, corners, data.mChArUcoIndices, detectorParams, rejected);

    // refind strategy to detect more markers
    if (mRefinedStrategy)
    {
        cv::aruco::refineDetectedMarkers(data.mGreyImg, mBoard, corners, data.mChArUcoIndices, rejected);
    }

    if (data.mChArUcoIndices.size() > 0)
    {
        // interpolate charuco corners
        cv::aruco::interpolateCornersCharuco(corners, data.mChArUcoIndices, data.mGreyImg, mCharucoboard, currentCharucoCorners,
                                         currentCharucoIds, data.mCameraMatrix);
        // draw markers and corners onto an image
        cv::aruco::drawDetectedMarkers(data.mColImg, corners, data.mChArUcoIndices);
        if (currentCharucoCorners.total() > 0)
        {
            cv::aruco::drawDetectedCornersCharuco(data.mColImg, currentCharucoCorners, currentCharucoIds);
        }

        // convert ChArUco corners and ids into image and object points for camera calibration
        data.mSingleObjectPoints.push_back(std::vector<cv::Point3f>());
        cv::aruco::getBoardObjectAndImagePoints(mBoard, corners, data.mChArUcoIndices, data.mSingleObjectPoints.back(), data.mRawPoints);
        data.mSingleImgPoints.push_back(data.mRawPoints);
    }

    return (data.mChArUcoIndices.size() > 0);
}

void Calibration::calibrateSingleCamera(const std::string& folder, SingleCamDataStruct& data) const
{
	bool test = true;
    double rms, error;
    long int i;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    std::vector<cv::Point3f> newObjPoints;
    cv::Mat stdDevObjPoints;

    cv::FileStorage fs(folder + "/" + data.mName + CALIB_FILE_EXTENSION, cv::FileStorage::WRITE);

    while (test && !data.mSingleImgPoints.empty())
    {
    	test = false;
        rvecs.clear();
        tvecs.clear();
        newObjPoints.clear();
		rms = calibrateCameraRO(data.mSingleObjectPoints, data.mSingleImgPoints, mImageSize, mBoardSize.width - 1, data.mCameraMatrix,
							    data.mDist, rvecs, tvecs, newObjPoints, data.mStdDevIntrinsics, data.mStdDevExtrinsics, 
                                stdDevObjPoints, data.mPerViewErrors, mCalibrationFlags | cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_FIX_ASPECT_RATIO);
		printf("Re-projection error reported by calibrateCameraRO: %g \n", rms);
        printf("Average reprojection error: %g \n", cv::mean(data.mPerViewErrors)[0]);

        if (rms > RMS_THRESHOLD)
        {
		    // we check now RMS for individual images. If any exceeds threshold, remove it and repeat calibration.
		    for (i = data.mSingleImgPoints.size() - 1; i >= 0; --i)
		    {
			    if (data.mPerViewErrors.at<double>(i) > RMS_THRESHOLD * 2)
			    {
				    data.mSingleImgPoints.erase(data.mSingleImgPoints.begin() + i);
                    data.mSingleObjectPoints.erase(data.mSingleObjectPoints.begin() + i);
				    test = true;
			    }
		    }
        }
    }

    data.mNewCameraMatrix = cv::getOptimalNewCameraMatrix(data.mCameraMatrix, data.mDist, mImageSize, 0);

    std::cout << CAMERA_MATRIX << ": " << data.mCameraMatrix << "\n";
    std::cout << DISTORTION << ": " << data.mDist << "\n";
    std::cout << "Per View Errors: " << data.mPerViewErrors << "\n";
    std::cout << "Optimal Camera Matrix for mono camera system: " << data.mNewCameraMatrix << "\n";
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
	static const cv::TermCriteria DEFAULT_CRITERIA_STEREO(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 500, 1e-6);

	bool test = true;
    double rms;
    long int i;
    int scale = 4; // 4
	cv::Mat temp1, temp2;
    cv::FileStorage fs(folder + "/" + STEREO_CALIB_FILE + CALIB_FILE_EXTENSION, cv::FileStorage::WRITE);

    puts("Running stereoCalibrate");
    while (test && !stereo.mLCam.mStereoImgPoints.empty() && !stereo.mRCam.mStereoImgPoints.empty())
    {
    	test = false;
        stereo.mLCam.mDist.copyTo(temp1);
        stereo.mRCam.mDist.copyTo(temp2);
        stereo.mLCam.mCameraMatrix.copyTo(stereo.mLCam.mNewCameraMatrix);
        stereo.mRCam.mCameraMatrix.copyTo(stereo.mRCam.mNewCameraMatrix);
        stereo.mLCam.mRectification = cv::Mat::eye(3, 3, CV_64F);
        stereo.mRCam.mRectification = cv::Mat::eye(3, 3, CV_64F);

        /** Provide a copy of distortion parameters because otherwise they will be overridden!
         *  CALIB_FIX_INTRINSIC somehow still modifies them. */
		rms = stereoCalibrate(stereo.mLCam.mStereoObjectPoints, stereo.mLCam.mStereoImgPoints, stereo.mRCam.mStereoImgPoints,
						stereo.mLCam.mCameraMatrix, stereo.mLCam.mDist,
						stereo.mRCam.mCameraMatrix, stereo.mRCam.mDist,
						mImageSize, stereo.mRotation, stereo.mTranslation, stereo.mEssential, stereo.mFundamental,
						stereo.mPerViewErrors,
						mCalibrationFlags | cv::CALIB_FIX_INTRINSIC | cv::CALIB_USE_EXTRINSIC_GUESS | cv::CALIB_FIX_ASPECT_RATIO,
						DEFAULT_CRITERIA_STEREO);
		printf("RMS error reported by stereoCalibrate: %g\n", rms);

        std::cout << "Rotation Matrix: " << stereo.mRotation << "\n";
        std::cout << "Translation matrix: " << stereo.mTranslation << "\n";
        std::cout << "Per view errors: " << stereo.mPerViewErrors << "\n";
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

        if (rms > RMS_THRESHOLD * 2)
        {
            printf("Stereo image points size: %lu \n", stereo.mLCam.mStereoImgPoints.size());
		    // we check now RMS for individual images. If any exceeds threshold, remove the pair and repeat calibration.
		    for (i = stereo.mLCam.mStereoImgPoints.size() - 1; i >= 0; --i)
		    {
			    // stereo calibration often gives higher RMS which is OK, but needs to be reflected in the test
			    if (stereo.mPerViewErrors.at<double>(i, 0) > RMS_THRESHOLD * scale ||
				    stereo.mPerViewErrors.at<double>(i, 1) > RMS_THRESHOLD * scale)
			    {
				    stereo.mLCam.mStereoImgPoints.erase(stereo.mLCam.mStereoImgPoints.begin() + i);
				    stereo.mRCam.mStereoImgPoints.erase(stereo.mRCam.mStereoImgPoints.begin() + i);
                    stereo.mLCam.mStereoObjectPoints.erase(stereo.mLCam.mStereoObjectPoints.begin() + i);
                    stereo.mRCam.mStereoObjectPoints.erase(stereo.mRCam.mStereoObjectPoints.begin() + i);
                    stereo.mLCam.mImagesStereo.erase(stereo.mLCam.mImagesStereo.begin() + i);
                    stereo.mRCam.mImagesStereo.erase(stereo.mRCam.mImagesStereo.begin() + i);
                    stereo.mLCam.mImgPathsStereo.erase(stereo.mLCam.mImgPathsStereo.begin() + i);
                    stereo.mRCam.mImgPathsStereo.erase(stereo.mRCam.mImgPathsStereo.begin() + i);
				    test = true;
			    }
		    }
        }
    }
    checkEpipolarLines(stereo, folder, mImageSize);

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
}

void Calibration::setZeroTangentialDist()
{
    mCalibrationFlags |= cv::CALIB_ZERO_TANGENT_DIST;
}

void Calibration::setChArUcoDictionary(const int dictionaryId)
{
    mDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
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
