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

#ifndef __DATASTRUCT_H__
#define __DATASTRUCT_H__

#include <opencv2/core/mat.hpp>
#include <CameraConstants.h>

/**
 * Data structure with all required buffers and lists for a single camera calibration.
 */
struct SingleCamDataStruct
{
    /** The name of this camera data structure. */
    std::string mName;
    /** The colour image as acquired from camera (after resize). */
    cv::Mat mColImg;
    /** The greyscale version of the coloured image. */
    cv::Mat mGreyImg;
    /** The resized image to display in canvas. */
    cv::Mat mDispImg;
    /** The camera matrix. */
    cv::Mat mCameraMatrix;
    /** The distortion parameters. */
    cv::Mat mDist;
    /** The rectification matrix R. */
    cv::Mat mRectification;
    /** The new camera matrix obtained after stereo rectification P. */
    cv::Mat mNewCameraMatrix;
    /** Output vector of standard deviations estimated for intrinsic parameters. */
    cv::Mat mStdDevIntrinsics;
    /** Output vector of standard deviations estimated for extrinsic parameters. */
    cv::Mat mStdDevExtrinsics;
    /** Per view errors for camera calibration. */
    cv::Mat mPerViewErrors;
    /** Image points for single camera calibration. */
    std::vector<std::vector<cv::Point2f> > mSingleImgPoints;
    /** Image points for stereo camera calibration. */
    std::vector<std::vector<cv::Point2f> > mStereoImgPoints;
    /** The list of image points for one image. */
    std::vector<cv::Point2f> mRawPoints;
    /** Object points for single camera calibration. */
    std::vector<std::vector<cv::Point3f> > mSingleObjectPoints;
    /** Object points for stereo camera calibration. */
    std::vector<std::vector<cv::Point3f> > mStereoObjectPoints;
    /** The list of ChArUco indices that were detected in an image. */
    std::vector<int> mChArUcoIndices;
    /** The list of all greyscale images. */
    std::vector<cv::Mat> mImagesStereo;
    /** The list of image paths. */
    std::vector<std::string> mImgPathsStereo;

    /**
     * Structure constructor initialises matrices with images size.
     *  @param name the name of this camera data structure.
     *  @param imgSize the size of images.
     *  @param dispSize the size of an image to display.
     *  @param focalLenth the focal length in metres for camera matrix initialisation.
     *  @param sensorWidth the width of sensor in metres for camera matrix initialisation.
     */
    SingleCamDataStruct(const std::string& name, const cv::Size& imgSize, const cv::Size& dispSize,
        const double focalLenth, const double sensorWidth) : mName(name), mColImg(imgSize, CV_8UC3), 
        mGreyImg(imgSize, CV_8UC1), mDispImg(dispSize, CV_8UC3), mCameraMatrix(cv::Mat::eye(3, 3, CV_64F))
    {
        mCameraMatrix.at<double>(0, 0) = focalLenth * imgSize.width / sensorWidth;
        mCameraMatrix.at<double>(1, 1) = mCameraMatrix.at<double>(0, 0);
        mCameraMatrix.at<double>(0, 2) = static_cast<double>(imgSize.width  / 2);
        mCameraMatrix.at<double>(1, 2) = static_cast<double>(imgSize.height / 2);
        mCameraMatrix.copyTo(mNewCameraMatrix);
    };
};

/**
 * Data structure with all required buffers and lists for a stereo camera calibration.
 */
struct StereoCamDataStruct
{
    /** The left camera. */
    SingleCamDataStruct mLCam;
    /** The right camera. */
    SingleCamDataStruct mRCam;
    /** The rotation matrix R. */
    cv::Mat mRotation;
    /** The translation matrix T. */
    cv::Mat mTranslation;
    /** The essential matrix E. */
    cv::Mat mEssential;
    /** The fundamental matrix F. */
    cv::Mat mFundamental;
    /** Per view errors for stereo calibration */
    cv::Mat mPerViewErrors;
    /** The re-projection matrix Q. */
    cv::Mat mReProjection;

    /**
     * Structure constructor initialises single camera's structures with images size.
     *  @param imgSize the size of images.
     *  @param dispSize the size of an image to display.
     *  @param focalLenth the focal length in metres for camera matrix initialisation.
     *  @param sensorWidth the width of sensor in metres for camera matrix initialisation.
     *  @param initialBaseline the initial baseline on X axis of stereo camera in the same units as checkerboard square size.
     */
    StereoCamDataStruct(const cv::Size& imgSize, const cv::Size& dispSize, const double focalLenth,
    		const double sensorWidth, const double initialBaseline) :
        mLCam(LEFT_CALIB_FILE,  imgSize, dispSize, focalLenth, sensorWidth),
        mRCam(RIGHT_CALIB_FILE, imgSize, dispSize, focalLenth, sensorWidth),
        mRotation(cv::Mat::eye(3, 3, CV_64F)), mTranslation(cv::Mat::zeros(3, 1, CV_64F))
    {
        mTranslation.at<double>(0) = initialBaseline;
    };
};

#endif // __DATASTRUCT_H__
