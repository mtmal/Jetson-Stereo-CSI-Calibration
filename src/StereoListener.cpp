////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2023 Mateusz Malinowski
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

#include "StereoListener.h"

StereoListener::StereoListener(CSI_StereoCamera& stereoCam)
: IGenericListener<CameraData>(),
  mStereoCam(stereoCam), 
  mId(0), 
  mLeft(), 
  mRight()
{
    pthread_mutex_init(&mLock, nullptr);
}

StereoListener::~StereoListener()
{
    pthread_mutex_destroy(&mLock);
}

void StereoListener::initialise(const cv::Size& imageSize, const bool colour)
{
    mLeft  = cv::Mat(imageSize, colour ? CV_8UC3 : CV_8UC1);
    mRight = cv::Mat(imageSize, colour ? CV_8UC3 : CV_8UC1);
}

void StereoListener::update(const CameraData& camData)
{
    ScopedLock lock(mLock);
    mLeft  = camData.mImage[0].createMatHeader();
    mRight = camData.mImage[1].createMatHeader();
}

void StereoListener::getImages(cv::Mat& left, cv::Mat& right) const
{
    ScopedLock lock(mLock);
    std::swap(mLeft,  left);
    std::swap(mRight, right);
}