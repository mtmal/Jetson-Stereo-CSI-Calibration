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

#ifndef __STEREO_LISTENER_H__
#define __STEREO_LISTENER_H__

#include <CSI_StereoCamera.h>

/**
 * A simple listener class that saves images and allows them to be copied over by another thread.
 */
class StereoListener : public IGenericListener<CameraData, CameraData>
{
public:
    /**
     * Basic constructor.
     *  @param stereoCam a reference to the stereo camera class.
     */
    explicit StereoListener(CSI_StereoCamera& stereoCam);

    /**
     * Basic destructor.
     */
    virtual ~StereoListener();

    /**
     * Initialises insternal buffers for images. To be called then this listener is not registered.
     *  @param imageSize the size of expected images.
     *  @param colour true to initialise three channels, otherwise a single channel buffers will be allocated.
     */
    void initialise(const cv::Size& imageSize, const bool colour);

    /**
     * Registers itself as a listener to the stereo camera class.
     */
    inline void registerListener()
    {
        mId = mStereoCam.registerListener(*this);
    }

    /**
     * Unregisters itself from the stereo camera class.
     */
    inline void unregisterListener()
    {
        mStereoCam.unregisterListener(mId);
    }

    // override
    void update(const CameraData& left, const CameraData& right) override;

    /**
     * Returns received images.
     *  @param[out] left an image from the left camera.
     *  @param[out] right an image from the right camera.
     */
    void getImages(cv::Mat& left, cv::Mat& right) const;

private:
    /** Reference to the stereo camera class. */
    CSI_StereoCamera& mStereoCam;
    /** ID received from the stereo camera class upon registration. */
    int mId;
    /** The left camera image. */
    mutable cv::Mat mLeft;
    /** The right camera image. */
    mutable cv::Mat mRight;
    /** Mutex for accessing images. */
    mutable pthread_mutex_t mLock;
};

#endif // __STEREO_LISTENER_H__
