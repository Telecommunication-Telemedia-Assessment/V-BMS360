// **************************************************************************************************
//
// The MIT License (MIT)
// 
// Copyright (c) 2017 Pierre Lebreton
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
// associated documentation files (the "Software"), to deal in the Software without restriction, including 
// without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the 
// following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial 
// portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// **************************************************************************************************



#ifndef _ObjectMotionFeatureMap_
#define _ObjectMotionFeatureMap_

#include <opencv2/core.hpp>
#include "MotionFeatureMap.h"
#include "BMSSaliency.h"
#include <opencv2/video/background_segm.hpp>


class ObjectMotionFeatureMap: public MotionFeatureMap {

private:
    cv::Ptr<cv::BackgroundSubtractorKNN> m_bgsubtractor;
    boost::shared_ptr<BMSSaliency>       m_bms;

public:
    ObjectMotionFeatureMap();
    virtual ~ObjectMotionFeatureMap() {};



    virtual cv::Mat compute     (int frame);

};




#endif
