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



#ifndef _AdaptiveMotionFeatureMap_
#define _AdaptiveMotionFeatureMap_

#include <opencv2/core.hpp>

#include "MotionFeatureMap.h"
#include "FlowClassifier.h"



class AdaptiveMotionFeatureMap: public MotionFeatureMap {

private:
    boost::shared_ptr<FlowClassier>      m_flowClassifier;
    bool                                 m_pedestrianDriven;
    cv::Mat                              m_lastMap;

public:
    AdaptiveMotionFeatureMap();
    virtual ~AdaptiveMotionFeatureMap() {};


    virtual cv::Mat compute             (int frame);

    inline void setPedestrianDriven     (bool enable)               { m_pedestrianDriven = enable; };	

private:

    std::vector<float> flowClassif(int frame);
    void getFeatureMapJob(int frame, int feature, cv::Mat& mat);

};




#endif
