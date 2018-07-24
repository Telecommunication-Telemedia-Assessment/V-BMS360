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



#include "ObjectMotionFeatureMap.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


ObjectMotionFeatureMap::ObjectMotionFeatureMap() : MotionFeatureMap(1) {

    m_bms = boost::shared_ptr<BMSSaliency>(new BMSSaliency(true, m_ocl));

    m_bms->m_maxDim = 2000;
    m_bms->m_dilatationWidth1 = static_cast<int>(fmax(round(7 * m_bms->m_maxDim / 400.f), 1.f));
    m_bms->m_dilatationWidth2 = static_cast<int>(fmax(round(9 * m_bms->m_maxDim / 400.f), 1.f));
    m_bms->m_blurStd = round(9 * m_bms->m_maxDim / 400);

}

cv::Mat ObjectMotionFeatureMap::compute(int frame) {

    if(m_optFlow.empty()) 
        grabRequiredData(frame);

    if(m_optFlow.empty()) return cv::Mat();

    cv::Mat motionMap = m_optFlow[0].frame; 

    if(motionMap.empty()) return cv::Mat();

    motionMap *= 15;
    cv::Mat colMotion(motionMap.size(), CV_8UC3, cv::Scalar(0,0,0));
    for(int i = 0 ; i < motionMap.rows ; ++i) {
        float c = std::cos(3.1415926535898f * static_cast<float>(motionMap.rows / 2 - i) / motionMap.rows );

        for(int j = 0 ; j < motionMap.cols ; ++j) {
            cv::Point3_<unsigned char> &dP = colMotion.at< cv::Point3_<unsigned char> >(i,j);
            cv::Point2f &sP = motionMap.at< cv::Point2f >(i,j);
            
            dP.x = static_cast<unsigned char>( std::max(0.f, std::min(255.f, sP.x*c)));
            dP.y = static_cast<unsigned char>( std::max(0.f, std::min(255.f, sP.y*c)));
            
        }
    }

    cv::Mat master_map;
    m_bms->process(colMotion, master_map, true);

    return master_map;
}

