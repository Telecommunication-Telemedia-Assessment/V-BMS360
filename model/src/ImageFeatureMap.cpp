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


#include "ImageFeatureMap.h"

ImageFeatureMap::ImageFeatureMap() {
    m_bms = boost::shared_ptr<BMSSaliency>(new BMSSaliency(true, m_ocl));

    m_bms->m_maxDim = 2000; // was 2000
    m_bms->m_dilatationWidth1 = static_cast<int>(fmax(round(7 * m_bms->m_maxDim / 400.f), 1.f));
    m_bms->m_dilatationWidth2 = static_cast<int>(fmax(round(9 * m_bms->m_maxDim / 400.f), 1.f));
    m_bms->m_blurStd = round(9 * m_bms->m_maxDim / 400);
    
}

cv::Mat ImageFeatureMap::compute(int frame) {

    grabRequiredData(frame);

    if(m_frame.color.empty()) return cv::Mat();

    cv::Mat master_map;
    m_bms->process(m_frame.color, master_map, true);

    return master_map;
}


void ImageFeatureMap::grabRequiredData(int frame) {
    m_frame = FlowManager::get()->getFrame(frame);
}

cv::Mat ImageFeatureMap::getColor(int frame) {
    return FlowManager::get()->getFrame(frame).color;
}


