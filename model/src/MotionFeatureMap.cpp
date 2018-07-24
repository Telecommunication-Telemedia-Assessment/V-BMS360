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



#include "MotionFeatureMap.h"


void MotionFeatureMap::grabRequiredData(int frame) {
    
    if(!FlowManager::get()->getFlowGrabber()) return;

    std::vector<Flow> lFlow;
    // we need [frame frame+window]
    for(int i = frame ; i < frame + m_NbRequiredFrames ; ++i) {
        
        bool needCompute = true;
        for(size_t k = 0 ; k < m_optFlow.size() ; ++k) {
            if(m_optFlow[k].frameNumber == i) {
                needCompute = false;
                lFlow.push_back(m_optFlow[k]);
            }
        }

        if(needCompute) {
            Flow flow = FlowManager::get()->getFrame(i);
            if(!flow.frame.empty()) {
                lFlow.push_back(flow);
            }   
        }
    }
    
    m_optFlow = lFlow;
}


cv::Mat MotionFeatureMap::getFrontFlow() {
    
    if(m_optFlow.empty()) return cv::Mat();

    return m_optFlow.front().frame;
}

cv::Mat MotionFeatureMap::getColor(int frame) {
    return FlowManager::get()->getFrame(frame).color;
}


