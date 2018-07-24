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

#include "FlowGrabber.h"
#include "FlowIO.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

FlowManager  *FlowManager::m_This = NULL;


FlowManager *FlowManager::get() {
    if(m_This == NULL) m_This = new FlowManager();

    return m_This;
}


Flow FlowManager::getFrame(int frame) {
    for(std::list<Flow>::iterator it = m_cache.begin() ; it != m_cache.end() ; ++it) {
        if(it->frameNumber == frame) return *it;
    }

    if(!m_grabber) return Flow();


    m_cache.push_back(m_grabber->getFrame(frame));
    if(m_cache.size() > m_cacheSize) {
        m_cache.pop_front();
    }

    return m_cache.back();

}

float FlowManager::getFrameRate() {
    if(!m_grabber) return 30.0;

    return m_grabber->getFrameRate();
}

cv::Size FlowManager::getSourceFrameSize() {
	return m_grabber->getSourceFrameSize();
}



Flow FileFlowGrabber::getFrame(int frame) {    
	Flow flow;

	flow.frameNumber = frame;
	if(frame >= 0 && frame < static_cast<int>(m_filelist.size()))
		flow.frame = readFlow(m_filelist[frame]);

    if(!m_colorImagePath.empty())
        flow.color = cv::imread(m_colorImagePath);

    return flow;
}

cv::Size FileFlowGrabber::getSourceFrameSize() {
	return getFrame(0).frame.size();
}


VideoFlowGrabber::VideoFlowGrabber(const std::string& filename) {

    #ifdef GPU_MODE
        m_compute = cv::cuda::OpticalFlowDual_TVL1::create();
    #else
        // createOptFlow_DeepFlow() / createOptFlow_SimpleFlow() / createOptFlow_Farneback() // createOptFlow_SparseToDense // createVariationalFlowRefinement / createOptFlow_DIS

        m_compute = cv::optflow::createOptFlow_DIS();
    #endif


    m_capture.open(filename);
    if(!m_capture.isOpened()) {
        std::cerr << "cannot open: " << filename << std::endl;
        return ;
    } 

    m_curFrame = 0;
    m_scalingFactor = 2; // 2

}


int VideoFlowGrabber::getFrameCount() {
    if(!m_capture.isOpened()) {
		std::cerr << "[I] VideoFlowGrabber::getFrameCount: No file was open, cannot get the number of frames! " << std::endl;
        return -1;
    }

    return m_capture.get(cv::CAP_PROP_FRAME_COUNT);
}


float VideoFlowGrabber::getFrameRate() {
    if(!m_capture.isOpened()) {
        std::cerr << "[I] VideoFlowGrabber::getFrameRate: No file was open, cannot get a frame rate! " << std::endl;
        return -1;
    }

    return m_capture.get(cv::CAP_PROP_FPS);
}


cv::Size VideoFlowGrabber::getSourceFrameSize() {
	return cv::Size(m_capture.get(cv::CAP_PROP_FRAME_WIDTH), m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
}

Flow VideoFlowGrabber::getFrame(int frame) {

    cv::Mat frame2;
    cv::Mat flow;
    cv::Mat colorFrame;

    #ifdef GPU_MODE
        cv::cuda::GpuMat gpuframe2;
        cv::cuda::GpuMat gpuflow;
    #endif

	// if file not opened, cannot compute flow
    if(!m_capture.isOpened()) {
    	Flow res;
    	res.frameNumber = frame;
    	return res;	
    }


    if(frame == 0) ++frame;

    bool cached = true;
    for(int i = m_curFrame ; i < frame ; ++i) {
        m_capture >> m_frame;
        ++m_curFrame;
        cached = false;
    }

    if(m_frame.empty()) {
        Flow res;
    	res.frameNumber = frame;
    	return res;	
        // m_capture.set(cv::CAP_PROP_POS_FRAMES, m_capture.get(cv::CAP_PROP_FRAME_COUNT)-20);
        // m_capture >> m_frame;
    }


    if(!cached) {
        cv::resize(m_frame, m_frame, m_frame.size() / m_scalingFactor);
        cv::cvtColor(m_frame, m_frame, cv::COLOR_BGR2GRAY);

        #ifdef GPU_MODE
            m_gpuframe.upload(m_frame);
        #endif
    }

    m_capture >> frame2;
    ++m_curFrame;
    if(frame2.empty()) {
    	Flow res;
    	res.frameNumber = frame;
    	return res;	
    }


    cv::resize(frame2, colorFrame, frame2.size()/ m_scalingFactor);
    cv::cvtColor(colorFrame, frame2, cv::COLOR_BGR2GRAY);

    #ifdef GPU_MODE
        gpuframe2.upload(frame2);
        m_compute->calc(m_gpuframe, gpuframe2, gpuflow);
        gpuflow.download(flow); 
        std::swap(gpuframe2, m_gpuframe);
    #else
        m_compute->calc(m_frame, frame2, flow); 
        // flow = cv::Mat(frame2.size(), CV_32FC2, cv::Scalar(0,0));
        std::swap(frame2, m_frame);
    #endif

    
    

    Flow res;
    res.frameNumber = frame;
    res.color = colorFrame;
    res.frame = flow.clone();

    return res;
    
}
