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



#include "PedestrianDetectFeatureMap.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

PedestrianFeatureMap::PedestrianFeatureMap(PedestrianMode mode) {
    m_HOG.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    m_FaceCascadeEnabled = false;
    if(mode == ADD_FACE_DETECTOR) {
        if (m_face_cascade.load("./data/haarcascade_profileface.xml")) {
		    m_FaceCascadeEnabled = true;
	    } else {
            std::cerr << "[I] PedestrianFeatureMap::PedestrianFeatureMap(): Cannot find ./data/haarcascade_profileface.xml " << std::endl;
        }
    }
}

cv::Mat PedestrianFeatureMap::compute(int frame) {

    cv::Mat img = m_frame.color;

    std::vector<cv::Rect> found, found_filtered;
    std::vector<double> weights;
    m_HOG.detectMultiScale(img, found, weights, 0.7, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    for (size_t i = 0 ; i < found.size() ; i++) {
        cv::Rect r = found[i];

        found_filtered.push_back(r);
    }


    cv::Mat smap(img.size(), CV_32FC1, cv::Scalar(0.f));


    for (size_t i = 0 ; i < found_filtered.size() ; i++) {
        cv::Rect r2 = found_filtered[i];
        r2.x += roundl(r2.width*0.1);
        r2.width = roundl(r2.width*0.8);
        r2.y += roundl(r2.height*0.06);
        r2.height = roundl(r2.height*0.9);

        // show bounding box
        // cv::rectangle(img, r2.tl(), r2.br(), cv::Scalar(0,255*weights[i],0), 2);

        cv::rectangle(smap, r2.tl(), r2.br(), cv::Scalar(1), cv::FILLED);
	}
    


    if(m_FaceCascadeEnabled) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, cv::Size(1400, 788));
        std::vector<cv::Rect> faceFeatures;
        if (m_FaceCascadeEnabled)
            m_face_cascade.detectMultiScale(gray, faceFeatures, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(15, 15));
        
        float scalingFactorX = static_cast<float>(img.cols) / 1400.f;
        float scalingFactorY = static_cast<float>(img.rows) / 788.f;

        for (size_t i = 0; i < faceFeatures.size(); ++i) {
            cv::Rect r2 = faceFeatures[i];
            r2.x *= scalingFactorX;
            r2.y *= scalingFactorY;
            r2.width *= scalingFactorX;
            r2.height *= scalingFactorY;

            r2.x += roundl(r2.width*0.1);
            r2.width = roundl(r2.width*0.8);
            r2.y += roundl(r2.height*0.06);
            r2.height = roundl(r2.height*0.9);

            cv::rectangle(smap, r2.tl(), r2.br(), cv::Scalar(1), cv::FILLED);
            
        }
    }
    
    return smap;
}


bool PedestrianFeatureMap::havePedestrian(int frame) {
    grabRequiredData(frame);

    cv::Mat img = m_frame.color;

    std::vector<cv::Rect> found, found_filtered;
    std::vector<double> weights;
    m_HOG.detectMultiScale(img, found, weights, 0.4, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    return !found.empty();
}


void PedestrianFeatureMap::grabRequiredData(int frame) {
    m_frame = FlowManager::get()->getFrame(frame);
}

cv::Mat PedestrianFeatureMap::getColor(int frame) {
    return FlowManager::get()->getFrame(frame).color;
}


