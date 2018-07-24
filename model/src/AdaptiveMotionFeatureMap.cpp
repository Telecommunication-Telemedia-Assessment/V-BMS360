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



#include "AdaptiveMotionFeatureMap.h"
#include "SalientFeatureFactory.h"
#include <iostream>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

AdaptiveMotionFeatureMap::AdaptiveMotionFeatureMap() : MotionFeatureMap(0) {

    m_flowClassifier = boost::shared_ptr<FlowClassier>(new FlowClassier("./data/fdeep_model.json"));
    m_pedestrianDriven = false;
}




std::vector<float> AdaptiveMotionFeatureMap::flowClassif(int frame) {

    ObjectMotionFeatureMap *objMotionModel = reinterpret_cast< ObjectMotionFeatureMap * >(SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ObjectMotionFeature));
    
    objMotionModel->grabRequiredData(frame);
    
    return m_flowClassifier->predict(objMotionModel->getFrontFlow());

}




cv::Mat AdaptiveMotionFeatureMap::compute(int frame) {

    // If we use the pedestrian driven approach we detect, and then use the bounding box as a mask for the object motion feature 
    if(m_pedestrianDriven) {
        PedestrianFeatureMap *detector = dynamic_cast<PedestrianFeatureMap*>(SalientFeatureFactory::get()->getModel(SalientFeatureFactory::PedestrianFeature));
        if(detector) {
            detector->grabRequiredData(frame);
            cv::Mat mask = detector->compute(frame);

            double mn, mx;
            cv::minMaxLoc(mask, &mn, &mx);

            // There is a pedestrian in the scene, in that case we are going to use the mask 
            if(mx > .5) {
                SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ObjectMotionFeature)->grabRequiredData(frame);
                cv::Mat map2 = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ObjectMotionFeature)->compute(frame);
                
                map2 = map2.mul(mask);

                cv::normalize(map2, map2, 0.0, 1.0, cv::NORM_MINMAX);

                return(map2);
            }
        }
    }

    // If we do not use pedestrian detector, or no pedestrian was found, we use the flow classification to switch between models
    std::vector<float> probs = flowClassif(frame);

    if(probs.empty()) return cv::Mat();

    float w1 = probs[0];
    float w2 = probs[1] + probs[2];

    // Some NA values appears to happen in the classifier when the input flow to classify is null. In that case, return the last successfull flow. 
    if(std::isnan(w1) || std::isnan(w2)) {
        return m_lastMap;
    }


    SalientFeatureMap *salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::MotionSourceFeature);
    salientFeature->grabRequiredData(frame);

	cv::Mat map1, map2;
    boost::thread_group g;
	if (probs[0] > 0.1) {
		// map1 = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::MotionSourceFeature)->compute(frame);
        g.create_thread(boost::bind(&AdaptiveMotionFeatureMap::getFeatureMapJob, this, frame, 1, boost::ref(map1)));
	}
	else {
		map1 = cv::Mat::zeros(FlowManager::get()->getFrame(frame).frame.size(), CV_32FC1);
	}

	if ((probs[1] + probs[2]) > 0.1) {
		// map2 = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ObjectMotionFeature)->compute(frame);
        g.create_thread(boost::bind(&AdaptiveMotionFeatureMap::getFeatureMapJob, this, frame, 2, boost::ref(map2)));
	} else {
		map2 = cv::Mat::zeros(FlowManager::get()->getFrame(frame).frame.size(), CV_32FC1);
	}
    g.join_all();
    
	if(m_verbose)
		std::cout << "probabilities: " << probs[0] << ", " << probs[1] << ", " << probs[2] << std::endl;

    if(map1.empty() && !map2.empty())
        return map2;
    
    if(map2.empty() && !map1.empty())
        return map1;

    if(map1.empty() && map2.empty())
        return cv::Mat();

    m_lastMap = (w1 * map1 + w2 * map2);

    return m_lastMap;

}

void AdaptiveMotionFeatureMap::getFeatureMapJob(int frame, int feature, cv::Mat& mat) {
    if(feature == 1) {
        mat = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::MotionSourceFeature)->compute(frame);
    }

    if(feature == 2) {
        mat = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ObjectMotionFeature)->compute(frame);
    }
}





