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


#include "SpatioTemporalFeatureMap.h"


#include <iostream>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "SalientFeatureFactory.h"

SpatioTemporalFeatureMap::SpatioTemporalFeatureMap() : MotionFeatureMap(15) {

}


cv::Mat SpatioTemporalFeatureMap::compute(int frame) {
	
	boost::thread_group group;

	cv::Mat imageFeature, motionFeature;

	group.create_thread(boost::bind(&SpatioTemporalFeatureMap::getMapJob, this, frame, 1, boost::ref(motionFeature)));
	group.create_thread(boost::bind(&SpatioTemporalFeatureMap::getMapJob, this, frame, 2, boost::ref(imageFeature)));

	group.join_all();

	if(imageFeature.empty() && !motionFeature.empty()) return motionFeature;
	if(!imageFeature.empty() && motionFeature.empty()) return imageFeature;
	if(imageFeature.empty() && motionFeature.empty())  return cv::Mat();

    cv::Mat result = 0.75 * imageFeature + 0.25 * motionFeature;
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);

	return result;
}

void SpatioTemporalFeatureMap::getMapJob(int frame, int feature, cv::Mat &smap) {
	if(feature == 1)
		smap = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::AdaptiveMotionFeature)->compute(frame);

	if (feature == 2)
		smap = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ImageFeature)->compute(frame);
}



