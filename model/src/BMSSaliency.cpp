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




#include "BMSSaliency.h"

#include <BMS.h>
#include <BMS360.h>
#include <UBMS.h>
#include <UBMS360.h>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "ShiftImage.hpp"
#include "EquatorialPrior.h"



BMSSaliency::BMSSaliency(bool loc_bms360, bool umat) {
	m_sampleStep			= 8;
	m_dilatationWidth1		= 3;
	m_dilatationWidth2		= 11;
	m_blurStd				= 20;
	m_normalize				= 1;
	m_handleBorder			= 0;
	m_colorSpace 			= CL_Lab;
	m_whitening				= true;
	m_maxDim				= 400.f;
	m_equatorialPrior 		= false;
	m_nb_projections 		= 4;
	m_bms360				= loc_bms360;
	m_useTAPI				= umat;
}




void BMSSaliency::process(const cv::Mat &inputImage, cv::Mat &sMap, bool normalize) {
	Configuration conf;

    conf.sampleStep = m_sampleStep;
	conf.dilatationWidth1 = m_dilatationWidth1;
	conf.dilatationWidth2 = m_dilatationWidth2;
	conf.blurStd = m_blurStd;
	conf.normalize = m_normalize;
	conf.handleBorder = m_handleBorder;
	conf.colorSpace = m_colorSpace;
	conf.whitening = m_whitening;
	conf.maxDim = static_cast<int>(m_maxDim);
	conf.equatorialPrior = m_equatorialPrior;


	std::vector<cv::Mat> outputs(m_nb_projections);

	// ------------------------------------------------------------------------------------
	// With the new version on GPU, we probably don't want to run that on multiple threads.

 
    boost::thread_group g;
    for(int i = 0 ; i < m_nb_projections ; ++i) {
    	g.create_thread(boost::bind(&BMSSaliency::processJob, this, i, m_nb_projections, boost::ref(inputImage), boost::ref(outputs), boost::ref(conf)));

    }
    g.join_all();



    //for(int i = 0 ; i < m_nb_projections; ++i) {
    // 	processJob(i, m_nb_projections, inputImage, outputs, conf);
    //}

    sMap = outputs[0];
    for(int i = 1 ; i < m_nb_projections ; ++i) {
    	sMap = sMap + shiftImage<float>(outputs[i], -i * outputs[i].cols / m_nb_projections, 0);
    }

	if(m_normalize) {
		if (m_blurStd > 0) {
			int blur_width = (int)MIN(floor(m_blurStd) * 4 + 1, 51);
			cv::GaussianBlur(sMap, sMap, cv::Size(blur_width, blur_width), m_blurStd, m_blurStd);
		}

		if(m_nb_projections > 1) {
			cv::normalize(sMap, sMap, 0.0, 1.0, cv::NORM_MINMAX);
		}
	}

	cv::resize(sMap, sMap, inputImage.size());



	if(m_equatorialPrior)
		applyEquatorialPrior(sMap, inputImage);


	

}



void BMSSaliency::processJob(int workerID, int nb_shift, const cv::Mat &input, std::vector<cv::Mat> &outputs, Configuration &conf) {
	cv::Mat inputImage = shiftImage<unsigned char>(input, workerID * input.cols / nb_shift, 0);

	processOneProjection(inputImage, outputs[workerID], conf.maxDim, conf.dilatationWidth1, conf.dilatationWidth2, conf.normalize, conf.handleBorder, conf.colorSpace, conf.whitening, conf.sampleStep, conf.blurStd);
}


// ------------------------------------------------------------------------------------------------------------------------------------------------------
// apply BMS on one frame

void BMSSaliency::processOneProjection(const cv::Mat &input, cv::Mat &output, int maxDim, int dilatationWidth1, int dilatationWidth2, int normalize, int handleBorder, int colorSpace, bool whitening, int sampleStep, float ) {

	cv::Mat src_small;
	float w = (float)input.cols, h = (float)input.rows;
	float maxD = fmax(w, h);

	cv::resize(input, src_small, cv::Size((int)(maxDim*w / maxD), (int)(maxDim*h / maxD)), 0.0, 0.0, cv::INTER_AREA);

	if (!m_useTAPI) {
		boost::shared_ptr<BMS> bms;
		if (m_bms360) {
			bms = boost::shared_ptr<BMS>(new BMS360(src_small, dilatationWidth1, normalize == 1, handleBorder == 1, colorSpace, whitening));
		}
		else {
			bms = boost::shared_ptr<BMS>(new BMS(src_small, dilatationWidth1, normalize == 1, handleBorder == 1, colorSpace, whitening));
		}

		bms->computeSaliency((double)sampleStep);

		cv::Mat result = bms->getSaliencyMap(false);
		
		if (dilatationWidth2 > 0)
			dilate(result, output, cv::Mat(), cv::Point(-1, -1), dilatationWidth2);
		else
			output = result;

	} else {
		boost::shared_ptr<UBMS> bms;
		if (m_bms360) {
			bms = boost::shared_ptr<UBMS>(new UBMS360(src_small, dilatationWidth1, normalize == 1, handleBorder == 1, colorSpace, whitening));
		}
		else {
			bms = boost::shared_ptr<UBMS>(new UBMS(src_small, dilatationWidth1, normalize == 1, handleBorder == 1, colorSpace, whitening));
		}
		
		bms->computeSaliency((double)sampleStep);

		output = bms->getSaliencyMap(false, dilatationWidth2);
	}

}




