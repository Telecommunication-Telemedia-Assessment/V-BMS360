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


#include "Saliency360.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <gnomonic-all.h>
#include "common-method.h"

#include <iostream>
#include "FlowIO.h"
#include <chrono>

#include "SalientFeatureFactory.h"
#include "EquatorialPrior.h"
#include "TemporalPrior.h"



Saliency360::Saliency360() {
    
    temporalWindow  = 15;
    model           = 6;
    benchmark       = false;
    m_callCount     = 0;
    equatorialPrior = false;
    temporalPrior   = 0;
	enableOverlay   = true;
	ocl				= false;
    erodeK          = 71;
}




cv::Mat Saliency360::compute(int frame) {

    using namespace std::chrono;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    high_resolution_clock::time_point t2;

    cv::Mat master_map;
    SalientFeatureMap *salientFeature;
    switch(model) {
        case 0: {
            salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ObjectMotionFeature);
            break;
        }

        case 1: {
            salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::MotionSourceFeature);
            break;
        }

        case 2:
        default: {
            salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::AdaptiveMotionFeature);
            break;
        }

        case 3: {
            salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::ImageFeature);
            break;
        }

        case 4: {
            salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::PedestrianFeature);
            break;
        }

        case 5: {
            salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::AdaptiveMotionFeature);
            AdaptiveMotionFeatureMap *adaptiveModel = dynamic_cast<AdaptiveMotionFeatureMap*>(salientFeature);
            if(adaptiveModel) {
                adaptiveModel->setPedestrianDriven(true);
            }
            break;
        }

        case 6: {
            salientFeature = SalientFeatureFactory::get()->getModel(SalientFeatureFactory::SpatioTemporalFeature);
            break;
        }
    }

	salientFeature->setOCLMode(ocl);

	std::cout << "[FL]";
    salientFeature->grabRequiredData(frame);
    t2 = high_resolution_clock::now();

	std::cout << "[SM]";
    master_map = salientFeature->compute(frame);


    if(!master_map.empty() && erodeK > 0) {
        cv::resize(master_map, master_map, cv::Size(2048, 1024));
		erode(master_map, master_map, cv::Mat(), cv::Point(-1, -1), erodeK);
    }

	if (equatorialPrior) {
		std::cout << "[SP]";
		applyEquatorialPrior(master_map, FlowManager::get()->getFrame(frame).color);
	}
    
    if(temporalPrior > 0) {
		std::cout << "[TP]";
        std::vector<float> startP;
        for(int k = 0 ; k < temporalPrior ; ++k) {
            startP.push_back(0.5f + static_cast<float>(k) * 1.f / static_cast<float>(temporalPrior));
        }
        applyTemporalPrior(master_map, frame / FlowManager::get()->getFrameRate(), startP);
    }

    high_resolution_clock::time_point t3 = high_resolution_clock::now();

	if(enableOverlay)
	    showOverlay(FlowManager::get()->getFrame(frame).color, master_map);

    

    duration<double> time_span1 = duration_cast<duration<double>>(t2 - t1);
    duration<double> time_span2 = duration_cast<duration<double>>(t3 - t2);
    
    if(benchmark)
        std::cout << "Chronometer -- grab " << time_span1.count() << " computeFeature " << time_span2.count() << std::endl;
    

    return master_map;
}



void Saliency360::showOverlay(const cv::Mat &colorImage, cv::Mat &sMap) const {
    if (colorImage.empty()) {
        std::cout << "color image is empty..." << std::endl;
    }  else {

        cv::Mat localColor;
        cv::resize(colorImage, localColor, sMap.size(), 0, 0);

        localColor.forEach<cv::Point3_<unsigned char>>([sMap](cv::Point3_<unsigned char> &p, const int *position) -> void {
            float sal = sMap.at<float>(position[0], position[1]);
            p.x = static_cast<int>(p.x) / 2;
            p.y = static_cast<int>(p.y) / 2;
            p.z = std::min(255.f, static_cast<int>(p.z) / 2 + 255 * sal);
        });

        if (logOutput.empty())
            cv::imshow("colorImage", localColor);
        else {
            cv::imwrite(logOutput, localColor);
        }
    }
}



void Saliency360::rectilinearToEquirectangular(const cv::Mat& inputImage, cv::Mat& output, float azim, float elev, float roll, float fov) const {
    lg_gte_apperturep( 
        ( inter_C8_t * ) output.data,
        output.cols,
        output.rows,
        output.channels(),
        ( inter_C8_t * ) inputImage.data,
        inputImage.cols,
        inputImage.rows,
        inputImage.channels(),
        azim  * ( LG_PI / 180.0 ),
        elev  * ( LG_PI / 180.0 ),
        roll  * ( LG_PI / 180.0 ),
        fov   * ( LG_PI / 180.0 ),
        lc_method( "bicubicf"  ),
        8
    );
}

void Saliency360::equirectangularToRectilinear(const cv::Mat& inputImage, cv::Mat& output, float azim, float elev, float roll, float fov) const {
    lg_etg_apperturep( 
        ( inter_C8_t * ) inputImage.data,
        inputImage.cols,
        inputImage.rows,
        inputImage.channels(),
        ( inter_C8_t * ) output.data,
        output.cols,
        output.rows,
        output.channels(),
        azim    * ( LG_PI / 180.0 ),
        elev    * ( LG_PI / 180.0 ),
        roll    * ( LG_PI / 180.0 ),
        fov     * ( LG_PI / 180.0 ),
        lc_method( "bicubicf" ),
        8
    );
}
