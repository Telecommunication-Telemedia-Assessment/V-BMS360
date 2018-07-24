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


#include "TemporalPrior.h"
#include <cstdlib>

void applyTemporalPrior(cv::Mat& image, float timeF, std::vector<float> &startP) {
    if(image.empty()) return;

    float timeFactor = (30.0f * timeF ) / (161.0720f);
    float cutOff = 180 * (1 - exp(-timeFactor*timeFactor));

    for (int i = 0; i < image.cols; ++i) {
		
        
        float prior = 0.f;

        for(size_t k = 0 ; k < startP.size() ; ++k ) {
            float lng = std::abs(startP[k] - static_cast<float>(i) / image.cols) * 360.f;
            lng = std::min(lng, std::abs(360.f-lng));

            if(std::abs(lng) > cutOff) {
                prior = std::fmax(prior, exp(-(lng-cutOff)*(lng-cutOff)/40));
            } else {
                prior = 1.f;
            }

        }

        

		for (int j = 0 ; j < image.rows ; ++j) {
			image.at<float>(j, i) = prior * image.at<float>(j, i);
		}
	}

    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);
}



