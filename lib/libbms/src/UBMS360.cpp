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




#include "UBMS360.h"
#include <opencv2/highgui.hpp>
//
//std::mutex UBMS360::scaleMutex;
//cv::UMat   UBMS360::scaleMatrix;



void UBMS360::bmsNormalize(cv::UMat& map1) {
    
    scaleBooleanMap(map1);

    cv::normalize(map1, map1, 1.0, 0.0, cv::NORM_L2);
}


void UBMS360::scaleBooleanMap(cv::UMat& map) {

	scaleMutex.lock();
	if (scaleMatrix.cols != map.cols && scaleMatrix.rows != map.rows) {
		cv::Mat scaleMat(map.size(), CV_32FC1);

		for (int i = 0; i < map.rows; ++i) {
			float c = std::cos(3.1415926535898f * static_cast<float>(map.rows / 2 - i) / map.rows);
			for (int j = 0; j < map.cols; ++j) {
				scaleMat.at<float>(i, j) = c;
			}
		}

		scaleMat.copyTo(scaleMatrix);
	}
	scaleMutex.unlock();

    
	cv::multiply(map, scaleMatrix, map);
    
}