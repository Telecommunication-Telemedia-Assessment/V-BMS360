/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Exploit Surroundedness for Saliency Detection: A Boolean Map Approach",
*   Jianming Zhang, Stan Sclaroff, submitted to PAMI, 2014
*
*	Copyright (C) 2014 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: jmzhang@bu.edu
*******************************************************************************/

#ifndef UBMS_H
#define UBMS_H

#ifdef IMDEBUG
#include <imdebug.h>
#endif
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "BMS.h"

class UBMS {

public:
	UBMS (const cv::Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening);
	cv::Mat getSaliencyMap(bool normalized = true, int dilationWidth2 = 11, float blurStd = 0);
	void computeSaliency(double step);
	virtual ~UBMS() {};


protected:
	virtual void bmsNormalize(cv::UMat& mat1);

private:
	cv::UMat mSaliencyMap;
	int mAttMapCount;
	cv::Mat mBorderPriorMap;
	cv::Mat mSrc;
	std::vector<cv::UMat> mFeatureMaps;
	int mDilationWidth_1;
	bool mHandleBorder;
	bool mNormalize;
	bool mWhitening;
	int mColorSpace;
	cv::UMat getAttentionMap(const cv::UMat& bm, int dilation_width_1, bool toNormalize, bool handle_border);
	void whitenFeatMap(const cv::Mat& img, float reg);
	void computeBorderPriorMap(float reg, float marginRatio);
};

void postProcessByRec8u(cv::Mat& salmap, int kernelWidth);
void postProcessByRec(cv::Mat& salmap, int kernelWidth);



#endif


