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

#include "UBMS.h"

#include <vector>
#include <cmath>
#include <ctime>
#include <opencv2/highgui.hpp>
#include <chrono>

using namespace cv;
using namespace std;

#define COV_MAT_REG 50.0f

UBMS::UBMS(const Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening)
:mAttMapCount(0), mDilationWidth_1(dw1), mHandleBorder(hb), mNormalize(nm), mWhitening(whitening), mColorSpace(colorSpace) {

	mSrc=src.clone();
	mSaliencyMap = cv::UMat::zeros(src.size(), CV_32FC1);
	mBorderPriorMap = cv::Mat::zeros(src.size(), CV_32FC1);

	if (CL_RGB & colorSpace)
		whitenFeatMap(mSrc, COV_MAT_REG);
	if (CL_Lab & colorSpace) {
		Mat lab;
		cvtColor(mSrc, lab, COLOR_BGR2Lab);
		whitenFeatMap(lab, COV_MAT_REG);
	}
	if (CL_Luv & colorSpace) {
		Mat luv;
		cvtColor(mSrc, luv, COLOR_BGR2Lab);
		whitenFeatMap(luv, COV_MAT_REG);
	}
}

void UBMS::computeSaliency(double step) {

	cv::UMat bm;
	for (size_t i = 0 ; i < mFeatureMaps.size() ; ++i) {
		
		double max_,min_;
		minMaxLoc(mFeatureMaps[i],&min_,&max_);
		for (double thresh = min_; thresh < max_; thresh += step) {

			cv::threshold(mFeatureMaps[i], bm, thresh, 255, cv::THRESH_BINARY);
			cv::UMat am = getAttentionMap(bm, mDilationWidth_1, mNormalize, mHandleBorder);

			cv::add(am, mSaliencyMap, mSaliencyMap);
			mAttMapCount++;
		}
	}

}


cv::UMat UBMS::getAttentionMap(const cv::UMat& bm, int dilation_width_1, bool toNormalize, bool handle_border) {
	using namespace std::chrono;

	cv::UMat ret = bm.clone();

	int jump;
	if (handle_border) {

		cv::Mat cpuRet = ret.getMat(cv::ACCESS_FAST);

		for (int i = 0; i < bm.rows; i++) {

			jump = BMS_RNG.uniform(0.0, 1.0)>0.99 ? BMS_RNG.uniform(5, 25) : 0;
			if (cpuRet.at<uchar>(i, 0 + jump) != 1) {
				cv::floodFill(ret, cv::Point(0 + jump, i), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}

			jump = BMS_RNG.uniform(0.0, 1.0)>0.99 ? BMS_RNG.uniform(5, 25) : 0;
			if (cpuRet.at<uchar>(i, bm.cols - 1 - jump) != 1) {
				cv::floodFill(ret, cv::Point(bm.cols - 1 - jump, i), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}

		}

		for (int j = 0; j < bm.cols; j++) {

			jump = BMS_RNG.uniform(0.0, 1.0)>0.99 ? BMS_RNG.uniform(5, 25) : 0;
			if (cpuRet.at<uchar>(0 + jump, j) != 1) {
				cv::floodFill(ret, cv::Point(j, 0 + jump), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}

			jump = BMS_RNG.uniform(0.0, 1.0)>0.99 ? BMS_RNG.uniform(5, 25) : 0;
			if (cpuRet.at<uchar>(bm.rows - 1 - jump, j) != 1) {
				cv::floodFill(ret, cv::Point(j, bm.rows - 1 - jump), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}
		}
	}
	else {

		cv::Mat cpuRet = ret.getMat(cv::ACCESS_FAST);
		for (int i = 0; i < bm.rows; i++) {

			if (cpuRet.at<uchar>(i, 0) != 1) {
				cv::floodFill(ret, cv::Point(0, i), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}

			if (cpuRet.at<uchar>(i, bm.cols - 1) != 1) {
				cv::floodFill(ret, cv::Point(bm.cols - 1, i), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}

		}

		for (int j = 0; j < bm.cols; j++) {

			if (cpuRet.at<uchar>(0, j) != 1) {
				cv::floodFill(ret, cv::Point(j, 0), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}


			if (cpuRet.at<uchar>(bm.rows - 1, j) != 1) {
				cv::floodFill(ret, cv::Point(j, bm.rows - 1), cv::Scalar(1), 0, cv::Scalar(0), cv::Scalar(0), 8);
				//cpuRet = ret.getMat(cv::ACCESS_READ);
			}

		}
	}

	

	cv::bitwise_xor(ret, cv::Scalar(1), ret);
	cv::UMat map1, map2;

	cv::bitwise_and(ret, bm, map1);
	cv::UMat notBM;

	cv::subtract(cv::Scalar(1), bm, notBM);
	cv::bitwise_and(ret, notBM, map2);

	if (dilation_width_1 > 0) {
		for (int k = 0; k < (dilation_width_1 - 1) / 2; ++k) {
			dilate(map1, map1, cv::UMat(), cv::Point(-1, -1), 3);
			dilate(map2, map2, cv::UMat(), cv::Point(-1, -1), 3);
		}
	}

	map1.convertTo(map1, CV_32FC1);
	map2.convertTo(map2, CV_32FC1);

	bmsNormalize(map1);
	bmsNormalize(map2);

	cv::add(map1, map2, map1);

	return map1;

}

void UBMS::bmsNormalize(cv::UMat& map1) {
	cv::normalize(map1, map1, 1.0, 0.0, cv::NORM_L2);
}

Mat UBMS::getSaliencyMap(bool normalized, int dilationWidth2, float blurStd) {

	if (dilationWidth2 > 0) {
		for (int k = 0; k < (dilationWidth2 - 1) / 2; ++k) {
			cv::dilate(mSaliencyMap, mSaliencyMap, cv::UMat(), cv::Point(-1, -1), 3);
		}
	}

	if (blurStd > 0) {
		int blur_width = (int)MIN(floor(blurStd) * 4 + 1, 51);
		cv::GaussianBlur(mSaliencyMap, mSaliencyMap, cv::Size(blur_width, blur_width), blurStd, blurStd);
	}

	if(normalized) {
		Mat ret; 
		normalize(mSaliencyMap, ret, 0.0, 1.0, NORM_MINMAX);
		//ret.convertTo(ret,CV_8UC1);
		return ret;
	} else {
		cv::Mat cpuSal;
		mSaliencyMap.copyTo(cpuSal);
		return cpuSal;
	}
}

void UBMS::whitenFeatMap(const cv::Mat& img, float reg) {

	assert(img.channels() == 3 && img.type() == CV_8UC3);
	
	std::vector<UMat> featureMaps;
	
	if (!mWhitening) {
		cv::UMat uImg;
		img.copyTo(uImg);

		cv::split(uImg, featureMaps);
		for (size_t i = 0; i < featureMaps.size(); i++) {

			cv::normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, cv::NORM_MINMAX);
			cv::medianBlur(featureMaps[i], featureMaps[i], 3);
			mFeatureMaps.push_back(featureMaps[i]);
		}
		return;
	}


	Mat srcF,meanF,covF;
	img.convertTo(srcF, CV_32FC3);
	Mat samples = srcF.reshape(1, img.rows*img.cols);
	calcCovarMatrix(samples, covF, meanF, COVAR_NORMAL | COVAR_ROWS | COVAR_SCALE, CV_32F);

	covF += Mat::eye(covF.rows, covF.cols, CV_32FC1)*reg;
	SVD svd(covF);
	Mat sqrtW;
	sqrt(svd.w,sqrtW);
	Mat sqrtInvCovF = svd.u * Mat::diag(1.0/sqrtW);

	Mat whitenedSrc = srcF.reshape(1, img.rows*img.cols)*sqrtInvCovF;
	whitenedSrc = whitenedSrc.reshape(3, img.rows);
	
	split(whitenedSrc, featureMaps);

	for (int i = 0; i < featureMaps.size(); i++) {
		normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX);
		featureMaps[i].convertTo(featureMaps[i], CV_8U);

		cv::UMat feature;
		featureMaps[i].copyTo(feature);

		medianBlur(feature, feature, 3);
		mFeatureMaps.push_back(feature);
	}
}

