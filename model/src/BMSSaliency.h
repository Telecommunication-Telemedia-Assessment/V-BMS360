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




#ifndef _BMSSaliency_
#define _BMSSaliency_


#include <opencv2/core.hpp>


struct Configuration {
	int sampleStep;
	int dilatationWidth1;
	int dilatationWidth2;
	float blurStd;
	int normalize;
	int handleBorder;
	int colorSpace;
	bool whitening;
	int maxDim;
	bool equatorialPrior;
}; 




class BMSSaliency  {

public:
	int 				m_sampleStep;
	int 				m_dilatationWidth1;
	int 				m_dilatationWidth2;
	float				m_blurStd;
	bool				m_normalize;
	bool				m_handleBorder;
	int 				m_colorSpace;
	bool 				m_whitening;
	float				m_maxDim;
	bool				m_equatorialPrior;
	int 				m_nb_projections;
	bool 				m_bms360;
	bool				m_useTAPI;


public:
	BMSSaliency 		(bool bms360, bool umat);
	virtual ~BMSSaliency()											{};

	virtual void process(const cv::Mat &input, cv::Mat &output, bool normalize = true);



private:


	// ------------------------------------------------------------------------------------------------
	// Apply saliency model + Multiple map fusion

	void processOneProjection(const cv::Mat &input, cv::Mat &output, int maxDim, int dilatationWidth1, int dilatationWidth2, int normalize, int handleBorder, int colorSpace, bool whitening, int sampleStep, float blurStd);
	void processJob(int workerID, int nb_shift, const cv::Mat &input, std::vector<cv::Mat> &outputs, Configuration &conf);





};



#endif

