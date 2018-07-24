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



#ifndef _Saliency360_
#define _Saliency360_

#include <opencv2/core.hpp>
#include <vector>

#include <boost/shared_ptr.hpp>


class Saliency360 {

public:
    
    int                 temporalWindow;
    int                 model;
    bool                benchmark;
    bool                equatorialPrior;
    int                 temporalPrior;
	bool				enableOverlay;
	bool				ocl;
    int                 erodeK;
    
    std::string         logOutput;
    
private:

    cv::Mat                              m_curColorFrame;
    int                                  m_callCount;

public:
    Saliency360                     ();
    

    cv::Mat compute                 (int frame);


private:

    void    showOverlay             (const cv::Mat &colorImage, cv::Mat &sMap)                                   const;

    void    cameraMotionEstimation  (int frame);
    std::vector<float>  flowClassif (int frame);

    void    rectilinearToEquirectangular(const cv::Mat &inputImage, cv::Mat &output, float azim, float elev, float roll = 0.f, float fov = 90.f) const;
    void    equirectangularToRectilinear(const cv::Mat& inputImage, cv::Mat& output, float azim, float elev, float roll = 0.f, float fov = 90.f) const;
};





#endif


