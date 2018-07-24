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



#ifndef _MotionSourceFeatureMap_
#define _MotionSourceFeatureMap_

#include <opencv2/core.hpp>
#include "MotionFeatureMap.h"

class MotionSourceFeatureMap: public MotionFeatureMap {

private:
    int                 m_salmapmaxsize;  // Master map resolution
    float               m_tol;            // theshold for convergence detection in eigen vector computation
    int                 m_multires;       // multi-resolution analysis (2^multires)
    bool                m_cyclic;         // while computing interraction btw pix: width  + 1 == 1 ? 
    float               m_smoothness;     // with multi-resolution, weight the impact of low res on higher res

    std::vector<int>    m_salmapmaxsize_v;
    bool                m_initDone;

public:
    MotionSourceFeatureMap();
    virtual ~MotionSourceFeatureMap() {};



    virtual cv::Mat compute(int frame);


private:
    
    bool    init                    ();
    cv::Mat transitionProb          (const cv::Mat &flo)                                                        const ;
    void    principalEigenvectorRaw (const cv::Mat& markovA, float tol, std::vector<float>&  AL, int &iteri)    const ;
    void    principalEigenvectorRawGPU(const cv::Mat& markovA, float tol, std::vector<float>& AL, int &iteri)   const ;

    cv::Mat computeFeature          ();
    cv::Mat computeActivation       (const cv::Mat &p);

    

};



#endif
