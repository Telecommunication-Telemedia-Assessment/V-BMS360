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



#ifndef _PedestrianFeatureMap_
#define _PedestrianFeatureMap_

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include "SalientFeatureMap.h"
#include "FlowGrabber.h"

class PedestrianFeatureMap: public SalientFeatureMap {

public:

    enum PedestrianMode {
        STRICT_PED = 0,
        ADD_FACE_DETECTOR
    };


private:
    cv::HOGDescriptor           m_HOG;
    Flow                        m_frame;
    cv::CascadeClassifier       m_face_cascade;
    bool                        m_FaceCascadeEnabled;

public:

    


    PedestrianFeatureMap(PedestrianMode mode = STRICT_PED);
    virtual ~PedestrianFeatureMap() {};



    virtual cv::Mat compute                 (int frame);
    virtual void    grabRequiredData        (int targetFrame);
    virtual cv::Mat getColor                (int frame);
    bool            havePedestrian          (int frame);
    

};




#endif
