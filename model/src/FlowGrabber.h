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

#ifndef _FlowManager_
#define _FlowManager_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/shared_ptr.hpp>
#include <list>

// #define GPU_MODE 1

#ifdef GPU_MODE
    #include <opencv2/cudaoptflow.hpp>
#else
	#ifndef CV_OVERRIDE
		#define CV_OVERRIDE
	#endif

    #include <opencv2/optflow.hpp>
#endif

struct Flow {
    int frameNumber;
    cv::Mat frame;
    cv::Mat color;
    cv::Mat flowProb;
};



class FlowGrabber {

public:
    virtual Flow  getFrame          (int frame) = 0; 
    virtual float getFrameRate      () = 0;
	virtual cv::Size getSourceFrameSize() = 0;
    virtual ~FlowGrabber            ()                  {}
};




class FileFlowGrabber : public FlowGrabber {

    std::vector<std::string> m_filelist;
    std::string              m_colorImagePath;

public:
    FileFlowGrabber                     (const std::vector<std::string> &filelist, const std::string& colorImgPath = "") : m_filelist(filelist), m_colorImagePath(colorImgPath) {}
    virtual ~FileFlowGrabber            () 									{}

    virtual float   getFrameRate        ()                  { return 30.0f; }
    virtual Flow    getFrame            (int frame);
	virtual cv::Size getSourceFrameSize	();

};





class VideoFlowGrabber : public FlowGrabber {

#ifdef GPU_MODE
    cv::Ptr<cv::cuda::DenseOpticalFlow> m_compute;
    cv::cuda::GpuMat                    m_gpuframe;
#else
    cv::Ptr<cv::DenseOpticalFlow> m_compute;
#endif

	cv::VideoCapture 			  m_capture;
    int							  m_curFrame;
    cv::Mat 					  m_frame;
    cv::Mat                       m_colorFrame;

    int                           m_scalingFactor;


public:
	VideoFlowGrabber		        (const std::string& filename);
    virtual ~VideoFlowGrabber       ()									{}

    virtual Flow  getFrame          (int frame);
    virtual float getFrameRate      ();
    int           getFrameCount     ();
	virtual cv::Size getSourceFrameSize();
}; 



class FlowManager {

    boost::shared_ptr<FlowGrabber>      m_grabber;
    std::list<Flow>                     m_cache;
    size_t                              m_cacheSize;
    static FlowManager                 *m_This;


public:
    static FlowManager *get();

    void setFlowGrabber(boost::shared_ptr<FlowGrabber> grabber)         { m_grabber = grabber; }
    
    boost::shared_ptr<FlowGrabber>
         getFlowGrabber()                                               { return m_grabber; }

    Flow    getFrame      (int frame);
    float   getFrameRate  ();
	cv::Size
		getSourceFrameSize();


private:
    FlowManager() : m_cacheSize(30) {};

} ;



#endif