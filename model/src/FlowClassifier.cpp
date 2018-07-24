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


#include "FlowClassifier.h"

#include <opencv2/imgproc.hpp>
#include "FlowIO.h"

void void_logger(const std::string& ) { }

FlowClassier::FlowClassier(const std::string& modelPath, int height, int width) : 
            m_kerasModel(fdeep::load_model(modelPath, true, void_logger)) {
                
    m_height = height;
    m_width = width;

}


std::vector<float> FlowClassier::predict(const cv::Mat& flow) const {

    if(flow.empty()) {
        return std::vector<float>();
    }
    
    // The model was trained to work with images having a size of m_height/m_width
    cv::Mat scaledFlow;
    if(flow.rows != m_height && flow.cols != m_width) {
        cv::resize(flow, scaledFlow, cv::Size(m_width, m_height), 0, 0, cv::INTER_AREA);
    } else {
        scaledFlow = flow;
    }

    // writeFlow("motion.flo", scaledFlow);

    // Copy the data into a tensor
    fdeep::tensor3 input_tensor(fdeep::shape3(2, scaledFlow.rows, scaledFlow.cols), 0);

    for(int j = 0 ; j < scaledFlow.rows ; ++j) {
        for(int i = 0 ; i < scaledFlow.cols ; ++i) {
            const cv::Point2f& p = scaledFlow.at<cv::Point2f>(j,i); 
            input_tensor.set(0, j, i, p.x);
            input_tensor.set(1, j, i, p.y);
        }
    }

    // Predict the tensor
    fdeep::tensor3s inputs;
    inputs.push_back(input_tensor);

    const auto result = m_kerasModel.predict(inputs);

    std::vector<float> classif;
    for(size_t k = 0 ; k < result.size() ; ++k) {
        classif.resize(result[k].depth());
        for(size_t n = 0 ; n < result[k].depth() ; ++n)
            classif[n] = result[k].get(n,0,0);
    }

    // Return probability of each class
    return classif;
}



