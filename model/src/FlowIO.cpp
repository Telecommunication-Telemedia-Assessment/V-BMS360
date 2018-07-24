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


#include "FlowIO.h"

#include <fstream>
#include <iostream>
#include <vector>


cv::Mat readFlow(const std::string& filename) {
    FILE *f = fopen(filename.c_str(), "rb");
    if(f == NULL) {
        return cv::Mat();
    }


    float magic = 0;
    fread(reinterpret_cast<unsigned char*>(&magic), sizeof(float), 1, f);
    if(magic != 202021.25f) {
        std::cerr << "Magic number is not correct!" << std::endl;
        fclose(f);
        return cv::Mat();
    }

    int width, height;
    fread(reinterpret_cast<unsigned char*>(&width), sizeof(int), 1, f);
    fread(reinterpret_cast<unsigned char*>(&height), sizeof(int), 1, f);

    cv::Mat flow(height, width, CV_32FC2, cv::Scalar(0.f));
    fread(flow.data, sizeof(float), width*height*2, f);

    fclose(f);

    return flow;
}

void writeFlow(const std::string& filename, const cv::Mat &flow) {
    FILE *f = fopen(filename.c_str(), "wb");
    if(f == NULL) {
        std::cerr << "cannot write flo file" << std::endl;
        return ;
    }

    float magic = 202021.25f;
    fwrite(reinterpret_cast<unsigned char*>(&magic), sizeof(float), 1, f);

    int width = flow.cols;
    int height = flow.rows;
    fwrite(reinterpret_cast<unsigned char*>(&width), sizeof(int), 1, f);
    fwrite(reinterpret_cast<unsigned char*>(&height), sizeof(int), 1, f);


    fwrite(flow.data, sizeof(float), width*height*2, f);

    fclose(f);

}








