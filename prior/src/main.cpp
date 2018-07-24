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




#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <boost/program_options.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <list>
#include "TemporalPrior.h"
#include "EquatorialPrior.h"

bool getFrameFromFile(cv::Mat &frame, FILE *file) {

	return (fread(frame.data, sizeof(float), frame.cols*frame.rows, file) == frame.cols*frame.rows);

}

bool getFrameFromVideoCapture(cv::Mat &frame, cv::VideoCapture &capture, int width, int height) {
	cv::Mat mat;
	capture >> mat;

	if(mat.empty()) return false;

	cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);

	if(width != -1 && height != -1) {
		cv::resize(mat, mat, cv::Size(width, height));
	}

	mat.convertTo(frame, CV_32FC1);

	return true;
}


int main(int argc, char **argv) {


	// ------------------------------------------------------------------------------------------------------------------------------------------
	// parse parameters

	namespace po = boost::program_options;

	po::positional_options_description p;
	p.add("input-file", -1);

	po::options_description desc("Allowed options");
	desc.add_options()
			("help", "produce help message")
			("input-file,i", po::value< std::string >(), "Saliency input video.")
			("color-file,c", po::value< std::string >(), "Color video corresponding to the saliency map.")
			("output-file,o", po::value< std::string >(), "Saliency map with the equatorial prior.")
			("input-file-mix", po::value< std::string >(), "Perform a mixture with another salieny map.")
			("width,w", po::value< int >(), "Width of the saliency map (required when the input is binary)" )
			("height,h", po::value< int >(), "Height of the saliency map (required when the input is binary)" )
			("equatorial-prior", "Apply the equatorial prior")
			("temporal-prior", po::value< int >(), "Apply the temporal prior (N sources)")
			("no-adapt", "No adaptation")
			("mixture", po::value< float >(), "Weighting between the first and second saliency map (default 0.5)" )
			("erode",  po::value< int >(), "Perform an erosion on the output saliency map (before prior)")
			("smooth", "Temporal smoothing")
			("temporal-scaling", po::value< float >(), "Add a scaling factor on the temporal prior (make it shorter or longer)" )
	;


	po::variables_map vm;
	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
		po::notify(vm);
	} catch(boost::exception &) {
		std::cerr << "Error incorect program options. See --help... \n";
		return 0;
	}

	if (vm.count("help")) {
		std::cout << "--------------------------------------------------------------------------------\n";
		std::cout << "\t\tApply the priors on the saliency maps\n";
		std::cout << "--------------------------------------------------------------------------------\n";
		std::cout << "\n\n";
		std::cout << desc << "\n";
		return 1;
	}

	
	std::string inputImagePath;
	std::string inputMixturePath;
	std::string inputColorImagePath;
	std::string outputPath;

	bool noAdapt = false;
	
	if (vm.count("input-file")) {
		inputImagePath = vm["input-file"].as< std::string >();
	} else {
		std::cerr << "It is required to provide the an input saliency map. See --help\n";
		return 0;
	}

	if (vm.count("input-file-mix")) {
		inputMixturePath = vm["input-file-mix"].as< std::string >();
	} 

	if (vm.count("color-file")) {
		inputColorImagePath = vm["color-file"].as< std::string >();
	} else {
		std::cerr << "It is required to provide the an input color image. See --help\n";
		return 0;
	}

	if (vm.count("output-file")) {
		outputPath = vm["output-file"].as< std::string >();
	} 

	if(vm.count("no-adapt")) {
		noAdapt = true;
	}

	int width = -1;
	if (vm.count("width")) {
		width = vm["width"].as< int >();
	}

	int height = -1;
	if (vm.count("height")) {
		height = vm["height"].as< int >();
	}

	float frame_rate = 25.0f;
	int erodeK = 0;
	if (vm.count("erode")) {
		erodeK = vm["erode"].as< int >();
	}

	bool equatorialPrior = false;
	if (vm.count("equatorial-prior")) {
		equatorialPrior = true;
	}
	
	int temporalPrior = 0;
	if (vm.count("temporal-prior")) {
		temporalPrior = vm["temporal-prior"].as< int >();
	}

	float temporalScaling = 1.0f;
	if(vm.count("temporal-scaling")) {
		temporalScaling = vm["temporal-scaling"].as< float >();
	}

	float mixture = 0.5f;
	if (vm.count("mixture")) {
		mixture = vm["mixture"].as< float >();
	}

	bool smooth = false;
	if(vm.count("smooth")) {
		smooth = true;
	}


	FILE *file = NULL;
	FILE *mixFile = NULL;
	FILE *oFile = NULL;

	// ---------------------------------------------------------------------------------------------------
	// Get statistics

	std::string extension = inputImagePath.substr(inputImagePath.find_last_of('.'), 4);
	std::transform(extension.begin(), extension.end(), extension.begin(), ::toupper);

	boost::function<bool (cv::Mat&)> getFrame;
	boost::function<bool (cv::Mat&)> getFrameMix;
	cv::VideoCapture salCapture;

	std::vector<cv::Mat> tempCache(4);
	std::vector<float>  tempCacheW = {3, 2.8f, 2, 1};

	if (extension == ".BIN") {
		if(width == -1 || height == -1) {
			std::cerr << "[E] With a binary input it is required to specify the width and height. See --help." << std::endl;
			return -1;
		} else {
			file = fopen(inputImagePath.c_str(), "rb");
			if(file == NULL) {
				std::cerr << "[E] Cannot open: " << inputImagePath << std::endl;
				return -1;
			}

			getFrame = boost::bind(getFrameFromFile, _1, file);
		}
	} else {
		salCapture.open(inputImagePath);
		if(!salCapture.isOpened()) {
			std::cerr << "[E] Cannot open: " << inputImagePath << std::endl;
			if(file != NULL) {
				fclose(file);
			}
			return -1;
		} 

		getFrame = boost::bind(getFrameFromVideoCapture, _1, boost::ref(salCapture), width, height);
	}

	if(!inputMixturePath.empty()) {
		if(width == -1 || height == -1) {
			std::cerr << "[E] With a binary input it is required to specify the width and height. See --help." << std::endl;
			return -1;
		} else {
			mixFile = fopen(inputMixturePath.c_str(), "rb");
			if(mixFile == NULL) {
				std::cerr << "[E] Cannot open: " << inputImagePath << std::endl;
				return -1;
			}

			getFrameMix = boost::bind(getFrameFromFile, _1, mixFile);
		}
	}


	cv::VideoCapture colorVideo(inputColorImagePath);
	if(!colorVideo.isOpened()) {
		std::cerr << "[E] Cannot open: " << inputColorImagePath << std::endl;
		if(file != NULL) {
			fclose(file);
		}
		return -1;
	}
	frame_rate = colorVideo.get(cv::CAP_PROP_FPS);


	if(!outputPath.empty()) {
		oFile = fopen(outputPath.c_str(), "wb");
		if(oFile == NULL) {
			std::cerr << "[E] Cannot write to: " << outputPath << std::endl;
			if(file != NULL) {
				fclose(file);
			}

			return -1;
		}
	}


	cv::Mat sMap(cv::Size(width, height), CV_32FC1);
	cv::Mat sMapMix;
	cv::Mat cMat;

	bool success = getFrame(sMap);
	if(!inputMixturePath.empty()) {
		sMapMix = cv::Mat(cv::Size(width, height), CV_32FC1);
		success &= getFrameMix(sMapMix);
	}
	colorVideo >> cMat;

	int frIdx = 0;
	while(!cMat.empty() && success) {

		if(!sMapMix.empty()) {
			sMap = mixture * sMap + (1 - mixture) * sMapMix;
		}

		if(erodeK > 0) {
			erode(sMap, sMap, cv::Mat(), cv::Point(-1, -1), erodeK);
		}

		if (equatorialPrior) {
			applyEquatorialPrior(sMap, cMat);
		}
		
		if(temporalPrior > 0) {
			std::vector<float> startP;
			for(int k = 0 ; k < temporalPrior ; ++k) {
				startP.push_back(0.5f + static_cast<float>(k) * 1.f / static_cast<float>(temporalPrior));
			}
			applyTemporalPrior(sMap, frIdx / frame_rate, startP, temporalScaling);
		}

		cv::normalize(sMap, sMap, 0, 1, cv::NORM_MINMAX);

		if(smooth) {
			for(int k = 0 ; k < tempCache.size() - 1 ; ++k) {
				tempCache[k+1] = tempCache[k];
			}
			tempCache[0] = sMap.clone();

			float sumW = 0;
			cv::Mat res(sMap.size(), CV_32FC1, cv::Scalar(0));
			for(int k = 0 ; k < tempCache.size() ; ++k) {
				if(!tempCache[k].empty()) {
					res += tempCache[k] * tempCacheW[k];
					sumW += tempCacheW[k];
				}
			}

			sMap = res / sumW;
		}


		if(!outputPath.empty()) {
			fwrite(sMap.data, sizeof(float), sMap.cols*sMap.rows, oFile);
		} else {
			cv::imshow("with prior", sMap);
			cv::waitKey();
		}

		++frIdx;
		success = getFrame(sMap);
		if(!inputMixturePath.empty()) {
			success &= getFrameMix(sMapMix);
		}
		colorVideo >> cMat;
	}


	if(oFile != NULL) {
		fclose(oFile);
	}

	if(file != NULL) {
		fclose(file);
	}

	return 0;
}

