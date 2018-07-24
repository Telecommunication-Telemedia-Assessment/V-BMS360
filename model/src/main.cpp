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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include "SalientFeatureFactory.h"

#include "FlowIO.h"
#include "Saliency360.h"
#include "FlowGrabber.h"
#include <opencv2/core/ocl.hpp>


void showSaliency(const cv::Mat &sMap, int delay) {
	if (!sMap.empty()) {
		cv::imshow("saliency", sMap);
		cv::waitKey(delay);
	}
}

void saveSaliency(const cv::Mat &sMap, std::string filename) {
	if(sMap.empty()) return;

	cv::Mat tmp;
	cv::cvtColor(sMap, tmp, cv::COLOR_GRAY2BGR);
	tmp *= 255;
	tmp.convertTo(tmp, CV_8UC3);
	cv::imwrite(filename, tmp);
}

void saveBinarySaliency(const cv::Mat &sMap, FILE *file) {
	if(sMap.empty()) return;

	fwrite(sMap.data, sizeof(float), sMap.cols*sMap.rows, file);
}

#define SUBMISSION 1

// ------------------------------------------------------------------------------------------------------------------------------------------------------


int main(int argc, char **argv) {



	// ------------------------------------------------------------------------------------------------------------------------------------------
	// parse parameters

	namespace po = boost::program_options;

	po::positional_options_description p;
	p.add("input-flow", -1);

	po::options_description desc("Allowed options");
	desc.add_options()
			("help", "produce help message")
			("input-video,i", po::value< std::string >(), "Video file to process.")
			("output-file,o", po::value< std::string >(), "Output image/video. Write images if only a frame is requested, write a video otherwise. Output format guessed from file extension. .bin -> binary file. .jpg -> an image. If not set, show in a window.")
			("frame,f",  po::value< int >(), "Process a specific frame. If not set, process all the video.")
			("duration,d",  po::value< int >(), "Process a specific number of frames. If not set, process all the video.")
	
#ifndef SUBMISSION
			("input-flow,v", po::value< std::vector<std::string> >()->composing(), "Input flow files to process.")
			("model,m", po::value< int >(), "Choose the model used by the tool: 0) motion source, 1) object motion.")
			("equatorial-prior", "Add the equatorial prior to the predictions")
			("overlay", po::value< std::string >(), "RGB image to overlay on the saliency map.")
			("gpu", po::value< int >(), "Choose the GPU which will be used by the application.")
			("benchmark", "Show stats on processing time.")
			
			("verbose", "Enable verbose mode to show details on the processing...")
			

#else
			("model,m", po::value< int >(), "Choose the model used by the tool. Options are: [0,1]. Default [0]")
			("temporal-prior", po::value< int >(), "Add the temporal prior to the predictions. The argument set the number of starting point: [1] center only, [2] center and 180 degree, [N] N starting positions. Default: [2]")
			("target-width", po::value< int >(), "Choose the width of the output saliency map. -1 for same as source. Default [2048]")
			("target-height", po::value< int >(), "Choose the height of the output saliency map. -1 for same as source. Default [1024]")
			//("ocl", "Prefer using OpenCL code when available.") // That code performs really slow, you should not use it CPU version of BMS360 is faster... 
#endif // !SUBMISSION
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
		std::cout << "\tV-BMS360 -- Saliency model for 360 videos\n";
		std::cout << "--------------------------------------------------------------------------------\n";
		std::cout << " Contact for this program:\n";
		std::cout << "\tPierre Lebreton, Zhejiang University, China, lebreton.pier@gmail.com\n";
		std::cout << "\tStephan Fremerey, TU Ilmenau, Germany, stephan.fremerey@tu-ilmenau.de\n";
		std::cout << "\tAlexander Raake, TU Ilmenau, Germany, alexander.raake@tu-ilmenau.de\n";
		std::cout << "--------------------------------------------------------------------------------\n";
		std::cout << "\n\n";
		std::cout << desc << "\n";
		return 1;
	}

	



	std::vector<std::string> inputPath;
	std::string outputPath;
	int frame = -1;
	int nbFrames = 1;
	int targetW = 2048;
	int targetH = 1024;

	Saliency360 salient;
	int numberOfFrames = 0;

	if(vm.count("input-video")) {
		VideoFlowGrabber* grabber = new VideoFlowGrabber(vm["input-video"].as<std::string>());
		numberOfFrames = grabber->getFrameCount();
		FlowManager::get()->setFlowGrabber(boost::shared_ptr<FlowGrabber>(grabber));
	}

	
	if (vm.count("input-flow")) {
		std::string overlayPath;

		if(vm.count("overlay")) {
			overlayPath = vm["overlay"].as<std::string>();
		}

		FlowManager::get()->setFlowGrabber(boost::shared_ptr<FlowGrabber>(new FileFlowGrabber(vm["input-flow"].as<std::vector<std::string>>(), overlayPath)));

	} else {
		if(!vm.count("input-video")) {
			std::cerr << "It is required to provide input data. See --help\n";
			return 0;	
		}
	}


	if (vm.count("output-file")) {
		outputPath = vm["output-file"].as< std::string >();
	} 

	if (vm.count("frame")) {
		frame = vm["frame"].as<int>();
	}

	if(vm.count("duration")) {
		nbFrames = vm["duration"].as<int>();
		numberOfFrames = std::min(numberOfFrames, std::max(0, frame) + vm["duration"].as<int>());
	}

	if (vm.count("model")) {

#ifdef SUBMISSION 
		if (vm["model"].as<int>() == 0)
			salient.model = 3;
		if (vm["model"].as<int>() == 1)
			salient.model = 6;

		if (vm["model"].as<int>() != 0 && vm["model"].as<int>() != 1) {
			std::cerr << "[W] Invalid model. --model should be only 0 or 1. Fallback: use mode 0." << std::endl;
			salient.model = 3;
		} 
#else
		salient.model = vm["model"].as<int>();
#endif
	} else {
		salient.model = 3;
	}


	if(vm.count("target-width")) {
		targetW = vm["target-width"].as<int>();
	}

	if(vm.count("target-height")) {
		targetH = vm["target-height"].as<int>();
	}

	if (vm.count("ocl")) {
		salient.ocl = true;
	}

	if (vm.count("verbose")) {
		SalientFeatureFactory::get()->getModel(SalientFeatureFactory::AdaptiveMotionFeature)->setVerbose(true);
	}

	if(vm.count("gpu")) {
		if (!cv::ocl::haveOpenCL()) {
			std::cout << "OpenCL is not available..." << std::endl;
		} else {
			cv::ocl::Context context;
			if (context.create(cv::ocl::Device::TYPE_GPU)) {

				std::cout << context.ndevices() << " GPU devices are detected." << std::endl; //This bit provides an overview of the OpenCL devices you have in your computer
				for (int i = 0; i < context.ndevices(); i++) {

					cv::ocl::Device device = context.device(i);
					std::cout << "name:              " << device.name() << std::endl;
					std::cout << "available:         " << device.available() << std::endl;
					std::cout << "imageSupport:      " << device.imageSupport() << std::endl;
					std::cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << std::endl;
					std::cout << std::endl;
				}

				cv::ocl::Device(context.device(vm["gpu"].as<int>())); //Here is where you change which GPU to use (e.g. 0 or 1)
			}
		}
	}

	if (vm.count("benchmark")) {
		salient.benchmark = true;
	}

	if (vm.count("equatorial-prior")) {
		salient.equatorialPrior = true;
	}

	if (vm.count("temporal-prior")) {
		salient.temporalPrior = vm["temporal-prior"].as<int>();
	} else {
		salient.temporalPrior = 2;
	}

	// ---------------------------------------------------------------------------------------------------
	// Saliency computation
	using namespace std::chrono;

	boost::function<void (const cv::Mat&)> handleOutput;
	
	if(!vm.count("frame") && vm.count("input-video"))
		handleOutput = boost::bind(showSaliency, _1, 10);
	else
		handleOutput = boost::bind(showSaliency, _1, 0);

	FILE *fileBinOut = NULL;

	if(!outputPath.empty()) {
		int idx = outputPath.find_last_of('.');
		
		std::string extension = outputPath.substr(outputPath.find_last_of('.'), 4);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::toupper);

		if (extension == ".BIN") {
			fileBinOut = fopen(outputPath.c_str(), "wb");
			if (fileBinOut == NULL) {
				std::cerr << "[E] Cannot open file for write: " << outputPath << std::endl;
				return -1;
			}

			handleOutput = boost::bind(saveBinarySaliency, _1, fileBinOut);
			salient.enableOverlay = false;
		}
		else {
			if (idx > 0)
				salient.logOutput = outputPath.substr(0, outputPath.find_last_of('.')) + "_color.png";
			else
				salient.logOutput = outputPath + "_color.png";

			handleOutput = boost::bind(saveSaliency, _1, outputPath);
		}
	}


	// compute saliency
	if (frame != -1 && nbFrames == 1) {
		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		std::cout << "[" << frame << "]";

		if (numberOfFrames < frame) {
			std::cout << "[W] Frame requested beyond the end of video. Quit" << std::endl;
			return 0;
		}

		cv::Mat sMap = salient.compute(frame);

		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		std::cout << " in " << time_span.count() << " sec. " << std::endl;

		if(sMap.empty()) return 0;

		if(targetH != -1 && targetW != -1)
			cv::resize(sMap, sMap, cv::Size(targetW, targetH));
		else
			cv::resize(sMap, sMap, FlowManager::get()->getSourceFrameSize());
		handleOutput(sMap);

	} else {
		if (numberOfFrames <= 0) return 0;

		int frIdx = 0;
		if(frame != -1) frIdx = frame;

		std::cout << "[" << frIdx << "/" << numberOfFrames <<  "]";

		high_resolution_clock::time_point t1 = high_resolution_clock::now();

		cv::Mat sMap = salient.compute(frIdx);
		
		if (!sMap.empty()) {
			if (targetH != -1 && targetW != -1)
				cv::resize(sMap, sMap, cv::Size(targetW, targetH));
			else
				cv::resize(sMap, sMap, FlowManager::get()->getSourceFrameSize());
		}
		
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		std::cout << " in " << time_span.count() << " sec. " << std::endl;

		handleOutput(sMap);

		cv::Mat lastMap;
		while (frIdx < numberOfFrames && !sMap.empty()) {
			++frIdx;
			std::cout << "[" << frIdx << "/" << numberOfFrames << "]" << std::flush;

			t1 = high_resolution_clock::now();

			lastMap = sMap.clone();

			sMap = salient.compute(frIdx);

			if (!sMap.empty()) {
				if (targetH != -1 && targetW != -1)
					cv::resize(sMap, sMap, cv::Size(targetW, targetH));
				else
					cv::resize(sMap, sMap, FlowManager::get()->getSourceFrameSize());
			}

			t2 = high_resolution_clock::now();
			time_span = duration_cast<duration<double>>(t2 - t1);
			std::cout << " in " << time_span.count() << " sec. " << std::endl;

			handleOutput(sMap);
		}

		// Perform padding to have the same number of output frames as frames in the video file
		for( ; frIdx < numberOfFrames ; ++frIdx) {
			handleOutput(lastMap);
		}


	}

	std::cout<<std::endl;

	if (fileBinOut != NULL) {
		fclose(fileBinOut);
	}


	return 0;
}

