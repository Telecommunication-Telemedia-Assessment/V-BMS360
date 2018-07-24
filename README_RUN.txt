// **************************************************************************************************
//
// The MIT License (MIT)
// 
// Copyright (c) 2018 Pierre Lebreton
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



Submission for the ICME Grand Challenge "Salient360! 2018: Visual attention modeling for 360 Images - 2018 edition"

With the archive, two binary can be found:
	- salient-image.exe
	- salient-video.exe


salient-image is the proposed submission to the image track model type 1: Head motion based saliency model. This is the BMS360 image saliency model, see the following link for further information (https://github.com/Telecommunication-Telemedia-Assessment/GBVS360-BMS360-ProSal).
salient-video contains two models which both address the video track, model type 1: Head motion based saliency model. This is the V-BMS360 video saliency model (https://github.com/Telecommunication-Telemedia-Assessment/V-BMS360).


Both model have a manual available through the option --help


---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The expected input for models are Omnidirectional images in EQUIRECTANGULAR format !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------

The usage of the image model is as follow:

.\salient-image.exe -i [input_image.jpg/.png/...] -o prediction.bin 

In this case, the model will write a binary image using floating number (32bit), each pixels are ordered by row (raster scan), and the output frame will have size of 2048x1024. Different size of output image can be obtained using the options --target-width and --target-height. (Note that the model can also output an image in JPEG or PNG if the extension of the output file is .jpg or .png).



---------------------------------------------------------------------------------------------------

The usage of the video models is as follow:

.\salient-video.exe -i [input_video.mp4/.mkv/.avi/...] -o prediction_model_0.bin --model 0
.\salient-video.exe -i [input_video.mp4/.mkv/.avi/...] -o prediction_model_1.bin --model 1

In this case, the model will write a binary video using floating number (32bit), each pixels are ordered by row (raster scan), and the output frame will have size of 2048x1024. Each frame of the video is written one after the other into the binary file. Different size of output image can be obtained using the options --target-width and --target-height. It is also possible to process only a limited number of frames: it can be requested to process only one frame using the option --frame [-f] with the requested frame number as argument. For example, "-f 10" will output the result for the frame 10 only. The model can also process a range of images, using the option --duration [-d]. For example, "-f 10 -d 10" will output 10 frames from the frame 10 to 20. 

In case only a single frame is processed, it is possible to output the saliency map to an output image in PNG or JPEG format by specifying an output file with a .jpeg or .png extension.

Example:

.\salient-video.exe -i [input_video.mp4/.mkv/.avi/...] -o prediction_model_0.png --model 0 -f 10

This will compute only one frame and write the output into a PNG image. 



In the context of the grand challenge, we want to process all the frame, then the simple call: 

.\salient-video.exe -i [input_video.mp4/.mkv/.avi/...] -o prediction_model_0.bin --model 0

Will do the requested task. 




---------------------------------------------------------------------------------------------------

Before running the models, please make sure:

- That the folder "data" is available next to the binary. 
- That you have the VC redist for VS2015 installed (https://www.microsoft.com/en-us/download/details.aspx?id=52685)
- And for the video model in its GPU version, make sure that you have an NVidia graphic card with the latest drivers installed (the model will use CUDA 9.1). The image model do not have such requirements.





---------------------------------------------------------------------------------------------------

Contact for these programs:

      	Pierre Lebreton, Zhejiang University, China, lebreton.pier@gmail.com
        Stephan Fremerey, TU Ilmenau, Germany, stephan.fremerey@tu-ilmenau.de
        Alexander Raake, TU Ilmenau, Germany, alexander.raake@tu-ilmenau.de





