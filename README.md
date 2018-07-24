# V-BMS360: A video extention to the BMS360 image saliency model 

In this project, it is studied how existing image saliency models for head motion prediction designed for omnidirectional images can be applied to videos. A new model called V-BMS360 which extends the BMS360 model is presented. Due to the specific properties of omnidirectional videos, new key features had to be introduced: the first one is the introduction of a temporal prior that accounts for the delay needed by the users to explore the content during the first seconds of the videos. The second key novel idea is the consideration of camera motion and its consequences on the exploration of the visual scenes. It was translated into two key aspects new features: the "motion surroundness" and the "motion source", and an informed pooling allowing to provide different strengths to different motion-based features based on a camera-motion analysis. All of these contribute to the performance of the new saliency model called V-BMS360. 


# Context 

This project was intiated to address the ICME2018 Grand Challenge on Visual attention for 360 images and videos. The proposed model will only address one particular case: Head motion-based saliency map for 360 videos. 


# Binaries

In the release section of GitHub, different set of binaries can be found. One is the "GPU version" of the V-BMS360 model, and one is the CPU version. It should be noted that at this time, only the optical flow estimation is performed on GPU. Therefore, you may observe that the "GPU version" of the model may still heavily use CPU. The CPU version of the algorithm uses a different algorithm than the GPU version, therefore you may observe some differences between the two version (although they should not be large). 

In the context of the ICME grand challenge, it is the GPU version which was submitted.  



# Build from source

To compile the project, only two dependencies are required: 

   - Eigen (tested with 3.3.3, any version beyond should work)

   - boost (>= 1.63)

   - OpenCV (>= 3.0) For the CPU build, the contributions libraries are needed in order to get the dense optical flow algorithms.  


## UNIX system

run  `make`.  

## Windows: 

A visual studio solution file is provided (tested using the 2015 Community edition). A complete set of all depending libraries can be found at the following URL: 
Unzip the folder Library next to the clone of the repository. Then, open the visual studio solution file, and you can compile the project (The path to the external library folder is set relatively to the root folder). 



# Citation

The paper related to this work has been accepted within the proceedings of ICME 2018. In case you use this work, please cite:

```
@InProceedings{lebreton2018VBMS360,
  title={V-BMS360: A video extention to the BMS360 image saliency model},
  author={Lebreton, Pierre and Fremerey, Stephan and Raake, Alexander},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME) 2018, Grand Challenge Track},
  year={2018}
}
```





For the evaluation scripts used in this study, please refer and cite:

```
@article{Gutierrez2018Toolbox, 
  title={Toolbox and Dataset for the Development of Saliency and Scanpath Models for Omnidirectional / 360{$^{\circ}$} Still Images}, 
  author={Guti{\'e}rrez, Jes{\'u}s and David, Erwan and Rai, Yashas and Le Callet, Patrick 
}, 
  journal={Signal Processing: Image Communication}, 
  volume={??}, 
  pages={??--??}, 
  year={2018}, 
  publisher={Elsevier} 
}
```



# Shipped dependencies

The following dependencies are shipped in the `lib` folder:


   - A modified version of the original BMS implementation from Jianming Zhang   (http://cs-people.bu.edu/jmzhang/BMS/BMS.html). Please refer to the readme in the corresponding folder for further information.

   - frugally-deep: A VERY cool library allowing to run models built and trained using Keras/tensorflow from C++ using header only libraries (https://github.com/Dobiasd/frugally-deep)  

   - gnomonic library: A library allowing to perform projections from equirectangular to rectilinear domain. This was used mostly for research purpose and is not involved in the final software.  


# License

The MIT License (MIT)

Copyright (c) 2018 Pierre Lebreton

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.