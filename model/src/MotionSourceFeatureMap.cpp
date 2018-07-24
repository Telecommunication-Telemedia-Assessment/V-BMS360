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



#include "MotionSourceFeatureMap.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef GPU_MODE
    #include <opencv2/cudaarithm.hpp>
#endif



MotionSourceFeatureMap::MotionSourceFeatureMap() : MotionFeatureMap(15) {
    m_tol             = .00000000001f;
    m_salmapmaxsize   = 42;
    m_multires        = 3;
    m_cyclic          = true;
    m_smoothness      = 1;
    m_initDone        = false;
}



cv::Mat MotionSourceFeatureMap::transitionProb(const cv::Mat &flo) const {
    int N = flo.cols*flo.rows;
    cv::Mat p(N, N, CV_32FC1, cv::Scalar(0.f));


    if(m_cyclic) {

        for(int i = 0 ; i < flo.rows ; ++i) {
            for(int j = 0 ; j < flo.cols ; ++j) {
                const cv::Point2f &vp = flo.at<cv::Point2f>(i,j);
                cv::Point2f v = vp;
    
                // if(v.x*v.x+v.y*v.y < 1) continue;
    
                int xd = j+static_cast<int>(v.x);
                int yd = i+static_cast<int>(v.y);
    
                while(xd < 0) xd += flo.cols;
                while(xd >= flo.cols) xd -= flo.cols;
                while(yd < 0) yd += flo.rows;
                while(yd >= flo.rows) yd -= flo.rows;
    
    
                float xr = v.x - static_cast<int>(v.x);
                float yr = v.y - static_cast<int>(v.y);
    
                int src = i*flo.cols+j;
    
                float norm = sqrt(v.x*v.x+v.y*v.y);
                
                int signX = signbit(xr) ? -1 : 1;
                int signY = signbit(yr) ? -1 : 1;
    
                int xdPsignX = xd+signX;
                int ydPsignY = yd+signY;
    
                if(xdPsignX < 0) xdPsignX += flo.cols;
                if(xdPsignX >= flo.cols) xdPsignX -= flo.cols;
                if(ydPsignY < 0) ydPsignY += flo.rows;
                if(ydPsignY >= flo.rows) ydPsignY -= flo.rows;
    
                p.at<float>(src, yd*flo.cols+xd)                += (1.f - fabs(xr))*(1.f - fabs(yr)) * norm;            
                p.at<float>(src, yd*flo.cols+(xdPsignX))        += fabs(xr)*(1.f - fabs(yr)) * norm;
                p.at<float>(src, ydPsignY*flo.cols+xd)          += fabs(yr)*(1.f - fabs(xr)) * norm;
                p.at<float>(src, ydPsignY*flo.cols+xdPsignX)    += fabs(yr)*fabs(xr) * norm;
    
            }
        }

    

    } else {

        for(int i = 0 ; i < flo.rows ; ++i) {

            //float c = std::cos(3.1415926535898f * static_cast<float>(flo.rows / 2 - i) / flo.rows );

            for(int j = 0 ; j < flo.cols ; ++j) {
                const cv::Point2f &vp = flo.at<cv::Point2f>(i,j);
                cv::Point2f v = vp;

                // if(v.x*v.x+v.y*v.y < 1) continue;


                int xd = j+static_cast<int>(v.x);
                int yd = i+static_cast<int>(v.y);

                if(xd < 0 || xd > flo.cols || yd < 0 || yd > flo.rows) continue;

                float xr = v.x - static_cast<int>(v.x);
                float yr = v.y - static_cast<int>(v.y);

                int src = i*flo.cols+j;

                float norm = sqrt(v.x*v.x+v.y*v.y) ;
                
                int signX = signbit(xr) ? -1 : 1;
                int signY = signbit(yr) ? -1 : 1;

                p.at<float>(src, yd*flo.cols+xd)                         += (1.f - fabs(xr))*(1.f - fabs(yr)) * norm;

                if(xd+signX >= 0 && xd+signX < flo.cols) {
                    p.at<float>(src, yd*flo.cols+(xd+signX))             += fabs(xr)*(1.f - fabs(yr)) * norm;
                    
                    if(yd+signY >= 0 && yd+signY < flo.rows) {
                        p.at<float>(src, (yd+signY)*flo.cols+xd)         += fabs(yr)*(1.f - fabs(xr)) * norm;
                        p.at<float>(src, (yd+signY)*flo.cols+(xd+signX)) += fabs(yr)*fabs(xr) * norm;
                    }
                }
            }
        }

    }


    // normalize each column to get transition probabilities
    for(int i = 0 ; i < N ; ++i) {
        float sum = 0;
        for(int k = 0 ; k < N ; ++k) {
            sum += p.at<float>(k,i);
        }

        if(sum > 0.000001f) {
            for(int k = 0 ; k < N ; ++k) {
                p.at<float>(k,i) /= sum;
            }
        }
    }

    return p;

}


// computes the principal eigenvector of a [nm nm] markov matrix
void MotionSourceFeatureMap::principalEigenvectorRaw(const cv::Mat& markovA, float itol, std::vector<float>& AL, int &iteri) const {

	int D = markovA.rows;
	double df = 1.0f;

	cv::Mat v(D, 1, CV_32FC1, cv::Scalar(1.f/D));
	cv::Mat oldv = v.clone();
	cv::Mat oldoldv = v.clone();

	iteri = 0;

	while(df > itol && iteri < 10000 ) { // 

		oldoldv = oldv.clone();		
		oldv = v.clone();

		
		v = markovA * v;
		df = 0.f;

		double sum = 0;
		for(int i = 0 ; i < v.cols ; ++i) {
			double diff = oldv.at<float>(i,0) - v.at<float>(i,0);
			df  += diff*diff;
			sum += v.at<float>(i,0);
		}

		// TODO: This should be a single square root. But the norm 2 differ with matlab? differences due to usage of sparse matrix by MATLAB? 
		// Here several additional iteration are enforced to converge a little bit more... 
		df = std::sqrt(df); 

		++iteri;

		if( sum >= 0 )
			continue;
		else {
			v = oldoldv; 
			break;
		}
	}

	double sum = cv::sum( v )[0];
	for(int i = 0 ; i < D ; ++i) {
		AL[i] = reinterpret_cast<float*>(v.data)[i] / sum;
	}

}


#ifndef GPU_MODE
void MotionSourceFeatureMap::principalEigenvectorRawGPU(const cv::Mat &, float , std::vector<float> &, int &) const {
#else
void MotionSourceFeatureMap::principalEigenvectorRawGPU(const cv::Mat& markovA_CPU, float itol, std::vector<float>& AL, int &iteri) const {

    int D = markovA_CPU.rows;
    double df = 1.0f;

    cv::cuda::GpuMat markovA;
    markovA.upload(markovA_CPU);

    cv::cuda::GpuMat v(D, 1, CV_32FC1, cv::Scalar(1.f / D));
    cv::cuda::GpuMat oldv = v.clone();
    cv::cuda::GpuMat oldoldv = v.clone();

    iteri = 0;
    cv::cuda::GpuMat diffMat;

    while (df > itol && iteri < 10000) { //

        oldoldv = oldv.clone();
        oldv = v.clone();

        //v = markovA * v;
        cv::cuda::gemm(markovA, oldv, 1.f, cv::cuda::GpuMat(), 0.f, v);
        
        cv::cuda::subtract(oldv, v, diffMat);

        cv::Scalar dfS = cv::cuda::sqrSum(diffMat);

        // TODO: This should be a single square root. But the norm 2 differ with matlab? differences due to usage of sparse matrix by MATLAB?
        // Here several additional iteration are enforced to converge a little bit more...
        df = std::sqrt(dfS[0]);

        ++iteri;
    }

    cv::Mat v_CPU;
    v.download(v_CPU);

    
    double sum = cv::sum(v_CPU)[0];
    for (int i = 0; i < D; ++i) {
        AL[i] = reinterpret_cast<float *>(v_CPU.data)[i] / sum;
    }
#endif

}


bool MotionSourceFeatureMap::init() {
    if(m_optFlow.empty()) return false;

    float width  = static_cast<float>(m_optFlow[0].frame.cols);
	float height  = static_cast<float>(m_optFlow[0].frame.rows);
	float scale = static_cast<float>(m_salmapmaxsize) / std::fmax(width,height);

    m_salmapmaxsize_v.clear();
	m_salmapmaxsize_v.push_back(static_cast<int>(round(height * scale)));
    m_salmapmaxsize_v.push_back(static_cast<int>(round(width  * scale)));
    
    m_initDone = true;

    return true;
}


cv::Mat MotionSourceFeatureMap::computeFeature() {
    if(m_optFlow.empty()) return cv::Mat();


    // if(m_optFlow.back().flowProb.empty()) 
    {
        cv::Mat &A = m_optFlow.back().frame;

        cv::Mat fmap;

        // reshape the matrix to avoid too much computation.
        cv::resize(A, fmap, cv::Size(m_salmapmaxsize_v[1], m_salmapmaxsize_v[0]), 0, 0, cv::INTER_AREA);

		// rescale motion vectors such as they scale to the right amplitude
		//cv::divide(fmap, cv::Scalar(static_cast<float>(A.cols) / static_cast<float>(m_salmapmaxsize_v[1]), static_cast<float>(A.rows) / static_cast<float>(m_salmapmaxsize_v[2])), fmap);

        // compute the markov matrix
        m_optFlow.back().flowProb = transitionProb(fmap);

    }

    cv::Mat p = m_optFlow.back().flowProb.clone();

    // cv::Mat &A = m_optFlow.front();
    

    for(int i = static_cast<int>(m_optFlow.size()) - 2 ; i >= 0 ; --i) {
    // for(int i = 1 ; i < static_cast<int>(m_optFlow.size()) ; ++i) {
        cv::Mat lp;
        
        if(m_optFlow[i].flowProb.empty()) {
            cv::Mat fmap;

            // resize the optical flow to avoid too much computation
            cv::resize(m_optFlow[i].frame, fmap, cv::Size(m_salmapmaxsize_v[1], m_salmapmaxsize_v[0]), 0, 0, cv::INTER_AREA);

			// rescale motion vectors such as they scale to the right amplitude
			//cv::divide(fmap, cv::Scalar(static_cast<float>(m_optFlow[i].frame.cols) / static_cast<float>(m_salmapmaxsize_v[1]), static_cast<float>(m_optFlow[i].frame.rows) / static_cast<float>(m_salmapmaxsize_v[2])), fmap);

            // compute the markov matrix
            lp = transitionProb(fmap);


            // Compute the other scales
            int pTwo = 1;
            for(int s = 1 ; s < m_multires ; ++s) {
                pTwo *= 2;

                cv::resize(m_optFlow[i].frame, fmap, cv::Size(m_salmapmaxsize_v[1]/pTwo, m_salmapmaxsize_v[0]/pTwo), 0, 0, cv::INTER_AREA);
                cv::resize(fmap, fmap, cv::Size(m_salmapmaxsize_v[1], m_salmapmaxsize_v[0]));

				// rescale motion vectors such as they scale to the right amplitude
				//cv::divide(fmap, cv::Scalar(static_cast<float>(m_optFlow[i].frame.cols) / static_cast<float>(m_salmapmaxsize_v[1]), static_cast<float>(m_optFlow[i].frame.rows) / static_cast<float>(m_salmapmaxsize_v[2])), fmap);

				fmap *= pTwo;

                lp += transitionProb(fmap) / (sqrt(pTwo)*m_smoothness);

            }

            if(m_multires > 1) {
                lp /= m_multires;
            }

            m_optFlow[i].flowProb = lp;
        } else {
            lp = m_optFlow[i].flowProb;
        }

        // multiply transition matrix to integrate different motion maps
        p = p * lp;

    }

    return p;

}


cv::Mat MotionSourceFeatureMap::computeActivation(const cv::Mat &p) {
    // the eigen vectors    
    std::vector<float> AL(p.rows);
    
    // compute eigen vectors
    int nbIter = 0; 

#ifdef GPU_MODE
    principalEigenvectorRaw(p, m_tol, AL, nbIter);
#else
    principalEigenvectorRaw(p, m_tol, AL, nbIter);
#endif

    // reshape the matrix to a rectangular format 
    cv::Mat master_map(m_salmapmaxsize_v[0], m_salmapmaxsize_v[1], CV_32FC1, cv::Scalar(0.f));
    int curindex = 0;
    for(int ii = 0 ; ii < master_map.rows ; ++ii) {
        for(int jj = 0 ; jj < master_map.cols ; ++jj) {    
            master_map.at<float>(ii,jj) = AL[curindex];
            ++curindex;
        }
    }

    return master_map;
}




cv::Mat MotionSourceFeatureMap::compute(int frame) {
    if(!m_initDone)
        init();

    // Compute feature maps
    cv::Mat p = computeFeature();

    if(p.empty()) return cv::Mat();

    // Compute activation
    cv::Mat master_map = computeActivation(p);

    // Rescale master map to original size
    cv::Mat result;
    cv::resize(master_map, result, cv::Size(m_optFlow[0].frame.cols, m_optFlow[0].frame.rows), 0, 0, cv::INTER_LANCZOS4);

    // apply 360 degree normalization
    scaleSaliency(result);

    double mn, mx;
    cv::minMaxLoc(result, &mn, &mx);
    // If motion source failed, we should return an empty mat... 
    if(std::isnan(mx) || std::isnan(mn)) {
        return cv::Mat(result.size(), CV_32FC1, cv::Scalar(0));
    }

    // normalize the saliency map to 0-1
    cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

    return result;
}



