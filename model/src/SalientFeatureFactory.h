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


#ifndef _SalientFeatureFactory_
#define _SalientFeatureFactory_

#include "SalientFeatureMap.h"
#include "PedestrianDetectFeatureMap.h"
#include "ObjectMotionFeatureMap.h"
#include "MotionSourceFeatureMap.h"
#include "AdaptiveMotionFeatureMap.h"
#include "ImageFeatureMap.h"
#include "TrackedObjectFeatureMap.h"
#include "SpatioTemporalFeatureMap.h"

class SalientFeatureFactory {

private:
    static SalientFeatureFactory *m_This;


    ImageFeatureMap          *m_imageFeature;
    MotionSourceFeatureMap   *m_motionSourceFeature;
    ObjectMotionFeatureMap   *m_objectMotionFeature;
    AdaptiveMotionFeatureMap *m_adaptiveMotionFeature;
    TrackedObjectFeatureMap  *m_trackedObjectFeature;
    PedestrianFeatureMap     *m_pedestrianFeature;
    SpatioTemporalFeatureMap *m_spatioTemporalFeature;


public:

    enum FeatureMap {
        ImageFeature = 0,
        MotionSourceFeature,
        ObjectMotionFeature,
        AdaptiveMotionFeature,
        PedestrianFeature,
        TrackedObjectFeature,
        SpatioTemporalFeature
    } ;

    static SalientFeatureFactory *get();


    SalientFeatureMap *getModel(FeatureMap model);
    ~SalientFeatureFactory();


private:
    SalientFeatureFactory() : m_imageFeature(NULL), m_motionSourceFeature(NULL), m_objectMotionFeature(NULL), m_adaptiveMotionFeature(NULL), m_trackedObjectFeature(NULL), m_pedestrianFeature(NULL), m_spatioTemporalFeature(NULL) {};

}; 


#endif


