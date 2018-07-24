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



#include "SalientFeatureFactory.h"

SalientFeatureFactory* SalientFeatureFactory::m_This = NULL;


SalientFeatureFactory *SalientFeatureFactory::get() {
    if(m_This == NULL) {
        m_This = new SalientFeatureFactory();

    }

    return m_This;
}

SalientFeatureFactory::~SalientFeatureFactory() {
    if(m_imageFeature != NULL)              delete m_imageFeature;
    if(m_motionSourceFeature != NULL)       delete m_motionSourceFeature;
    if(m_objectMotionFeature != NULL)       delete m_objectMotionFeature;
    if(m_adaptiveMotionFeature != NULL)     delete m_adaptiveMotionFeature;
    if(m_trackedObjectFeature != NULL)      delete m_trackedObjectFeature;
    if(m_pedestrianFeature != NULL)         delete m_pedestrianFeature;
}

SalientFeatureMap *SalientFeatureFactory::getModel(FeatureMap model) {

    switch(model) {
        case ImageFeature:
            if(m_imageFeature == NULL)
                m_imageFeature = new ImageFeatureMap();
            return m_imageFeature;
        
        case MotionSourceFeature:
            if(m_motionSourceFeature == NULL)
                m_motionSourceFeature = new MotionSourceFeatureMap();
            return m_motionSourceFeature;

        case ObjectMotionFeature:
            if(m_objectMotionFeature == NULL)
                m_objectMotionFeature = new ObjectMotionFeatureMap();
            return m_objectMotionFeature;

        case AdaptiveMotionFeature:
            if(m_adaptiveMotionFeature == NULL)
                m_adaptiveMotionFeature = new AdaptiveMotionFeatureMap();
            return m_adaptiveMotionFeature;

        case TrackedObjectFeature:
            if(m_trackedObjectFeature == NULL)
                m_trackedObjectFeature = new TrackedObjectFeatureMap();
            return m_trackedObjectFeature;

        case PedestrianFeature:
            if(m_pedestrianFeature == NULL)
                m_pedestrianFeature = new PedestrianFeatureMap();
            return m_pedestrianFeature;

        case SpatioTemporalFeature:
            if(m_spatioTemporalFeature == NULL)
                m_spatioTemporalFeature = new SpatioTemporalFeatureMap();
            return m_spatioTemporalFeature;

        default:
            return NULL;
    }

}