// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/global_pooling_2d_layer.hpp"

namespace fdeep { namespace internal
{

class global_average_pooling_2d_layer : public global_pooling_2d_layer
{
public:
    explicit global_average_pooling_2d_layer(const std::string& name) :
    global_pooling_2d_layer(name)
    {
    }
protected:
    tensor3 pool(const tensor3& in) const override
    {
        tensor3 out(shape3(in.shape().depth_, 1, 1), 0);
        for (std::size_t z = 0; z < in.shape().depth_; ++z)
        {
            float_type val = 0;
            for (std::size_t y = 0; y < in.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < in.shape().width_; ++x)
                {
                    val += in.get(z, y, x);
                }
            }
            out.set(z, 0, 0, val /
                static_cast<float_type>(in.shape().without_depth().area()));
        }
        return out;
    }
};

} } // namespace fdeep, namespace internal
