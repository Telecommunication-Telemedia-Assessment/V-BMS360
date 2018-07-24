// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

namespace fdeep { namespace internal
{

class softmax_layer : public activation_layer
{
public:
    explicit softmax_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor3 transform_input(const tensor3& input) const override
    {
        // Get unnormalized values of exponent function.
        const auto ex = [](float_type x) -> float_type
        {
            return std::exp(x);
        };
        auto output = transform_tensor3(ex, input);

        // Softmax function is applied along channel dimension.
        for (size_t y = 0; y < input.shape().height_; ++y)
        {
            for (size_t x = 0; x < input.shape().width_; ++x)
            {
                // Get the sum of unnormalized values for one pixel.
                // We are not using Kahan summation, since the number
                // of object classes is usually quite small.
                float_type sum = 0.0f;
                for (size_t z_class = 0; z_class < input.shape().depth_; ++z_class)
                {
                    sum += output.get(z_class, y, x);
                }
                // Divide the unnormalized values of each pixel by the stacks sum.
                for (size_t z_class = 0; z_class < input.shape().depth_; ++z_class)
                {
                    output.set(z_class, y, x, output.get(z_class, y, x) / sum);
                }
            }
        }
        return output;
    }
};

} } // namespace fdeep, namespace internal
