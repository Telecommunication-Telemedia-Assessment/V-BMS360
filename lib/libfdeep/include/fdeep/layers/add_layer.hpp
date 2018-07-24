// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

namespace fdeep { namespace internal
{

class add_layer : public layer
{
public:
    explicit add_layer(const std::string& name)
        : layer(name)
    {
    }
protected:
    tensor3s apply_impl(const tensor3s& input) const override
    {
        return {sum_tensor3s(input)};
    }
};

} } // namespace fdeep, namespace internal
