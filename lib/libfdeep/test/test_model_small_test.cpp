// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <fdeep/fdeep.hpp>

TEST_CASE("test_model_small_test, load_model")
{
    const auto model = fdeep::load_model("../test_model_small.json",
        true, fdeep::cout_logger, static_cast<fdeep::float_type>(0.00001));
    const auto multi_inputs = fplus::generate<std::vector<fdeep::tensor3s>>(
        [&]() -> fdeep::tensor3s {return model.generate_dummy_inputs();},
        10);
    model.predict_multi(multi_inputs, false);
    model.predict_multi(multi_inputs, true);
}
