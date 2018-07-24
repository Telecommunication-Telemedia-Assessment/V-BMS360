#!/usr/bin/env python3
"""Convert a Keras model to frugally-deep format.
"""

import base64
import datetime
import json
import sys

import numpy as np

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

STORE_FLOATS_HUMAN_READABLE = False


def write_text_file(path, text):
    """Write a string to a file"""
    with open(path, "w") as text_file:
        print(text, file=text_file)


def arr3_to_channels_first_format(arr):
    """Convert a 3-tensor for channels_first"""
    assert len(arr.shape) == 3
    assert K.image_data_format() == 'channels_last'
    return np.swapaxes(np.swapaxes(arr, 2, 1), 1, 0)


def arr_as_arr3(arr):
    """Convert a n-tensor to a 3-tensor"""
    depth = len(arr.shape)
    if depth == 1:
        return arr.reshape(arr.shape[0], 1, 1)
    if depth == 2:
        return arr.reshape(arr.shape[1], 1, arr.shape[0])
    if depth == 3:
        return arr3_to_channels_first_format(arr)
    if depth == 4 and arr.shape[0] in [None, 1]:
        return arr3_to_channels_first_format(arr.reshape(arr.shape[1:]))
    else:
        raise ValueError('invalid number of dimensions')


def show_tensor3(tens):
    """Serialize 3-tensor to a dict"""
    values = tens.flatten()
    return {
        'shape': tens.shape,
        'values': encode_floats(values)
    }


def show_test_data_as_3tensor(arr):
    """Serialize model test data"""
    return show_tensor3(arr_as_arr3(arr))


def gen_test_data(model):
    """Generate data for model verification test."""

    def set_shape_idx_0_to_1(shape):
        """Change first element in tuple to 1."""
        shape_lst = list(shape)
        shape_lst[0] = 1
        shape = tuple(shape_lst)
        return shape

    data_in = list(map(lambda l: np.random.random(
        set_shape_idx_0_to_1(l.input_shape)).astype(np.float32),
        model._input_layers))

    start_time = datetime.datetime.now()
    data_out = model.predict(data_in)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print('Forward pass took {} s.'.format(duration.total_seconds()))

    return {
        'inputs': list(map(show_test_data_as_3tensor, data_in)),
        'outputs': list(map(show_test_data_as_3tensor, data_out))
    }


def split_every(size, seq):
    """Split a sequence every seq elements."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def encode_floats(arr):
    """Serialize a sequence of floats."""
    if STORE_FLOATS_HUMAN_READABLE:
        return values
    return list(split_every(1024, base64.b64encode(arr).decode('ascii')))


def prepare_filter_weights_conv_2d(weights):
    """Change dimension order of 2d filter weights to the one used in fdeep"""
    return np.swapaxes(np.swapaxes(np.swapaxes(
        weights, 0, 2), 1, 3), 0, 1).flatten()


def prepare_filter_weights_conv_1d(weights):
    """Change dimension order of 1d filter weights to the one used in fdeep"""
    return np.swapaxes(weights, 0, 2).flatten()


def show_conv_1d_layer(layer):
    """Serialize Conv1D layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 3
    weights_flat = prepare_filter_weights_conv_1d(weights[0])
    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 3
    assert layer.input_shape[0] is None
    result = {
        'weights': encode_floats(weights_flat)
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_conv_2d_layer(layer):
    """Serialize Conv2D layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 4
    weights_flat = prepare_filter_weights_conv_2d(weights[0])
    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 4
    assert layer.input_shape[0] is None
    result = {
        'weights': encode_floats(weights_flat)
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_conv_2d_transpose_layer(layer):
    """Serialize Conv2DTranspose layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 4
    weights_flat = prepare_filter_weights_conv_2d(weights[0])
    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 4
    assert layer.input_shape[0] is None
    result = {
        'weights': encode_floats(weights_flat)
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_separable_conv_2d_layer(layer):
    """Serialize SeparableConv2D layer to dict"""
    weights = layer.get_weights()
    assert layer.depth_multiplier == 1
    assert len(weights) == 2 or len(weights) == 3
    assert len(weights[0].shape) == 4
    assert len(weights[1].shape) == 4

    # probably incorrect for depth_multiplier > 1?
    slice_weights = prepare_filter_weights_conv_2d(weights[0])
    stack_weights = prepare_filter_weights_conv_2d(weights[1])

    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 4
    assert layer.input_shape[0] is None
    result = {
        'slice_weights': encode_floats(slice_weights),
        'stack_weights': encode_floats(stack_weights),
    }
    if len(weights) == 3:
        bias = weights[2]
        result['bias'] = encode_floats(bias)
    return result


def show_batch_normalization_layer(layer):
    """Serialize batch normalization layer to dict"""
    moving_mean = K.get_value(layer.moving_mean)
    moving_variance = K.get_value(layer.moving_variance)
    result = {}
    result['moving_mean'] = encode_floats(moving_mean)
    result['moving_variance'] = encode_floats(moving_variance)
    if layer.center:
        beta = K.get_value(layer.beta)
        result['beta'] = encode_floats(beta)
    if layer.scale:
        gamma = K.get_value(layer.gamma)
        result['gamma'] = encode_floats(gamma)
    return result


def is_flat_shape(shape):
    """Check if only one dimension of shape is > 1"""
    if shape[0] != None:
        return False
    if len(shape) == 2:
        return True
    if len(shape) == 3:
        return False
    return shape[1] == 1 and shape[2] == 1


def show_dense_layer(layer):
    """Serialize dense layer to dict"""
    assert is_flat_shape(layer.input_shape)
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 2
    weights_flat = weights[0].flatten()
    result = {
        'weights': encode_floats(weights_flat)
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def get_dict_keys(d):
    """Return keys of a dictionary"""
    return [key for key in d]


def merge_two_disjunct_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy.
    No Key is allowed to be present in both dictionaries.
    """
    assert set(get_dict_keys(x)).isdisjoint(get_dict_keys(y))
    z = x.copy()
    z.update(y)
    return z


def is_ascii(some_string):
    """Check if a string only contains ascii characters"""
    try:
        some_string.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True


def get_all_weights(model):
    """Serialize all weights of the models layers"""
    show_layer_functions = {
        'Conv1D': show_conv_1d_layer,
        'Conv2D': show_conv_2d_layer,
        'Conv2DTranspose': show_conv_2d_transpose_layer,
        'SeparableConv2D': show_separable_conv_2d_layer,
        'BatchNormalization': show_batch_normalization_layer,
        'Dense': show_dense_layer
    }
    result = {}
    layers = model.layers
    for layer in layers:
        layer_type = type(layer).__name__
        if layer_type in ['Model', 'Sequential']:
            result = merge_two_disjunct_dicts(result, get_all_weights(layer))
        else:
            show_func = show_layer_functions.get(layer_type, None)
            name = layer.name
            assert is_ascii(name)
            if name in result:
                raise ValueError('duplicate layer name ' + name)
            if show_func:
                result[name] = show_func(layer)
    return result


def convert_sequential_to_model(model):
    """Convert a sequential model to the underlying functional format"""
    if type(model).__name__ == 'Sequential':
        name = model._name
        if hasattr(model, '_inbound_nodes'):
            inbound_nodes = model._inbound_nodes
        elif hasattr(model, 'inbound_nodes'):
            inbound_nodes = model.inbound_nodes
        else:
            assert False
        model = model.model
        model._name = name
        if hasattr(model, '_inbound_nodes'):
            model._inbound_nodes = inbound_nodes
        elif hasattr(model, 'inbound_nodes'):
            model.inbound_nodes = inbound_nodes
    assert model._input_layers
    assert model.layers
    for i in range(len(model.layers)):
        if type(model.layers[i]).__name__ in ['Model', 'Sequential']:
            model.layers[i] = convert_sequential_to_model(model.layers[i])
    return model


def offset_conv2d_eval(depth, padding, x):
    """Perform a conv2d on x with a given padding"""
    kernel = K.variable(value=np.array([[[[1]] + [[0]] * (depth - 1)]]),
        dtype='float32')
    return K.conv2d(x, kernel, strides=(3, 3), padding=padding)


def offset_sep_conv2d_eval(depth, padding, x):
    """Perform a separable conv2d on x with a given padding"""
    depthwise_kernel = K.variable(value=np.array([[[[1]] * depth]]),
                                  dtype='float32')
    pointwise_kernel = K.variable(value=np.array([[[[1]] + [[0]] * (depth - 1)]]),
        dtype='float32')
    return K.separable_conv2d(x, depthwise_kernel,
                              pointwise_kernel, strides=(3, 3), padding=padding)


def conv2d_offset_max_pool_eval(_, padding, x):
    """Perform a max pooling operation on x"""
    return K.pool2d(x, (1, 1), strides=(3, 3), padding=padding, pool_mode='max')


def conv2d_offset_average_pool_eval(_, padding, x):
    """Perform an average pooling operation on x"""
    return K.pool2d(x, (1, 1), strides=(3, 3), padding=padding, pool_mode='avg')


def check_operation_offset(depth, eval_f, padding):
    """Check if backend used an offset while placing the filter
    e.g. during a convolution.
    TensorFlow is inconsistent in doing so depending
    on the type of operation, the used device (CPU/GPU) and the input depth.
    """
    in_arr = np.array([[[[i] * depth for i in range(6)]]])
    input_data = K.variable(value=in_arr, dtype='float32')
    output = eval_f(depth, padding, input_data)
    result = K.eval(output).flatten().tolist()
    assert result in [[0, 3], [1, 4]]
    return result == [1, 4]


def get_shapes(tensor3s):
    """Return shapes of a list of tensors"""
    return [t['shape'] for t in tensor3s]


def convert(in_path, out_path):
    """Convert any Keras model to the frugally-deep model format."""

    assert K.backend() == "tensorflow"
    assert K.floatx() == "float32"
    assert K.image_data_format() == 'channels_last'

    print('loading {}'.format(in_path))
    model = load_model(in_path)

    # Force creation of underlying functional model.
    # see: https://github.com/fchollet/keras/issues/8136
    # Loss and optimizer type do not matter, since to don't train the model.
    model.compile(loss='mse', optimizer='sgd')

    model = convert_sequential_to_model(model)
    test_data = gen_test_data(model)

    json_output = {}
    json_output['architecture'] = json.loads(model.to_json())

    json_output['image_data_format'] = K.image_data_format()
    for depth in range(1, 3, 1):
        json_output['conv2d_valid_offset_depth_' + str(depth)] =\
            check_operation_offset(depth, offset_conv2d_eval, 'valid')
        json_output['conv2d_same_offset_depth_' + str(depth)] =\
            check_operation_offset(depth, offset_conv2d_eval, 'same')
        json_output['separable_conv2d_valid_offset_depth_' + str(depth)] =\
            check_operation_offset(depth, offset_sep_conv2d_eval, 'valid')
        json_output['separable_conv2d_same_offset_depth_' + str(depth)] =\
            check_operation_offset(depth, offset_sep_conv2d_eval, 'same')
    json_output['max_pooling_2d_valid_offset'] =\
        check_operation_offset(1, conv2d_offset_max_pool_eval, 'valid')
    json_output['max_pooling_2d_same_offset'] =\
        check_operation_offset(1, conv2d_offset_max_pool_eval, 'same')
    json_output['average_pooling_2d_valid_offset'] =\
        check_operation_offset(1, conv2d_offset_average_pool_eval, 'valid')
    json_output['average_pooling_2d_same_offset'] =\
        check_operation_offset(1, conv2d_offset_average_pool_eval, 'same')
    json_output['input_shapes'] = get_shapes(test_data['inputs'])
    json_output['output_shapes'] = get_shapes(test_data['outputs'])
    json_output['tests'] = [test_data]
    json_output['trainable_params'] = get_all_weights(model)

    print('writing {}'.format(out_path))
    write_text_file(out_path, json.dumps(
        json_output, allow_nan=False, indent=2, sort_keys=True))


def main():
    """Parse command line and convert model."""

    usage = 'usage: [Keras model in HDF5 format] [output path]'

    if len(sys.argv) != 3:
        print(usage)
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    convert(in_path, out_path)


if __name__ == "__main__":
    main()
