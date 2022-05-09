import numpy as np
import pytest

import aesara
from aesara.tensor.math import _allclose
from aesara.tensor.signal import conv
from aesara.tensor.type import TensorType, dtensor3, dtensor4, dvector, matrix
from tests import unittest_tools as utt


_ = pytest.importorskip("scipy.signal")


def validate(image_shape, filter_shape, out_dim, verify_grad=True):

    image_dim = len(image_shape)
    filter_dim = len(filter_shape)
    input = TensorType("float64", [False] * image_dim)()
    filters = TensorType("float64", [False] * filter_dim)()

    bsize = image_shape[0]
    if image_dim != 3:
        bsize = 1
    nkern = filter_shape[0]
    if filter_dim != 3:
        nkern = 1

    # AESARA IMPLEMENTATION ############
    # we create a symbolic function so that verify_grad can work
    def sym_conv2d(input, filters):
        return conv.conv2d(input, filters)

    output = sym_conv2d(input, filters)
    assert output.ndim == out_dim
    aesara_conv = aesara.function([input, filters], output)

    # initialize input and compute result
    image_data = np.random.random(image_shape)
    filter_data = np.random.random(filter_shape)
    aesara_output = aesara_conv(image_data, filter_data)

    # REFERENCE IMPLEMENTATION ############
    out_shape2d = np.array(image_shape[-2:]) - np.array(filter_shape[-2:]) + 1
    ref_output = np.zeros(tuple(out_shape2d))

    # reshape as 3D input tensors to make life easier
    image_data3d = image_data.reshape((bsize,) + image_shape[-2:])
    filter_data3d = filter_data.reshape((nkern,) + filter_shape[-2:])
    # reshape aesara output as 4D to make life easier
    aesara_output4d = aesara_output.reshape(
        (
            bsize,
            nkern,
        )
        + aesara_output.shape[-2:]
    )

    # loop over mini-batches (if required)
    for b in range(bsize):

        # loop over filters (if required)
        for k in range(nkern):

            image2d = image_data3d[b, :, :]
            filter2d = filter_data3d[k, :, :]
            output2d = np.zeros(ref_output.shape)
            for row in range(ref_output.shape[0]):
                for col in range(ref_output.shape[1]):
                    output2d[row, col] += (
                        image2d[
                            row : row + filter2d.shape[0],
                            col : col + filter2d.shape[1],
                        ]
                        * filter2d[::-1, ::-1]
                    ).sum()

            assert _allclose(aesara_output4d[b, k, :, :], output2d)

if __name__ == "__main__":
    validate((1, 4, 5), (2, 2, 3), out_dim=4, verify_grad=True)
