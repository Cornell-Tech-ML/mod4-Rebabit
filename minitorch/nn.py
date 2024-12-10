from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0, "Height must be divisible by kernel height"
    assert width % kw == 0, "Width must be divisible by kernel width"

    # Calculate the new dimensions after pooling
    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor for pooling
    reshaped = input.contiguous().view(
        batch,
        channel,
        new_height,
        kh,
        new_width,
        kw,
    )

    # Permute to group kernel dimensions together
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Flatten kernel dimensions into a single dimension
    tiled = tiled.contiguous().view(
        batch,
        channel,
        new_height,
        new_width,
        kh * kw,
    )

    return tiled, new_height, new_width



# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on the input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple (kernel_height, kernel_width) specifying the pooling kernel size.

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width) after average pooling.
    """
    # Reshape the input using the tile function
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the average along the last dimension (kernel elements)
    pooled = tiled.mean(dim=-1)

    # Ensure the pooled output matches the expected dimensions
    pooled = pooled.contiguous().view(input.shape[0], input.shape[1], new_height, new_width)

    return pooled

# TODO: Implement for Task 4.4.