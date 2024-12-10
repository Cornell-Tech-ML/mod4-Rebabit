from typing import Tuple, Optional

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


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
    pooled = pooled.contiguous().view(
        input.shape[0], input.shape[1], new_height, new_width
    )

    return pooled


# TODO: Implement for Task 4.4.
class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Max function."""
        ctx.save_for_backward(t1, int(dim.item()))
        return t1.f.max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward method for max."""
        t1, dim = ctx.saved_values
        max_mask = argmax(t1, dim)
        return grad_output * max_mask, 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Returns the maximum value of the tensor along the specified dimension."""
    if dim is None:
        return Max.apply(
            input.contiguous().view(input.size),
            input._ensure_tensor(0),
        )
    return Max.apply(input, input._ensure_tensor(dim))


def argmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the argmax as a one-hot tensor along the specified dimension.

    Args:
    ----
        input: Input tensor.
        dim: Dimension along which to compute the argmax.

    Returns:
    -------
        A tensor with the same shape as `input`, where the maximum values
        along `dim` are marked as 1.0, and others are 0.0.

    """
    if dim is None:
        max_vals = input.f.max_reduce(input.contiguous().view(input.size), 0)
    else:
        max_vals = input.f.max_reduce(input, dim)
    return input == max_vals


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along the specified dimension.

    Args:
    ----
        input: Input tensor.
        dim: Dimension along which to compute the softmax.

    Returns:
    -------
        A tensor with the same shape as `input`, representing the softmax probabilities.

    """
    # Shift input by its max for numerical stability
    shifted_input = input - max(input, dim)

    # Compute exponential values
    exp_vals = shifted_input.exp()

    # Normalize by the sum along the specified dimension
    return exp_vals / exp_vals.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along the specified dimension.

    Args:
    ----
        input: Input tensor.
        dim: Dimension along which to compute the logsoftmax.

    Returns:
    -------
        A tensor with the same shape as `input`, representing the logsoftmax probabilities.

    """
    # Shift input by its max for numerical stability
    shifted_input = input - max(input, dim)

    # Compute log-sum-exp
    log_sum_exp = (shifted_input.exp()).sum(dim).log()

    # Subtract log-sum-exp from shifted input
    return shifted_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling on the input tensor.

    Args:
    ----
        input: Input tensor of shape (batch, channel, height, width).
        kernel: Tuple (kernel_height, kernel_width) specifying pooling kernel size.

    Returns:
    -------
        A tensor of reduced dimensions after applying max pooling.

    """
    batch, channel, height, width = input.shape

    # Reshape the input for pooling using the tile function
    tiled, new_height, new_width = tile(input, kernel)

    # Apply max reduction along the last dimension (kernel elements)
    pooled = max(tiled, dim=-1)

    return pooled.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float = 0.5, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor using random noise.

    Args:
    ----
        input: Input tensor.
        rate: Dropout probability (default 0.5).
        ignore: Apply dropout only during training.

    Returns:
    -------
        A tensor with dropped elements set to zero.

    """
    if ignore:
        return input
    elif rate == 0.0:
        return input
    elif rate >= 1.0:
        return input.zeros(input.shape)
    else:
        random_noise = rand(input.shape, backend=input.backend)
        mask = random_noise > rate
        return input * mask / (1 - rate)  # Scale the remaining values
