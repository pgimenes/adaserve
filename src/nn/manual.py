import torch
import torch.nn as nn


class ManualLinear2D(nn.Linear):
    """
    This layer replaces the Linear layer from PyTorch with explicit
    invocations to the required kernels: torch.mm or torch.addmm,
    depending on the presence of bias. This is useful for mapping to
    sharding strategies in the autosharding pass.

    The input to the forward pass must be a 2d matrix.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default: True.
        device (str, optional): The desired device of the layer. Default: None.
        dtype (str, optional): The desired data type of the layer. Default: None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        weight = torch.transpose(
            self.weight,
            0,
            1,
        )

        # Invoke required kernel according to bias
        if self.bias is None:
            out = torch.mm(
                input,
                weight,
            )
        else:
            out = torch.addmm(
                self.bias,
                input,
                weight,
            )

        return out


class ManualBatchLinear(nn.Linear):
    """
    This layer replaces the Linear layer from PyTorch with explicit
    invocations to the required kernels: torch.bmm or torch.baddbmm,
    depending on the presence of bias. This is useful for mapping to
    sharding strategies in the autosharding pass.

    The input to the forward pass must be a 3D tensor.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default: True.
        device (str, optional): The desired device of the layer. Default: None.
        dtype (str, optional): The desired data type of the layer. Default: None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):

        # Ensure input and weights are a 3D batch
        # reshaped = input.reshape((-1,) + input.shape[-2:])

        weight = torch.transpose(
            self.weight,
            0,
            1,
        ).expand(
            input.shape[0],
            -1,
            -1,
        )

        # Invoke required kernel according to bias
        if self.bias is None:
            out = torch.bmm(
                input,
                weight,
            )
        else:
            out = torch.baddbmm(
                self.bias,
                input,
                weight,
            )

        # TO DO: this doesn't work with fx tracing, so output shape
        # will be wrong if the input is > 3D
        # new_shape = input.shape[:-2] + out.shape[-2:]
        # return torch.reshape(
        #     out,
        #     new_shape,
        # )

        return out


class ManualLayerNorm(nn.LayerNorm):
    """
    Same implementation as nn.LayerNorm, but used for tracing.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def __repr__(self):
        return f"ManualLayerNorm({self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, bias={self.bias})"
