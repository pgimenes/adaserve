import torch
import torch.nn as nn


class ManualLinear(nn.Linear):
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
        # if self.bias is None:
        #     return torch.bmm(input, self.weight.transpose(0, 1))

        return torch.addmm(
            self.bias,
            input,
            torch.transpose(self.weight, 0, 1),
        )


class ManualBatchLinear(nn.Linear):
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
        # if self.bias is None:
        #     return torch.bmm(input, self.weight.transpose(0, 1))

        return torch.baddbmm(
            self.bias,
            input,
            torch.transpose(self.weight, 0, 1).expand(input.size(0), -1, -1),
        )


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
