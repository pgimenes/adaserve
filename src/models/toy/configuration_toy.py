from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ToyConfig(PretrainedConfig):

    def __init__(
        self,
        hidden_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
