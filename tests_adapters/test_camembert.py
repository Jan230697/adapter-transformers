import unittest
from tests_adapters.methods.test_config_union import ConfigUnionAdapterTest

from transformers import CamembertConfig
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    UniPELTTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .composition.test_parallel import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class CamembertAdapterTestBase(AdapterTestBase):
    config_class = CamembertConfig
    config = make_config(
        CamembertConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        vocab_size=50265,
    )
    tokenizer_name = "camembert-base"


@require_torch
class CamembertAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ConfigUnionAdapterTest,
    CamembertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class CamembertClassConversionTest(
    ModelClassConversionTestMixin,
    CamembertAdapterTestBase,
    unittest.TestCase,
):
    pass
