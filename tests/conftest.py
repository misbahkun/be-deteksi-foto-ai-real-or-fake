import pathlib

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

MODEL_PATH = pathlib.Path("models/model.onnx")
_INPUT_DIM = 3 * 256 * 256


def _build_dummy_onnx() -> bytes:
    rng = np.random.default_rng(seed=0)
    W = rng.standard_normal((2, _INPUT_DIM)).astype(np.float32)
    B = np.zeros(2, dtype=np.float32)

    W_init = numpy_helper.from_array(W, name="W")
    B_init = numpy_helper.from_array(B, name="B")

    flatten = helper.make_node("Flatten", inputs=["input"], outputs=["flat"], axis=1)
    gemm = helper.make_node("Gemm", inputs=["flat", "W", "B"], outputs=["output"], transB=1)

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 256, 256])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph(
        [flatten, gemm],
        "dummy_airealnet",
        [input_vi],
        [output_vi],
        initializer=[W_init, B_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture(scope="module")
def dummy_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(_build_dummy_onnx())
    yield MODEL_PATH
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


@pytest.fixture(scope="module")
def client(dummy_model):  # noqa: ARG001
    from app.main import app

    with TestClient(app) as c:
        yield c
