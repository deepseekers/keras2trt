# import pytest
from pathlib import Path

import onnx
import tensorflow as tf
from keras2trt.enums import ModelObjective
from keras2trt.model_converter import ModelConverter


def test_convert_keras2onnx_local():
    conv = ModelConverter()
    conv.convert_keras2onnx(
        keras_model=Path("models/inceptionv3"),
        opset=13,
        save_path=Path("models/cli_test/tf2onnx_local"),
    )


def test_convert_keras2onnx_binary():
    model_path = Path("models/inceptionv3")
    keras_model = tf.keras.models.load_model(model_path, compile=False)
    conv = ModelConverter()
    conv.convert_keras2onnx(
        keras_model=keras_model,
        opset=13,
        save_path=Path("models/cli_test/tf2onnx_binary"),
    )


def test_convert_keras2trt_local():
    conv = ModelConverter()
    conv.convert_keras2trt(
        objective=ModelObjective.DETECTION,
        in_shape="(1,256,256,3)",
        keras_model=Path("models/inceptionv3"),
        save_path=Path("models/cli_test/tf2trt_local"),
    )


def test_convert_keras2trt_binary():
    model_path = Path("models/inceptionv3")
    keras_model = tf.keras.models.load_model(model_path, compile=False)
    conv = ModelConverter()
    conv.convert_keras2trt(
        objective=ModelObjective.DETECTION,
        in_shape="(1,256,256,3)",
        keras_model=keras_model,
        save_path=Path("models/cli_test/tf2trt_binary"),
    )


def test_convert_onnx2trt_local():
    conv = ModelConverter()
    conv.convert_onnx2trt(
        objective=ModelObjective.DETECTION,
        in_shape="(1,256,256,3)",
        onnx_model=Path("models/inceptionv3.onnx"),
        save_path=Path("models/cli_test/onnx2trt_local"),
    )


def test_convert_onnx2trt_binary():
    model_path = Path("models/inceptionv3.onnx")
    onnx_model = onnx.load(model_path)
    conv = ModelConverter()
    conv.convert_onnx2trt(
        objective=ModelObjective.DETECTION,
        in_shape="(1,256,256,3)",
        onnx_model=onnx_model,
        save_path=Path("models/cli_test/onnx2trt_binary"),
    )
