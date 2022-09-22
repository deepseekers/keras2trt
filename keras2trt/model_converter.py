import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from ast import literal_eval
from pathlib import Path
from typing import Any, List, Union

import onnx
import tensorflow as tf
import tensorrt as trt
import tf2onnx

from .enums import ModelObjective, OnnxOpset
from .logging import logger

TRT_LOGGER = trt.Logger(min_severity=trt.ILogger.ERROR)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class ModelConverter(object):
    def __init__(self) -> None:
        self.__logger = logger

    def __parse_shapes(self, *args):
        shapes = [literal_eval(a) if isinstance(a, str) else None for a in args]
        if len({len(s) for s in shapes if s is not None}) > 1:
            raise ValueError(f"Shape lengths need to be the same.\n{shapes}")
        if all(shapes[1:]):
            for min_s, opt_s, max_s in zip(*shapes[1:]):
                if min_s <= opt_s and opt_s <= max_s:
                    continue
                raise ValueError(
                    f"Dynamic shape dimensions need to be set correctly. \n0 <= min[i] <= opt[i] <= max[i] for all i"
                )
        else:
            self.__logger.warn(
                f"For dynamic batch size all min_shape, opt_shape and max_shape need to be set."
            )

        return shapes

    def __convert_keras_to_onnx(
        self, keras_model, opset: int
    ) -> onnx.onnx_ml_pb2.ModelProto:
        self.__logger.info(f"Converting Keras model to ONNX.")
        spec = (tf.TensorSpec(keras_model.input.shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            keras_model, input_signature=spec, opset=opset
        )
        model_proto.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "?"
        return model_proto

    def __load_keras_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Keras model not found: {model_path}")
        self.__logger.info(f"Loading Keras model: {model_path.name}.")
        keras_model = tf.keras.models.load_model(model_path, compile=False)
        return keras_model

    def convert_keras2onnx(
        self,
        keras_model: Union[Path, Any],
        opset: int,
        save_path: Path,
    ) -> onnx.onnx_ml_pb2.ModelProto:
        """This function converts Keras model to ONNX model.

        Args:
            keras_model (Union[Path, Any]): Path to keras model.
            opset (int): ONNX model opset.
            save_path (Path): Path to save the ONNX model.

        Returns:
            onnx.onnx_ml_pb2.ModelProto: ONNX model.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(keras_model, Path):
            keras_model = self.__load_keras_model(keras_model)
        onnx_model = self.__convert_keras_to_onnx(keras_model=keras_model, opset=opset)
        if not save_path.suffix:
            save_path = save_path.parent / (save_path.name + ".onnx")
        self.__logger.info(f"Saving onnx model: {save_path}")
        onnx.save(onnx_model, save_path)

        return onnx_model

    def __onnx_to_trt(
        self,
        onnx_path: str,
        in_shape: List[int],
        min_shape: List[int],
        opt_shape: List[int],
        max_shape: List[int],
    ) -> trt.tensorrt.ICudaEngine:
        """This is the function to convert ONNX model to TensorRT Engine

        Args:
            onnx_path (str): Path to onnx_file.
            objective (ModelObjective): Objective of the model (classification, segmentation, detection).
            in_shape (List[int]): Model input shape (batch_size, width, height, channel).

        Returns:
            trt.tensorrt.ICudaEngine: TensorRT engine
        """
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as builder_config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:
            builder_config.max_workspace_size = 256 << 20
            builder_config.flags = 1 << int(trt.BuilderFlag.FP16)

            parser.parse_from_file(onnx_path)
            input_shape = list(network.get_input(0).shape)

            if input_shape[0] != -1:
                assert min_shape == opt_shape == max_shape

            if all([min_shape, opt_shape, max_shape]) and input_shape[0] == -1:
                profile = builder.create_optimization_profile()
                profile.set_shape("input", min=min_shape, opt=opt_shape, max=max_shape)
                builder_config.add_optimization_profile(profile)
            elif in_shape is not None:
                input_shape[0] = in_shape[0]
                network.get_input(0).shape = input_shape
            else:
                input_shape[0] = 1
                network.get_input(0).shape = input_shape

            engine = builder.build_engine(network, builder_config)

            return engine

    def __save_trt_engine(
        self,
        in_shape: List[int],
        min_shape: List[int],
        opt_shape: List[int],
        max_shape: List[int],
        onnx_model: Union[onnx.onnx_ml_pb2.ModelProto, Path],
        save_path: Path,
    ) -> trt.tensorrt.ICudaEngine:
        if isinstance(onnx_model, onnx.onnx_ml_pb2.ModelProto):
            onnx_model_path = Path(save_path).parent / f"{Path(save_path).stem}.onnx"
            onnx.save(onnx_model, onnx_model_path)
        elif isinstance(onnx_model, Path):
            if not onnx_model.exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_model}")
            onnx_model_path = onnx_model

        self.__logger.info(
            f"Converting ONNX model '{onnx_model_path.name}' to TensorRT engine."
        )
        trt_engine = self.__onnx_to_trt(
            onnx_path=str(onnx_model_path),
            in_shape=in_shape,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
        )
        if not save_path.suffix:
            save_path = save_path.parent / (save_path.name + ".engine")
        self.__logger.info(f"Saving serialized TRT engine: {save_path}")
        with open(save_path, "wb") as f:
            f.write(trt_engine.serialize())

        return trt_engine

    def convert_keras2trt(
        self,
        opset: int,
        in_shape: str,
        min_shape: str,
        opt_shape: str,
        max_shape: str,
        keras_model: Union[Path, Any],
        save_path: Path,
    ) -> trt.tensorrt.ICudaEngine:
        """This function converts Tensorflow model to TensorRT engine.

        Args:
            opset (int): Opset value for ONNX conversion.
            in_shape (str): Model input shape (batch_size, width, height, channel).
            keras_model (Union[Path, Any]): Tensorflow saved_model path.
            save_path (Path): Path to save the TensorRT engine.

        Returns:
            trt.tensorrt.ICudaEngine: TensorRT engine.
        """

        in_shape, min_shape, opt_shape, max_shape = self.__parse_shapes(
            in_shape, min_shape, opt_shape, max_shape
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(keras_model, Path):
            keras_model = self.__load_keras_model(keras_model)
        onnx_model = self.__convert_keras_to_onnx(keras_model=keras_model, opset=opset)

        return self.__save_trt_engine(
            onnx_model=onnx_model,
            in_shape=in_shape,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            save_path=save_path,
        )

    def convert_onnx2trt(
        self,
        in_shape: str,
        min_shape: str,
        opt_shape: str,
        max_shape: str,
        onnx_model: Union[onnx.onnx_ml_pb2.ModelProto, Path],
        save_path: Path,
    ) -> trt.tensorrt.ICudaEngine:
        """This function converts ONNX model to TensorRT engine.

        Args:
            objective (ModelObjective): Objective of the model (classification, segmentation, detection).
            in_shape (str): Model input shape (batch_size, width, height, channel).
            onnx_model (Union[onnx.onnx_ml_pb2.ModelProto, Path]): Onnx model path.
            save_path (Path): Path to save the TensorRT engine.

        Returns:
            trt.tensorrt.ICudaEngine: TensorRT engine.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        in_shape, min_shape, opt_shape, max_shape = self.__parse_shapes(
            in_shape, min_shape, opt_shape, max_shape
        )
        # self.__logger.info(in_shape, min_shape, opt_shape, max_shape)

        return self.__save_trt_engine(
            onnx_model=onnx_model,
            in_shape=in_shape,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            save_path=save_path,
        )
