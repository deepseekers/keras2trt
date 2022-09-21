import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
import tensorflow as tf

import tensorrt as trt

from .config import logger

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class ModelConverter:
    def __init__(
        self, onnx_model, config, batch_size: int, trt_engine_dir: str, model_name: str
    ) -> None:
        self.logger = logger
        self.config = config
        self.__onnx_model = onnx_model
        self.__batch_size = batch_size
        self.__trt_engine_dir = trt_engine_dir
        self.__engine_path = self.__get_engine_path(model_name)

    def __get_engine_path(self, model_name: str):
        return f"{self.__trt_engine_dir}/{model_name}.engine"

    def __get_img_shape(self) -> Tuple:
        if self.config.img_shape is not None:
            w, h = self.config.img_shape
        else:
            w = self.config.max_img_shape_test
            h = self.config.max_img_shape_test
            w = (
                w - (w % self.config.model_shape_multiple)
                if (w % self.config.model_shape_multiple) > 0
                else w
            )
            h = (
                h - (h % self.config.model_shape_multiple)
                if (h % self.config.model_shape_multiple) > 0
                else h
            )
            max_shape = max(w, h)
            w, h = (max_shape, max_shape)

        return w, h

    def __onnx_to_trt(self, onnx_path: str) -> trt.tensorrt.ICudaEngine:
        """This is the function to create the TensorRT engine

        Args:
            onnx_path (str): Path to onnx_file.

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
            if self.config.objective == "detection":
                profile = builder.create_optimization_profile()
                width, height = self.__get_img_shape()
                shape = (
                    self.__batch_size,
                    width,
                    height,
                    3,
                )
                profile.set_shape("input", min=shape, opt=shape, max=shape)
                builder_config.add_optimization_profile(profile)
            parser.parse_from_file(onnx_path)
            input_shape = list(network.get_input(0).shape)
            input_shape[0] = self.__batch_size
            network.get_input(0).shape = input_shape
            engine = builder.build_engine(network, builder_config)

            return engine

    def convert_onnx2trt(self) -> trt.tensorrt.ICudaEngine:
        """This function converts ONNX model to TensorRT engine

        Returns:
            trt.tensorrt.ICudaEngine: TensorRT engine
        """
        if Path(self.__engine_path).exists():
            return self.__load_engine()

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            model_dir = f"{tmp_dir}/model.onnx"
            self.logger.info(f"Saving onnx model: {model_dir}")
            onnx.save(self.__onnx_model, model_dir)
            engine = self.__onnx_to_trt(onnx_path=model_dir)
            self.logger.info(f"Saving serialized TRT engine: {self.__engine_path}")
            with open(self.__engine_path, "wb") as f:
                f.write(engine.serialize())

        return engine
