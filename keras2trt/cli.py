from pathlib import Path

from typer import Option, Typer

from .enums import ModelObjective
from .model_converter import ModelConverter
from .version import __version__

app = Typer()


@app.command()
def keras2onnx(
    opset: int = Option(15, help="ONNX model opset."),
    keras_model: Path = Option(..., help="Path to the Keras model."),
    save_path: Path = Option(..., help="Path to save the TensorRT engine."),
):
    """Convert Keras model to ONNX model.

    - if --save-path does not have a suffix, ".onnx" suffix will be added to the saved ONNX model.
    """
    conv = ModelConverter()
    conv.convert_keras2onnx(
        opset=opset,
        keras_model=keras_model,
        save_path=save_path,
    )


@app.command()
def keras2trt(
    opset: int = Option(15, help="ONNX model opset."),
    in_shape: str = Option(None, help="Model input shape."),
    min_shape: str = Option(None, help="Minimum input shape for dynamic batch."),
    opt_shape: str = Option(None, help="Optimal input shape for dynamic batch."),
    max_shape: str = Option(None, help="Maximum input shape for dynamic batch."),
    keras_model: Path = Option(..., help="Path to the Keras model."),
    save_path: Path = Option(..., help="Path to save the TensorRT engine."),
):
    """
    Convert Keras model to TensorRT engine.

    - If --save-path does not have a suffix, ".engine" suffix will be added to the saved TensorRT engine.

    - All min_shape, opt_shape, and max_shape need to be set for dynamic batch size.

    - If none of the shape arguments is set, the batch size will be set as 1.
    """
    conv = ModelConverter()
    conv.convert_keras2trt(
        opset=opset,
        in_shape=in_shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        keras_model=keras_model,
        save_path=save_path,
    )


@app.command()
def onnx2trt(
    in_shape: str = Option(None, help="Model input shape."),
    min_shape: str = Option(None, help="Minimum input shape for dynamic batch."),
    opt_shape: str = Option(None, help="Optimal input shape for dynamic batch."),
    max_shape: str = Option(None, help="Maximum input shape for dynamic batch."),
    onnx_model: Path = Option(..., help="Path to the ONNX model."),
    save_path: Path = Option(..., help="Path to save the TensorRT engine."),
):
    """Convert ONNX model to TensorRT engine.

    - If --save-path does not have a suffix, ".engine" suffix will be added to the saved TensorRT engine.

    - All min_shape, opt_shape, and max_shape need to be set for dynamic batch size.

    - If none of the shape arguments is set, the batch size will be set as 1.
    """
    conv = ModelConverter()
    conv.convert_onnx2trt(
        in_shape=in_shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        onnx_model=onnx_model,
        save_path=save_path,
    )


@app.command()
def version():
    print(__version__)


if __name__ == "__main__":
    app()
