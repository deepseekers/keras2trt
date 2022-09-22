from pathlib import Path

from typer import Option, Typer

from .enums import ModelObjective
from .model_converter import ModelConverter
from .version import __version__

app = Typer()


@app.command()
def keras2onnx(
    opset: int = Option(15),
    keras_model: Path = Option(...),
    save_path: Path = Option(...),
):
    conv = ModelConverter()
    conv.convert_keras2onnx(
        opset=opset,
        keras_model=keras_model,
        save_path=save_path,
    )


@app.command()
def keras2trt(
    opset: int = Option(15),
    in_shape: str = Option(None),
    min_shape: str = Option(None),
    opt_shape: str = Option(None),
    max_shape: str = Option(None),
    keras_model: Path = Option(...),
    save_path: Path = Option(...),
):
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
    in_shape: str = Option(None),
    min_shape: str = Option(None),
    opt_shape: str = Option(None),
    max_shape: str = Option(None),
    onnx_model: Path = Option(...),
    save_path: Path = Option(...),
):
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
