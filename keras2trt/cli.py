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
    objective: ModelObjective = Option(...),
    in_shape: str = Option(...),
    keras_model: Path = Option(...),
    save_path: Path = Option(...),
):
    conv = ModelConverter()
    conv.convert_keras2trt(
        objective=objective,
        in_shape=in_shape,
        keras_model=keras_model,
        save_path=save_path,
    )


@app.command()
def onnx2trt(
    objective: ModelObjective = Option(...),
    in_shape: str = Option(...),
    onnx_model: Path = Option(...),
    save_path: Path = Option(...),
):
    conv = ModelConverter()
    conv.convert_onnx2trt(
        objective=objective,
        in_shape=in_shape,
        onnx_model=onnx_model,
        save_path=save_path,
    )


@app.command()
def version():
    print(__version__)


if __name__ == "__main__":
    app()
