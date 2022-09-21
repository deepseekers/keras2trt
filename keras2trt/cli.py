from pathlib import Path
from typer import Option, Typer

from .model_converter import ModelConverter
from .models import ModelObjective

app = Typer()


@app.command()
def keras2onnx(
    opset: int = Option(15),
    model_path: Path = Option(...),
    save_path: Path = Option(...),
):
    conv = ModelConverter()
    conv.convert_keras2onnx(
        opset=opset,
        model_path=model_path,
        save_path=save_path,
    )


@app.command()
def keras2trt(
    objective: ModelObjective = Option(...),
    in_shape: str = Option(...),
    model_path: Path = Option(...),
    save_path: Path = Option(...),
):
    conv = ModelConverter()
    conv.convert_keras2trt(
        objective=objective,
        in_shape=in_shape,
        model_path=model_path,
        save_path=save_path,
    )


@app.command()
def onnx2trt(
    objective: ModelObjective = Option(...),
    in_shape: str = Option(...),
    model_path: Path = Option(...),
    save_path: Path = Option(...),
):
    conv = ModelConverter()
    conv.convert_onnx2trt(
        objective=objective,
        in_shape=in_shape,
        model_path=model_path,
        save_path=save_path,
    )


if __name__ == "__main__":
    app()
