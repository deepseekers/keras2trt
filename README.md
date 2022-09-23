# keras2trt

Keras2TRT is a cli tool that is capable of converting keras saved_models to TensorRT engine. Currently supported conversions are:

- Keras to ONNX
- ONNX to TensorRT
- Keras to TensorRT

**_NOTE:_** The CLI is tested converting image segmentation, classification and detection models.

## Requirements

The following packages need to be installed to use the cli.

```bash
pip install nvidia-pyindex==1.0.9 \
&& pip install nvidia-tensorrt==8.4.2.4
```

**_NOTE:_** nvidia-tensorrt==8.4.2.4 is compatible with nvcr.io/nvidia/tritonserver:22.08-py3 docker image.

## Installation

```
pip install keras2trt
```

## Usage

```
Usage: keras2trt [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.

Commands:
  keras2onnx
  keras2trt
  onnx2trt
  version
```

### keras2onnx

```
Usage: keras2trt keras2onnx [OPTIONS]

  Convert Keras model to ONNX model.

  - if --save-path does not have a suffix, ".onnx" suffix will be added to the
  saved ONNX model.

Options:
  --opset INTEGER     ONNX model opset.  [default: 15]
  --keras-model PATH  Path to the Keras model.  [required]
  --save-path PATH    Path to save the TensorRT engine.  [required]
  --help              Show this message and exit.
```

#### Example

```
keras2trt keras2onnx --keras-model models/inceptionv3 --opset 13 --save-path models/tf2onnx
```

Model path is a keras saved_model directory.

```
models/inceptionv3
├── assets
├── keras_metadata.pb
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```

### keras2trt

```
Usage: keras2trt keras2trt [OPTIONS]

  Convert Keras model to tensorrt engine.

  - If --save-path does not have a suffix, ".engine" suffix will be added to
  the saved TensorRT engine.

  - All min_shape, opt_shape, and max_shape need to be set for dynamic batch
  size.

  - If none of the shape arguments is set, the batch size will be set as 1.

Options:
  --opset INTEGER     ONNX model opset.  [default: 15]
  --in-shape TEXT     Model input shape.
  --min-shape TEXT    Minimum input shape for dynamic batch.
  --opt-shape TEXT    Optimal input shape for dynamic batch.
  --max-shape TEXT    Maximum input shape for dynamic batch.
  --keras-model PATH  Path to the Keras model.  [required]
  --save-path PATH    Path to save the TensorRT engine.  [required]
  --help              Show this message and exit.
```

#### Example

```
keras2trt keras2trt --opset 17 --in-shape "(1,256,256,3)" --keras-model models/inceptionv3 --save-path models/keras2trt.trt

keras2trt keras2trt --opset 15 --min-shape "(5,256,256,3)" --opt-shape "(15,256,256,3)" --max-shape "(30,256,256,3)" --keras-model models/inceptionv3 --save-path models/keras2trt
```

Model path is a keras saved_model directory.

```
models/inceptionv3
├── assets
├── keras_metadata.pb
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```

### onnx2trt

```
Usage: keras2trt onnx2trt [OPTIONS]

  Convert ONNX model to tensorrt engine.

  - If --save-path does not have a suffix, ".engine" suffix will be added to
  the saved TensorRT engine.

  - All min_shape, opt_shape, and max_shape need to be set for dynamic batch
  size.

  - If none of the shape arguments is set, the batch size will be set as 1.

Options:
  --in-shape TEXT    Model input shape.
  --min-shape TEXT   Minimum input shape for dynamic batch.
  --opt-shape TEXT   Optimal input shape for dynamic batch.
  --max-shape TEXT   Maximum input shape for dynamic batch.
  --onnx-model PATH  Path to the ONNX model.  [required]
  --save-path PATH   Path to save the TensorRT engine.  [required]
  --help             Show this message and exit.
```

#### Example

```
keras2trt onnx2trt --in-shape "(1,256,256,3)" --onnx-model models/tf2onnx.onnx --save-path models/onnx2trt

keras2trt onnx2trt --min-shape "(5,256,256,3)" --opt-shape "(15,256,256,3)" --max-shape "(30,256,256,3)" --onnx-model models/tf2onnx.onnx --save-path models/onnx2trt
```
