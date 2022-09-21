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
&& pip install nvidia-tensorrt==8.4.1.5
```

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
```

### keras2onnx

```
Usage: keras2trt keras2onnx [OPTIONS]

Options:
  --opset INTEGER    [default: 15]
  --model-path PATH  [required]
  --save-path PATH   [required]
  --help             Show this message and exit.
```

#### Example

```
keras2trt keras2onnx --model-path models/inceptionv3 --opset 13 --save-path models/tf2onnx
```

- if --save-path does not have a suffix, ".onnx" suffix will be added to the saved ONNX model.
- Model path is a keras saved_model directory.

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

Options:
  --objective [classification|detection|segmentation]
                                  [required]
  --in-shape TEXT                 [required]
  --model-path PATH               [required]
  --save-path PATH                [required]
  --help                          Show this message and exit.
```

#### Example

```
keras2trt keras2trt --objective classification --in-shape "(1,256,256,3)" --save-path models/keras2trt.trt --model-path models/inceptionv3
```

- if --save-path does not have a suffix, ".engine" suffix will be added to the saved TensorRT engine.
- Model path is a keras saved_model directory.

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

Options:
  --objective [classification|detection|segmentation]
                                  [required]
  --in-shape TEXT                 [required]
  --model-path PATH               [required]
  --save-path PATH                [required]
  --help                          Show this message and exit.
```

#### Example

```
keras2trt onnx2trt --objective classification --in-shape "(1,256,256,3)" --save-path models/onnx2trt --model-path models/tf2onnx.onnx
```

- if --save-path does not have a suffix, ".engine" suffix will be added to the saved TensorRT engine.
