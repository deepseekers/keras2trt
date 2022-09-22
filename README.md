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
  version
```

### keras2onnx

```
Usage: keras2trt keras2onnx [OPTIONS]

Options:
  --opset INTEGER     [default: 15]
  --keras-model PATH  [required]
  --save-path PATH    [required]
  --help              Show this message and exit.
```

#### Example

```
keras2trt keras2onnx --keras-model models/inceptionv3 --opset 13 --save-path models/tf2onnx
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
  --opset INTEGER     [default: 15]
  --in-shape TEXT
  --min-shape TEXT
  --opt-shape TEXT
  --max-shape TEXT
  --keras-model PATH  [required]
  --save-path PATH    [required]
  --help              Show this message and exit.
```

#### Example

```
keras2trt keras2trt --opset 17 --in-shape "(1,256,256,3)" --keras-model models/inceptionv3 --save-path models/keras2trt.trt

keras2trt keras2trt --opset 15 --min-shape "(5,256,256,3)" --opt-shape "(15,256,256,3)" --max-shape "(30,256,256,3)" --keras-model models/inceptionv3 --save-path models/keras2trt
```

- if --save-path does not have a suffix, ".engine" suffix will be added to the saved TensorRT engine.
- All min_shape, opt_shape, and max_shape need to be set for dynamic batch size.
- If none of the shape arguments is set, the batch size will be set as 1.
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
  --in-shape TEXT
  --min-shape TEXT
  --opt-shape TEXT
  --max-shape TEXT
  --onnx-model PATH  [required]
  --save-path PATH   [required]
  --help             Show this message and exit.
```

#### Example

```
keras2trt onnx2trt --in-shape "(1,256,256,3)" --onnx-model models/tf2onnx.onnx --save-path models/onnx2trt

keras2trt onnx2trt --min-shape "(5,256,256,3)" --opt-shape "(15,256,256,3)" --max-shape "(30,256,256,3)" --onnx-model models/tf2onnx.onnx --save-path models/onnx2trt
```

- if --save-path does not have a suffix, ".engine" suffix will be added to the saved TensorRT engine.
- All min_shape, opt_shape, and max_shape need to be set for dynamic batch size.
- If none of the shape arguments is set, the batch size will be set as 1.
