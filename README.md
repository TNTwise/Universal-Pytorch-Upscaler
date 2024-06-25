### Universal Pytorch Upscaler
```
usage: UniversalTorchUpscaler.py [-h] [-i INPUT] [-o OUTPUT] [-t TILESIZE]
                                 [-m MODELPATH] -n MODELNAME [-c] [-f F]
                                 [--half] [--bfloat16] [-e EXPORT]

Upscale any image, with most torch models, using spandrel.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input image path (jpg/png/webp) or directory
  -o OUTPUT, --output OUTPUT
                        output image path (jpg/png/webp) or directory
  -t TILESIZE, --tilesize TILESIZE
                        tile size (>=32/0=auto, default=0)
  -m MODELPATH, --modelPath MODELPATH
                        folder path to the pre-trained models. default=models
  -n MODELNAME, --modelName MODELNAME
                        model name (include extension)
  -c, --cpu             use only CPU for upscaling, instead of cuda.
                        default=auto
  -f F                  output image format (jpg/png/webp, default=ext/png)
  --half                half precision, only works with NVIDIA RTX 20 series
                        and above.
  --bfloat16            like half precision, but more intesive. This can be
                        used with a wider range of models than half.
  -e EXPORT, --export EXPORT
                        Export PyTorch models to ONNX and NCNN. Options:
                        (onnx/ncnn)

```
