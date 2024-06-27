import torch
import argparse
import os
from PIL import Image
import onnx


from src.Util import is_image
from src.UpscaleTorch import UpscalePytorchImage
from src.UpscaleNCNN import UpscaleNCNNImage
from src.ConvertModels import ConvertModels


class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        self.checkArguments()
        if self.args.export == None:
            if self.args.backend == 'pytorch':
                self.pytorchRenderSingleImage(self.args.input)
            if self.args.backend == 'ncnn':
                self.ncnnRenderSingleImage(self.args.input)
        elif self.args.export.lower().strip() == "onnx":
            self.exportModelAsONNX()
        elif self.args.export.lower().strip() == "ncnn":
            self.exportModelAsNCNN()
    def returnDevice(self):
        if not self.args.cpu:
            return "cuda" if torch.cuda.is_available() else "cpu"

    

    def loadImage(self, image: str) -> Image:
        return Image.open(image)

    def exportModelAsNCNN(self):
        x = ConvertModels(
            modelName=self.args.modelName,
            pathToModel=self.fullModelPathandName(),
            inputFormat="pytorch",
            outputFormat="ncnn",
            device="cpu",
            half=True,
            bfloat16=False
        )
        x.convertModel()

    def exportModelAsONNX(self):
        x = ConvertModels(
            modelName=self.args.modelName,
            pathToModel=self.fullModelPathandName(),
            inputFormat="pytorch",
            outputFormat="onnx",
            device=self.returnDevice(),
            half=self.args.half,
            bfloat16=self.args.bfloat16,
            opset=17,
        )
        x.convertModel()

    def pytorchRenderSingleImage(self, imagePath: str):
        image = self.loadImage(imagePath)
        upscale = UpscalePytorchImage(
            modelPath=self.args.modelPath,
            modelName=self.args.modelName,
            device=self.returnDevice(),
            half=self.args.half,
            bfloat16=self.args.bfloat16,
        )

        imageTensor = upscale.imageToTensor(image)
        upscaledTensor = (
            upscale.renderImage(imageTensor)  # render image, tile if necessary
            if self.args.tilesize == 0
            else upscale.renderTiledImage(imageTensor, self.args.tilesize)
        )
        upscaledImage = upscale.tensorToImage(upscaledTensor)
        upscale.saveImage(upscaledImage, self.args.output)

    def ncnnRenderSingleImage(self, imagePath: str):
        upscale = UpscaleNCNNImage(
            modelPath=self.args.modelPath,
            modelName=self.args.modelName,
        )
        upscaledImage = upscale.renderImage(fullImagePath=imagePath)
        upscale.saveImage(upscaledImage,self.args.output)

    def handleArguments(self) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_

        """
        parser = argparse.ArgumentParser(
            description="Upscale any image, with most torch models, using spandrel."
        )

        parser.add_argument(
            "-i",
            "--input",
            default=None,
            help="input image path (jpg/png/webp) or directory",
        )
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            help="output image path (jpg/png/webp) or directory",
        )
        parser.add_argument(
            "-t",
            "--tilesize",
            help="tile size (>=32/0=auto, default=0)",
            default=0,
            type=int,
        )
        parser.add_argument(
            "-b",
            "--backend",
            help="backend used to upscale image. (pytorch/ncnn, default=pytorch)",
            default="pytorch",
        )
        parser.add_argument(
            "-m",
            "--modelPath",
            help="folder path to the pre-trained models. default=models",
            default="models",
        )
        parser.add_argument(
            "-n", "--modelName", required=True, help="model name (include extension)"
        )
        parser.add_argument(
            "-c",
            "--cpu",
            help="use only CPU for upscaling, instead of cuda. default=auto",
            action="store_true",
        )
        parser.add_argument(
            "-f", help="output image format (jpg/png/webp, default=ext/png)"
        )
        parser.add_argument(
            "--half",
            help="half precision, only works with NVIDIA RTX 20 series and above.",
            action="store_true",
        )
        parser.add_argument(
            "--bfloat16",
            help="like half precision, but more intesive. This can be used with a wider range of models than half.",
            action="store_true",
        )

        parser.add_argument(
            "-e",
            "--export",
            help="Export PyTorch models to ONNX and NCNN. Options: (onnx/ncnn)",
            default=None,
        )
        return parser.parse_args()

    def fullModelPathandName(self):
        return os.path.join(self.args.modelPath, self.args.modelName)

    def checkArguments(self):
        assert isinstance(self.args.tilesize, int)
        assert (
            self.args.tilesize > 31 or self.args.tilesize == 0
        ), "Tile size must be greater than 32 (inclusive), or 0."
        # put error messages here
        modelNotFoundError = f"Model {self.fullModelPathandName()} does not exist."

        if self.args.export == None:
            if self.args.input == None:
                raise os.error("Tried to upscale without Input Image!")
            if self.args.output == None:
                raise os.error("Tried to upscale without Output Image!")

            self.isDir = os.path.isdir(self.args.input)

            if self.args.input == self.args.output:
                raise os.error("Input and output cannot be the same image.")
            
            # checks for pytorch model existing, user input requires .pth extension
            if self.args.backend == 'pytorch':
                if not os.path.exists(self.fullModelPathandName()):
                    raise os.error(modelNotFoundError)
                
            # checking if ncnn model exists, user input excludes .bin or .param
            if self.args.backend == 'ncnn':
                if not os.path.exists(self.fullModelPathandName()  + '.bin') or not os.path.exists(self.fullModelPathandName() + '.param'):
                    raise os.error(modelNotFoundError)
                
            if not self.isDir:  # Executed if it is not rendering an image directory
                if not os.path.isfile(self.args.input):
                    raise os.error("Input File/Directory does not exist.")

                if not is_image(self.args.input):
                    raise os.error("Not a valid image File/Directory.")

            else:
                if len(os.listdir(self.args.input)) == 0:
                    raise os.error("Empty Input Directory!")

                if not os.path.isdir(self.args.output):
                    raise os.error(
                        "Output must be a directory if input is a directory."
                    )

        if self.args.half and self.args.bfloat16:
            raise os.error("Cannot use half and bfloat16 at the same time!")


if __name__ == "__main__":
    HandleApplication()
