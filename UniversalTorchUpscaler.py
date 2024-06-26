from spandrel import ImageModelDescriptor, ModelLoader
import sys
import torch
import argparse
import os
from torchvision import transforms
from PIL import Image
import math
import onnx
import onnxruntime
import pnnx
from io import BytesIO
# tiling code permidently borrowed from https://github.com/chaiNNer-org/spandrel/issues/113#issuecomment-1907209731
cwd = os.getcwd()

def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Check if it's a valid image file
        return True
    except (IOError, SyntaxError):
        return False


def loadModelWithScale(
    modelPath: str, half: bool = False, bfloat16: bool = False, device: str = "cuda"
):
    model = ModelLoader().load_from_file(modelPath)
    assert isinstance(model, ImageModelDescriptor)
    # get model attributes
    scale = model.scale

    if device == "cpu":
        model.eval().cpu()
    if device == "cuda":
        model.eval().cuda()
        if half:
            model.half()
        if bfloat16:
            model.bfloat16()
    return model, scale


class UpscaleImage:
    def __init__(
        self,
        modelPath: str = "models",
        modelName: str = "",
        device="cuda",
        tile_pad=10,
        half=False,
        bfloat16=False,
    ):
        self.half = half
        self.bfloat16 = bfloat16
        self.tile_pad = tile_pad

        path = os.path.join(modelPath, modelName)
        self.setDevice(device)
        self.model, self.scale = loadModelWithScale(path, half, bfloat16, device)

    def setDevice(
        self,
        device: str = "cuda",
    ):
        self.device = device

    def imageToTensor(self, image: Image) -> torch.Tensor:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL image to tensor
            ]
        )

        imageTensor = (
            transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        if self.half:
            return imageTensor.half()
        if self.bfloat16:
            return imageTensor.bfloat16()

        return imageTensor

    def tensorToImage(self, image: torch.Tensor) -> Image:
        transform = transforms.ToPILImage()
        return transform(image.squeeze(0).float())

    @torch.inference_mode()
    def renderImage(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(image)

    @torch.inference_mode()
    def renderImagesInDirectory(self, dir):
        pass

    def getScale(self, modelPath: str):
        return self.scale

    def renderTiledImage(
        self, image: torch.Tensor, tile_size: int = 32
    ) -> torch.Tensor:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = image.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = image.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = image[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                with torch.no_grad():
                    output_tile = self.renderImage(input_tile)

                print(f"\tTile {tile_idx}/{tiles_x * tiles_y}")

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
        return output


class ConvertModels:
    def __init__(
        self,
        modelName: str,
        pathToModel: str,
        inputFormat: str = "pytorch",
        outputFormat: str = "onnx",
        ncnnConversionMethod: str = "onnx",
        device: str = "cpu",
        half: bool = False,
        bfloat16: bool = False,
        opset: int = 18,
        onnxDynamicAxess: dict = None,
    ):
        self.modelName = modelName
        self.pathToModel = pathToModel
        self.basepath = os.path.dirname(pathToModel)
        self.inputFormat = inputFormat
        self.outputFormat = outputFormat
        self.ncnnConversionMethod = ncnnConversionMethod
        self.device = device
        self.opset = opset
        self.half = half
        self.bfloat16 = bfloat16
        self.onnxDynamicAxes = onnxDynamicAxess

    def convertModel(self):
        self.input = self.handleInput()
        # load model
        self.model, scale = loadModelWithScale(
            self.pathToModel, self.half, self.bfloat16, self.device
        )
        if self.outputFormat == "onnx":
            self.convertPyTorchToONNX()
        if self.outputFormat == "ncnn":
            self.convertPytorchToNCNN()
    def handleInput(self):
        x = torch.rand(1, 3, 256, 256)
        if self.half:
            x = x.half()
        if self.bfloat16:
            x = x.bfloat16()
        if self.device == "cuda":
            x = x.cuda()
        return x

    def generateONNXOutputName(self) -> str:
        if self.half:
            return f"{self.modelName}_op{self.opset}_half.onnx"
        if self.bfloat16:
            return f"{self.modelName}_op{self.opset}_bfloat16.onnx"
        return f"{self.modelName}_op{self.opset}.onnx"

    def convertPyTorchToONNX(self):
        model = self.model.model
        state_dict = model.state_dict()
        model.eval()
        model.load_state_dict(state_dict, strict=True)
        with torch.inference_mode():
            torch.onnx.export(
                model,
                self.input,
                self.generateONNXOutputName(),
                opset_version=self.opset,
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
                dynamic_axes=self.onnxDynamicAxes,
            )
    def fixNCNNParamInput(self,paramFile):
        """
        replaces in0 with data and out0 with output in a ncnn param file
        """
        newParamFile = []
        with open(paramFile, 'r') as f:
            for line in f.readlines():
                line = line.replace('in0','data')
                line = line.replace('out0','output')
                newParamFile.append(line)
        with open(paramFile, 'w') as f:
            f.writelines(newParamFile)

    def convertPytorchToNCNN(self):
        """
        Takes in a pytorch model, and uses JIT tracing with PNNX to convert it to end.
        This method removed unnecessary files, and fixes the param file to be compadible with most NCNN appliacitons.
        """
        model = self.model.model
        model.eval()
        input = torch.rand(1,3,256,256)
        jitTracedModelLocation = self.pathToModel+'.pt'
        jitTracedModel = torch.jit.trace(model,input)
        jitTracedModel.save(jitTracedModelLocation)

        pnnxBinLocation = self.pathToModel +'.pnnx.bin'
        pnnxParamLocation = self.pathToModel +'.pnnx.param'
        pnnxPythonLocation = self.pathToModel +'_pnnx.py'
        pnnxOnnxLocation = self.pathToModel +'.pnnx.onnx'
        ncnnPythonLocation = self.pathToModel +'_ncnn.py'
        ncnnParamLocation = self.pathToModel + '.ncnn.param'

        model = pnnx.convert(
            ptpath=jitTracedModelLocation,
            inputs=input,
            device=self.device,
            optlevel=2,
            fp16=True,
            pnnxbin=pnnxBinLocation,
            pnnxparam=pnnxParamLocation,
            pnnxpy=pnnxPythonLocation,
            pnnxonnx=pnnxOnnxLocation,
            ncnnpy=ncnnPythonLocation,
        )

        #remove stuff that we dont need
        try:
            os.remove(jitTracedModelLocation)
            os.remove(pnnxBinLocation)
            os.remove(pnnxParamLocation)
            os.remove(pnnxPythonLocation)
            os.remove(pnnxOnnxLocation)
            os.remove(ncnnPythonLocation)
        except:
            print("Could not remove unnecessary files.")
        try:
            os.remove(os.path.join(cwd, 'debug.bin'))
            os.remove(os.path.join(cwd, 'debug.param'))
            os.remove(os.path.join(cwd, 'debug2.bin'))
            os.remove(os.path.join(cwd, 'debug2.param'))
        except:
            print("Failed to remove debug pnnx files.")
        
        self.fixNCNNParamInput(ncnnParamLocation)

class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        self.checkArguments()
        if self.args.export == None:
            self.RenderSingleImage(self.args.input)
        elif self.args.export.lower().strip() == "onnx":
            self.exportModelAsONNX()
        elif self.args.export.lower().strip() == "ncnn":
            self.exportModelAsNCNN()
    def returnDevice(self):
        if not self.args.cpu:
            return "cuda" if torch.cuda.is_available() else "cpu"

    def saveImage(self, image: Image):
        image.save(self.args.output)

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

    def RenderSingleImage(self, image: Image) -> Image:
        image = self.loadImage(image)
        upscale = UpscaleImage(
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
            else upscale.renderTiledImage(imageTensor, self.tileSize)
        )
        upscaledImage = upscale.tensorToImage(upscaledTensor)
        self.saveImage(upscaledImage)

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

        if self.args.export == None:
            if self.args.input == None:
                raise os.error("Tried to upscale without Input Image!")
            if self.args.output == None:
                raise os.error("Tried to upscale without Output Image!")

            self.isDir = os.path.isdir(self.args.input)

            if self.args.input == self.args.output:
                raise os.error("Input and output cannot be the same image.")

            if not os.path.exists(self.fullModelPathandName()):
                error = f"Model {self.fullModelPathandName()} does not exist."
                raise os.error(error)

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
