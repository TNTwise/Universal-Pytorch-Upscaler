from spandrel import ImageModelDescriptor, ModelLoader
import sys
import torch
import argparse
import os
from torchvision import transforms
from PIL import Image
import math
# tiling code permidently borrowed from https://github.com/chaiNNer-org/spandrel/issues/113#issuecomment-1907209731


def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Check if it's a valid image file
        return True
    except (IOError, SyntaxError):
        return False


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
        self.loadModel(path, half, bfloat16)

    def setDevice(
        self,
        device: str = "cuda",
    ):
        self.device = device

    def loadModel(self, modelPath: str, half: bool = False, bfloat16: bool = False):
        self.model = ModelLoader().load_from_file(modelPath)
        assert isinstance(self.model, ImageModelDescriptor)
        # get model attributes
        self.scale = self.model.scale

        if self.device == "cpu":
            self.model.eval().cpu()
        if self.device == "cuda":
            self.model.eval().cuda()
            if half:
                self.model.half()
            if bfloat16:
                self.model.bfloat16()

    def imageToTensor(self, image: Image) -> torch.Tensor:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL image to tensor
            ]
        )

        imageTensor = transform(image).unsqueeze(0).to(self.device)

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

    def getScale(modelPath):
        return ModelLoader().load_from_file(modelPath).scale

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


class handleApplication:
    def __init__(self):
        self.handleArguments()
        self.checkArguments()

        image = self.loadImage(self.args.input)
        upscaledImage = self.RenderSingleImage(image=image)
        self.saveImage(upscaledImage)

    def returnDevice(self):
        if not self.args.cpu:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def saveImage(self, image: Image):
        image.save(self.args.output)

    def loadImage(self, image: str) -> Image:
        return Image.open(image)

    def RenderSingleImage(self, image: Image) -> Image:
        upscale = UpscaleImage(
            modelPath=self.args.modelPath,
            modelName=self.args.modelName,
            device=self.returnDevice(),
            half=self.args.half,
            bfloat16=self.args.bfloat16,
        )
        
        imageTensor = upscale.imageToTensor(image)
        upscaledTensor = (upscale.renderImage(imageTensor) # render image, tile if necessary
                          if self.args.tilesize == 0 
                          else upscale.renderTiledImage(imageTensor, self.tileSize)
                          )
        upscaledImage = upscale.tensorToImage(upscaledTensor)
        return upscaledImage

    def checkArguments(self):
        assert isinstance(self.args.tilesize, int)
        assert self.args.tilesize > 31 or self.args.tilesize == 0, "Tile size must be greater than 32 (inclusive), or 0."
        self.isDir = os.path.isdir(self.args.input)

        if self.args.input == self.args.output:
            raise os.error("Input and output cannot be the same image.")
        
        if not self.isDir:  # Executed if it is not rendering an image directory
            if not os.path.isfile(self.args.input):
                raise os.error("Input File/Directory does not exist.")

            if not is_image(self.args.input):
                raise os.error("Not a valid image File/Directory.")

        else:
            if len(os.listdir(self.args.input)) == 0:
                raise os.error("Empty Input Directory!")

            if not os.path.isdir(self.args.output):
                raise os.error("Output must be a directory if input is a directory.")

        if not os.path.exists(os.path.join(self.args.modelPath, self.args.modelName)):
            error = (
                f"Model {os.path.join(self.args.modelPath,self.args.modelName)} does not exist."
            )
            raise os.error(error)

        
       

    def handleArguments(self) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_

        """
        parser = argparse.ArgumentParser(
            description="Upscale any image, with most torch models, using spandrel."
        )

        parser.add_argument(
            "-i", "--input", required=True, help="input image path (jpg/png/webp) or directory"
        )
        parser.add_argument(
            "-o", "--output", required=True, help="output image path (jpg/png/webp) or directory"
        )
        parser.add_argument(
            "-t", "--tilesize", help="tile size (>=32/0=auto, default=0)", default=0, type=int
        )
        parser.add_argument(
            "-m",
            "--modelPath",
            help="folder path to the pre-trained models. default=models",
            default="models",
        )
        parser.add_argument(
                            "-n",
                            "--modelName",
                            required=True, 
                            help="model name (include extension)"
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

        self.args = parser.parse_args()


if __name__ == "__main__":
    handleApplication()
