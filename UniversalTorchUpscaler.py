from spandrel import ImageModelDescriptor, MAIN_REGISTRY, ModelLoader
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
    def __init__(self,
                 modelPath:str='models',
                 modelName:str='',
                 device="cuda",
                 tile_pad = 10):
        path = os.path.join(modelPath,modelName)
        self.loadModel(path)
        self.setDevice(device)

        self.tile_pad = tile_pad

    def loadModel(self, modelPath):
        self.model = ModelLoader().load_from_file(modelPath)
        assert isinstance(self.model, ImageModelDescriptor)
        #get model attributes
        self.scale = self.model.scale
        

    def setDevice(self,device):
        if device == "cpu":
            self.model.eval().cpu()
        if device == "cuda":
            self.model.eval().cuda()
        self.device = device
    
    

    def imageToTensor(self, image: Image) -> torch.Tensor:
        
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])

        return transform(image).unsqueeze(0).to(self.device)
    
    def tensorToImage(self, image: torch.Tensor) -> Image:
        transform = transforms.ToPILImage()
        return transform(image.squeeze(0))

    @torch.inference_mode()
    def renderImage(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(image)
    
    @torch.inference_mode()
    def renderImagesInDirectory(self,dir):
        pass
    
    def getScale(modelPath):
        return ModelLoader().load_from_file(modelPath).scale


    def renderTiledImage(self, image: torch.Tensor, tile_size: int = 32) -> torch.Tensor:
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
                input_tile = image[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                with torch.no_grad():
                    output_tile = self.renderImage(input_tile)
                
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

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
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output


class handleApplication:
    def __init__(self):
        self.args = self.handleArguments(sys.argv)
        self.checkArguments()
        self.Upscale()
    
    def returnDevice(self):
        if not self.isCPU:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    def saveImage(self,image: Image):
        image.save(self.outputImage)
    
    def loadImage(self,image:str) -> Image:
        return Image.open(self.inputImage)
    
    def Upscale(self):
        upscale = UpscaleImage(modelPath=self.modelPath,
                                modelName=self.modelName,
                                device=self.returnDevice())
        image = self.loadImage(self.inputImage)
        imageTensor = upscale.imageToTensor(image)
        if self.tileSize == 0:
            upscaledTensor = upscale.renderImage(imageTensor)
        else:
            upscaledTensor = upscale.renderTiledImage(imageTensor,self.tileSize)
        upscaledImage = upscale.tensorToImage(upscaledTensor)
        self.saveImage(upscaledImage)
        
    def checkArguments(self):
        
        isDir = os.path.isdir(self.args.i)
        
        self.modelPath = self.args.m
        
        self.modelName = self.args.n
        self.inputImage = self.args.i
        self.inputDir = self.args.i
        self.outputImage = self.args.o
        self.outputDir = self.args.o
        self.isCPU = self.args.c
        self.tileSize = int(self.args.t)

        if self.inputImage == self.outputImage:
            raise os.error("Input and output cannot be the same image.")
        if not isDir: # Executed if it is not rendering an image directory
            
            
            
            if not os.path.isfile(self.inputImage):
                raise os.error('Input File/Directory does not exist.')
            
            if not is_image(self.inputImage):
                raise os.error('Not a valid image File/Directory.')
            
            
        else:
            
            if len(os.listdir(self.inputDir)) == 0 :
                raise os.error('Empty Input Directory!')
            
            if not os.path.isdir(self.outputDir):
                raise os.error('Output must be a directory if input is a directory.')
            
            
        
        if not os.path.exists(os.path.join(self.modelPath,self.modelName)):
            error = f'Model {os.path.join(self.modelPath,self.modelName)} does not exist.'
            raise os.error(error)
            
        if type(self.tileSize) == int:
            if self.tileSize < 32 and self.tileSize != 0:
                raise os.error("Tile size must be greater than or equal to 32.")
        else:
            raise os.error("Tile size must be integer.")

    def handleArguments(self, args: list) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_
            
        """
        parser = argparse.ArgumentParser(description="Upscale any image, with most torch models, using spandrel.")
        
        parser.add_argument('-i', required=True, help='input image path (jpg/png/webp) or directory')
        parser.add_argument('-o', required=True, help='output image path (jpg/png/webp) or directory')
        parser.add_argument('-t', help='tile size (>=32/0=auto, default=0)',default=0,type=int)
        parser.add_argument('-m', help='folder path to the pre-trained models. default=models',default='models')
        parser.add_argument('-n', required=True, help='model name (include extension)')
        parser.add_argument('-c', help='use only CPU for upscaling, instead of cuda. default=auto',action='store_true')
        parser.add_argument('-f', help='output image format (jpg/png/webp, default=ext/png)')
        args = parser.parse_args()
        
        return args



if __name__ == '__main__':
    handleApplication()
