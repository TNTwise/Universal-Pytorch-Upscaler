from spandrel import ImageModelDescriptor, MAIN_REGISTRY, ModelLoader
import sys
import torch
import argparse
import os
from torchvision import transforms
from PIL import Image

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
                 device="cuda"):
        
        self.loadModel(os.path.join(modelPath,modelName))
        self.setDevice(device)
    
    def loadModel(self, modelPath):
        self.model = ModelLoader().load_from_file(modelPath)
        assert isinstance(self.model, ImageModelDescriptor)
    
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
        upscaledTensor = upscale.renderImage(imageTensor)
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
        
        if not isDir: # Executed if it is not rendering an image directory
            
            
            
            if not os.path.isfile(self.inputImage):
                raise os.error('Input File/Directory does not exist!')
            
            if not is_image(self.inputImage):
                raise os.error('Not a valid image File/Directory!')
            
            
        else:
            
            
            
            if len(os.listdir(self.inputDir)) == 0 :
                raise os.error('Empty Input Directory!')
            
            if not os.path.isdir(self.outputDir):
                raise os.error('Output must be a directory if input is a directory!')
            
            
        
        if not os.path.exists(os.path.join(self.modelPath,self.modelName)):
            error = f'Model {os.path.join(self.modelPath,self.modelName)} does not exist!'
            raise os.error(error)
        
        
    def handleArguments(self, args: list) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_
            
        """
        parser = argparse.ArgumentParser(description="Upscale any image, with most torch models, using spandrel.")
        
        parser.add_argument('-i', required=True, help='input image path (jpg/png/webp) or directory')
        parser.add_argument('-o', required=True, help='output image path (jpg/png/webp) or directory')
        parser.add_argument('-t', help='tile size (>=32/0=auto, default=0)')
        parser.add_argument('-m', help='folder path to the pre-trained models. default=models',default='models')
        parser.add_argument('-n', required=True, help='model name (include extension)')
        parser.add_argument('-c', help='use only CPU for upscaling, instead of cuda. default=auto',action='store_true')
        parser.add_argument('-f', help='output image format (jpg/png/webp, default=ext/png)')
        args = parser.parse_args()
        
        return args



if __name__ == '__main__':
    handleApplication()