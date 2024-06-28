
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
import os
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

def log(message:str):
    with open(
        os.path.join(cwd, 'log.txt'), 
        'a'
    ) as f:
        
        f.write(message + '\n')