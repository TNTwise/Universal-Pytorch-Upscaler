import os
import torch
import pnnx

from .Util import loadModelWithScale, log
cwd = os.getcwd()

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
        Takes in a pytorch model, and uses JIT tracing with PNNX to convert it to ncnn.
        This method removed unnecessary files, and fixes the param file to be compadible with most NCNN appliacitons.
        """
        model = self.model.model
        model.eval()
        input = torch.rand(1,3,256,256)
        jitTracedModelLocation = self.pathToModel +'.pt'
        jitTracedModel = torch.jit.trace(model,input)
        jitTracedModel.save(jitTracedModelLocation)

        pnnxBinLocation = self.pathToModel +'.pnnx.bin'
        pnnxParamLocation = self.pathToModel +'.pnnx.param'
        pnnxPythonLocation = self.pathToModel +'_pnnx.py'
        pnnxOnnxLocation = self.pathToModel +'.pnnx.onnx'
        ncnnPythonLocation = self.pathToModel +'_ncnn.py'
        ncnnParamLocation = self.pathToModel + '.ncnn.param'

        # pnnx gives out a lot of weird errors, so i will be try/excepting this.
        # usually nothing goes wrong, but it cant take in the pnnxbin/pnnxparam location on windows.

        try:
            model = pnnx.convert(
                ptpath=jitTracedModelLocation,
                inputs=input,
                device=self.device,
                optlevel=2,
                fp16=self.half, 
                pnnxbin=pnnxBinLocation,
                pnnxparam=pnnxParamLocation, 
                pnnxpy=pnnxPythonLocation,
                pnnxonnx=pnnxOnnxLocation,
                ncnnpy=ncnnPythonLocation,
            )
        except Exception as e:
            print("WARN: Something may have gone wrong with conversion!")
            log(f"WARN: Something may have gone wrong with conversion: {e}")

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