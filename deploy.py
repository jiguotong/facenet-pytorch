import torch
import os
import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet as facenet
from utils.utils import show_config

class ConvertModelType(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        #--------------------------------------------------------------------------#
        "model_path"    : "model_data/facenet_mobilenet.pth",
        #--------------------------------------------------------------------------#
        #   输入图片的大小。
        #--------------------------------------------------------------------------#
        "input_shape"   : [160, 160, 3],
        #--------------------------------------------------------------------------#
        #   所使用到的主干特征提取网络
        #--------------------------------------------------------------------------#
        "backbone"      : "mobilenet",
        #-------------------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------------------#
        "cuda"              : True,
    }    
    #---------------------------------------------------#
    #   初始化Facenet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self._defaults.update(kwargs)               ## 更新传进来的参数到_defaults
        self.__dict__.update(self._defaults)        ## 更新_defaults到self属性
        self.generate()
        
        show_config(**self._defaults)
        
    def generate(self):
        #---------------------------------------------------#
        #   载入模型与权值
        #---------------------------------------------------#
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net    = facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def pth_to_onnx(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        # prepare input data
        width, height, channel = self.input_shape
        img = torch.randn(1, channel, height, width)       
        if self.cuda:
            img = img.cuda()

        torch.onnx.export(self.net,
                        img,
                        output_path,
                        verbose=False,
                        opset_version=11,
                        input_names=['input'],  # the model's input names
                        output_names=['output'])   
        print("\033[32m[Info]:.pth->.onnx finished\033[0m")

if __name__ == '__main__':
    ## 构建转换示例
    ## Notice:通过下一行的形参来控制参数，其他地方不要修改
    convert = ConvertModelType(cuda=False)
    onnx_path = 'model_data/onnx/model.onnx'
    convert.pth_to_onnx(onnx_path)
    pass


# print("\033[31m这是红色字体\033[0m")
# print("\033[32m这是绿色字体\033[0m")
# print("\033[33m这是黄色字体\033[0m")
# print("\033[34m这是蓝色字体\033[0m")
# print("\033[38m这是默认字体\033[0m")  # 大于37将显示默认字体