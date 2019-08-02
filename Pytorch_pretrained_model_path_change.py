
'''
The default value of model_dir is $TORCH_HOME/models where  $TORCH_HOME defaults to ~/.torch.

The default directory can be overridden with the $TORCH_HOME environment variable.

https://stackoverflow.com/questions/52628270/is-there-any-way-i-can-download-the-pre-trained-models-available-in-pytorch-to-a

'''
import os

# Suppose you are trying to load pre-trained resnet model in directory- ..models\resnet

os.environ['TORCH_HOME'] = 'D:\dev\Pytorch_Models\models\\resnet' #setting the environment variable
resnet = torchvision.models.resnet18(pretrained=True)