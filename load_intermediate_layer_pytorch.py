# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer

#created a reference to the layer we want to extract from
layer = model._modules.get('avgpool')

# Set model to evaluation mode

#n order to ensure that any Dropout layers are not active during the forward pass.
model.eval()