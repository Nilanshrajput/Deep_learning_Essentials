import tensorflow as tf
from tensorflow.keras.preprocessing import image
keras = tf.keras

#Takes path to file as input
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)  
  return input_image


def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

# normalizing the images to [-1, 1]

def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

def preprocess_imgs(input_image):
    size=256
    input_image=resize(input,size,size)
    input_image = np.expand_dims(input_image, axis=0)
    
    ###
    #Not using normalize function as it is done internanlly the step below
    ###
    
    input_image = keras.applications.ResNet50.preprocess_inputpreprocess_input(x)

    return input_image
  
"""

####
Various models present in tf2.0
####

DenseNet121(...): Instantiates the DenseNet architecture.

DenseNet169(...): Instantiates the DenseNet architecture.

DenseNet201(...): Instantiates the DenseNet architecture.

InceptionResNetV2(...): Instantiates the Inception-ResNet v2 architecture.

InceptionV3(...): Instantiates the Inception v3 architecture.

MobileNet(...): Instantiates the MobileNet architecture.

NASNetLarge(...): Instantiates a NASNet model in ImageNet mode.

NASNetMobile(...): Instantiates a Mobile NASNet model in ImageNet mode.

ResNet50(...): Instantiates the ResNet50 architecture.

VGG16(...): Instantiates the VGG16 architecture.

VGG19(...): Instantiates the VGG19 architecture.

Xception(...): Instantiates the Xception architecture.

"""