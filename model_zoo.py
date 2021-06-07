"""
@author: Marco Penso
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import ResNet50V2

def VGG16model(images):
  
  return VGG16(weights='imagenet', include_top=False)
  
  
def ResNet50V2model(images):
  
  return ResNet50V2(weights='imagenet', include_top=False)
  




def get_model(images, config):
  
  return config.model_handle(images, nlabels=config.nlabels)
