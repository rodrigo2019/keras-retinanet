# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:01:17 2018

@author: Rodrigo
"""

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.models import Model
import keras

from ..models import retinanet


class Tiny_darknet:

    def build(input_shape=None, classes=1000, weights_path=None, input_tensor=None):
        
            
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        
                    
        x = Conv2D(16, (3, 3), padding="same", name="block1_conv1")(img_input)
        x = BatchNormalization(name="block1_batchNorm1")(x)
        x = LeakyReLU(alpha=0.1, name="block1_lkyRelu1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)
        
        x = Conv2D(32, (3, 3), padding="same", name="block2_conv1")(x)
        x = BatchNormalization(name="block2_batchNorm1")(x)
        x = LeakyReLU(alpha=0.1, name="block2_lkyRelu1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name="block2_pool")(x)
        
        x = Conv2D(16, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(16, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(x)
        
        x = Conv2D(32, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(32, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(x)

        x = Conv2D(64, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(64, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(128, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1, name="block5_lkyRelu5")(x)
        
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='tinyDarkNet')

        if weights_path is not None:
            model.load_weights(weights_path)
        return model
    

custom_objects = retinanet.custom_objects


def download_imagenet(backbone):
    pass


def validate_backbone(backbone):
    allowed_backbones = ['tiny_darknet']

    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))
        
        
def darknet_retinanet(num_classes, backbone='tinyDarkNet', inputs=None, modifier=None, **kwargs):
    
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))
     
    if backbone == 'tiny_darknet':
        darknet = Tiny_darknet.build(classes=num_classes,input_tensor=inputs)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))
    
    # create the full model
    layer_names =  ["block3_pool", "block4_pool", "block5_lkyRelu5"]
    layer_outputs = [darknet.get_layer(name).output for name in layer_names]
    
    return retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
