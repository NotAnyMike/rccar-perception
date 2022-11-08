from pl_bolts.models.vision import unet
from torchvision.models import resnet50
from torch import nn
from torch.nn import functional as F
import torch


def get_unet(input_shape, num_outputs):
    """
    """
    model = unet.UNet(input_shape=input_shape, num_classes=num_outputs)
    return model


def get_resnet50(input_shape, num_outputs):
    """
    """
    model = resnet50(pretrained=True)
    model.fc = nn.sigmoid(nn.Linear(model.fc.in_features, num_outputs))
    return model


def build_3d_cnn_pytorch():
    model = nn.Sequential(
        conv3d_relu_maxpool(1, 16, 3, 1, 0),
        conv3d_relu_maxpool(16, 32, 3, 1, 0),
        conv3d_relu_maxpool(32, 64, 3, 1, 0),
        conv3d_relu_maxpool(64, 128, 3, 1, 0),
        conv3d_relu_maxpool(128, 256, 3, 1, 0),
        nn.Flatten(),
        dense_layer(256 * 4 * 4 * 4, 1024),
        dense_layer(1024, 512),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model


def conv3d_relu_maxpool(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),
    )


def dense_layer(in_features, out_features):
    """A single layer of a dense network with batch norm and dropout.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
    )


def build_3d_cnn_keras(input_shape, s, num_outputs):
    """
    Credit: https://github.com/jessecha/DNRacing/blob/master/3D_CNN_Model/model.py

    :param input_shape:     image input shape
    :param s:               sequence length
    :param num_outputs:     output dimension
    :return:                keras model
    """
    """
    drop = 0.5
    input_shape = (s, ) + input_shape
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    # Second layer
    x = Conv3D(
            filters=16, kernel_size=(3, 3, 3), strides=(1, 3, 3),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Third layer
    x = Conv3D(
            filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
        data_format=None)(x)
    # Fourth layer
    x = Conv3D(
            filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Fifth layer
    x = Conv3D(
            filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Fully connected layer
    x = Flatten()(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    out = Dense(num_outputs, name='outputs')(x)
    model = Model(inputs=[img_in], outputs=out, name='3dcnn')
    return model
    """
    raise NotImplementedError


if __name__ == '__main__':
    import pdb; pdb.set_trace()

    model = get_resnet50((1, 128, 128), 2)
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    x = torch.randn(1, 3, 640, 640)
    output = model(x)
    print("H")
