from tensorflow.keras import layers, Model, Sequential, regularizers


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, weight_decay = 1e-4,**kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs  
        if self.downsample is not None:  
            identity = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.add([identity, x])
        x = self.relu(x)

        return x
    
    def get_config(self):
        config = super(BasicBlock, self).get_config()
        config.update({
            'out_channel': self.conv1.filters,
            'strides': self.conv1.strides,
            'downsample': self.downsample,
            'weight_decay': self.conv1.kernel_regularizer.l2
            # include other parameters that you need to save alongside the model
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def _make_layer(block, in_channel, channel, block_num, name, strides=1, weight_decay = 1e-4):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1",kernel_regularizer=regularizers.l2(weight_decay)),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")
    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))
    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True,weight_decay=1e-4):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1",weight_decay=weight_decay)(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2",weight_decay=weight_decay)(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3",weight_decay=weight_decay)(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4",weight_decay=weight_decay)(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model

def resnet34(im_width=224, im_height=224, num_classes=1000,weight_decay=1e-4):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes,weight_decay=weight_decay)
