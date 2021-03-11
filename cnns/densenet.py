#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Layer, Input, MaxPooling2D, ReLU
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Concatenate, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

tf.config.list_physical_devices('GPU')


# In[ ]:


class DenseBlock(Layer):
    def __init__(self, block_size, filters, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.filters = filters
        self.layers = []

        for _ in range(block_size):
            self.layers.append(BottleneckLayer(filters))

    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
            
        return x
    
    def get_config(self):
        config = super().get_config()

        config.update({
            'block_size': self.block_size,
            'filters': self.filters,
            'layers': self.layers
        })
        
        return config


# In[ ]:


class BottleneckLayer(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.conv1 = Conv2D(filters, 1, padding='same')
        self.conv2 = Conv2D(filters, 3, padding='same')
        self.relu = ReLU()
        self.concat = Concatenate()
        
    def call(self, x, training=False):
        previous_x = x
        x = self.batch_norm1(x, training=training)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.batch_norm2(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.concat([previous_x, x])
        
        return x
    
    def get_config(self):
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'batch_norm1': self.batch_norm1,
            'batch_norm2': self.batch_norm2,
            'conv1': self.conv1,
            'conv2': self.conv2,
            'relu': self.relu,
            'concat': self.concat
        })
        
        return config


# In[ ]:


class TransitionBlock(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        self.conv = Conv2D(filters, 1, padding='same')
        self.avg_pool = AveragePooling2D(strides=2, padding='same')
        
    def call(self, x, training=False):
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'batch_norm': self.batch_norm,
            'relu': self.relu,
            'conv': self.conv,
            'avg_pool': self.avg_pool
        })
        
        return config


# In[ ]:


class InputBlock(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
        self.conv = Conv2D(filters, 7, padding='same', strides=2)
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        self.max_pool = MaxPooling2D(strides=2)
        
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.batch_norm(x, training=training)
        x = self.relu(x)
        x = self.max_pool(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'conv': self.conv,
            'batch_norm': self.batch_norm,
            'relu': self.relu,
            'max_pool': self.max_pool
        })
        
        return config


# In[ ]:


class DenseNet(Sequential):
    def __init__(self, input_shape, num_classes, block_sizes=[6,12,24,16], init_filters=24, growth_rate=12, compression=0.5, low_res=False, **kwargs):
        super().__init__(**kwargs)
        config = locals()
        config.pop('self')
        config.pop('__class__')
        self.config = config
        
        self.add(Input(input_shape))

        if low_res:
            self.add(Conv2D(init_filters, 3, padding='same'))
        else:
            self.add(InputBlock(filters=init_filters))
        
        for block_size in block_sizes:
            self.add(DenseBlock(block_size, filters=growth_rate))
            init_filters = (init_filters + (block_size * growth_rate)) * compression
            self.add(TransitionBlock(filters=init_filters))

        self.add(BatchNormalization())
        self.add(ReLU())
        self.add(GlobalAveragePooling2D())
        self.add(Dense(num_classes, activation='softmax'))
        
    def get_config(self):
        config = super().get_config()
        
        config.update(**self.config)

        return config


# In[ ]:


def load_ds():
    (X_train, y_train), (X_val, y_val) = cifar100.load_data()

    X_train = X_train / 255.0
    X_val = X_val / 255.0

    train_ds = Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.batch(64)

    val_ds = Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.shuffle(10000)
    val_ds = val_ds.batch(64)
    
    return train_ds, val_ds


# In[ ]:


dense_net = DenseNet(input_shape=(32, 32, 3), num_classes=100, low_res=True)

dense_net.compile(Adam(lr=1e-02), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_ds, val_ds = load_ds()

callbacks = [
    TensorBoard('logs/densenet/test/tensorboard'),
    ModelCheckpoint('logs/densenet/test/models/epoch_{epoch}')
]

dense_net.fit(train_ds, epochs=200, validation_data=val_ds, callbacks=callbacks)

