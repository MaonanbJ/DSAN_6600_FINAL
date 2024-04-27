# -*- coding: utf-8 -*- 
# @Time : 2024/4/18 16:32
# @Author : Jerry Hao
# @File : VGG16.py
# @Desc :

import keras_tuner as kt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Vgg(hp):
    def build_vgg_model(hp):
        base_model = VGG16(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
        base_model.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(units=hp.Int('units', min_value=256, max_value=1024, step=256), activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(4, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model

    def tune(build_vgg_model):
        tuner = kt.RandomSearch(
            build_vgg_model,
            objective='val_accuracy',
            max_trials=5,
            executions_per_trial=1,
            directory='my_dir',
            project_name='vgg_tuning'
        )

        return tuner
