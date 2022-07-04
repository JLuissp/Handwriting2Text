import tensorflow as tf
import tensorflow.keras as K
from functools import partial
import warnings
warnings.filterwarnings("ignore")
import pandas as pd



DefaultConv2D = partial(K.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(K.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = K.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            K.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            K.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                K.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'filters': filters,
            'strides': strides,
            'activation': self.activation})
        return config

class ResNet34():
    def __init__(self, input_shape):
        self.DataAugmentation = K.Sequential([K.layers.RandomRotation(1/16, fill_mode='constant'),
                                 K.layers.Rescaling(1./255)])
        self.RLR = K.callbacks.ReduceLROnPlateau(monitor= 'val_accuracy',
                                   factor = 0.01,
                                   patience=3,
                                   mode='max',
                                   min_delta=1e-4)
        self.EarlyS  = K.callbacks.EarlyStopping(monitor = 'val_loss',
                                   min_delta = 1e-9,
                                   patience = 3,
                                   mode = 'min')
        self.model = K.models.Sequential()
        
        self.input_shape = input_shape

    def model_blueprint(self):
        tf.keras.backend.clear_session()

        self.model.add(self.DataAugmentation)
        self.model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                                input_shape=self.input_shape))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Activation("softmax"))
        self.model.add(K.layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"))
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.model.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        self.model.add(K.layers.GlobalAvgPool2D())
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(26, activation="softmax"))

        optimizer = K.optimizers.Adam(learning_rate=0.001345456,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-07,
                                    amsgrad=False,
                                    name="Adam")

        self.model.compile(loss ='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

        w_DIR='G:\projects\weights\HandwrittenCharacterClassifierWeights_003.h5'
        train_dataset, val_dataset = self.load_datasets()
        self.model.fit(train_dataset,validation_data=val_dataset,epochs=2,callbacks=[self.EarlyS, self.RLR])
        self.model.load_weights(w_DIR)
        return self.model


    def load_datasets(self):
        DIR = 'G:\projects\datasets\handw_dataset.csv'
        BATCH_SIZE = 64
        def buildImageDS(DIR, frac=None):
            if frac is None: frac=1.0
            
            print('fraction: ', frac)
            dataset = pd.read_csv(DIR).sample(frac=frac)
            shape = (dataset.shape[0],) + self.input_shape
            print('shape: ', shape)
            
            image_dataset = tf.data.Dataset.from_tensor_slices((dataset.values[:,1:].reshape(shape), dataset.values[:,0]))
            
            return image_dataset

        def train_valid_split(dataset, test_size):
            cardinality = dataset.cardinality().numpy()
            n_samples = round(cardinality*test_size)
            
            val_dataset = dataset.take(n_samples).batch(BATCH_SIZE)
            train_dataset = dataset.skip(n_samples).batch(BATCH_SIZE)
            
            print('train_steps: ', train_dataset.cardinality().numpy())
            print('validation_steps: ', val_dataset.cardinality().numpy())
            return train_dataset, val_dataset

        train_dataset, val_dataset = train_valid_split(buildImageDS(DIR, frac=0.002), 0.2)

    
        return train_dataset, val_dataset