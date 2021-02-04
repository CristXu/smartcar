import keras
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Activation, Flatten, AveragePooling2D, Deconvolution2D, Permute
from keras.models import Model, Sequential 
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 

import os 
import numpy as np 

x = np.load('./x.npy')
y = np.load('./y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40)

np.save("test_x", x_test)
np.save("test_y", y_test)

x_train = x_train / 128.0 - 1
y_train = to_categorical(y_train)

x_test = x_test / 128.0 - 1
y_test = to_categorical(y_test)

pooling = MaxPool2D
def model():
    _in = Input(shape=(32,32,3))
    x = Conv2D(32, (3,3), padding='same')(_in)
    x = pooling((2,2))(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = pooling((2,2))(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = pooling((2,2))(x)
    x = Activation("relu")(x)

    # x = Permute((3,2,1))(x)
    # x = Permute((3,2,1))(x)

    x = Flatten()(x)
    x = Dense(10)(x)
    x = Activation("softmax")(x)

    return Model(_in, x)

def model_sequential():
    mopdel = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
    model.add(pooling((2,2)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(pooling((2,2)))
    model.add(Activation("relu"))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(pooling((2,2)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation("softmax"))

    return model


if __name__ == "__main__":
    if not (os.path.exists('./models')):
        os.mkdir("./models")
    model = model()
    model.summary()

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
    early_stop = EarlyStopping(patience=20)
    reduce_lr = ReduceLROnPlateau(patience=15)
    save_weights = ModelCheckpoint("./models/model_{epoch:02d}_{val_acc:.4f}.h5", 
                                   save_best_only=True, monitor='val_acc')
    callbacks = [save_weights, reduce_lr]
    model.fit(x_train, y_train, epochs = 100, batch_size=32, 
              validation_data = (x_test, y_test), callbacks=callbacks)

    model.fit_generator(data, epochs=5, steps_per_epoch=(len(train_x)//bSize+1), 
              validation_data=[test_x, test_y], shuffle=True)


