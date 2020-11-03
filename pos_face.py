import keras
from PIL import Image
from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    # shear_range=0.2,
    # zoom_range=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    r'F:\only_faces\dataset\train_dir',
    target_size=(96, 96),
    color_mode='grayscale',
    class_mode='binary'
)
test_set = test_datagen.flow_from_directory(
    r'F:\only_faces\dataset\validation_dir',
    target_size=(96, 96),
    color_mode='grayscale',
    class_mode='binary'
)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = (96,96,1)))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = RMSprop(), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    return model


model = build_model()
history = model.fit(
    train_set,
    epochs=10,
    # steps_per_epoch=30,
    shuffle=True,
    validation_data=test_set
)
model.save("models/pos_face.h5")

pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='val_loss')
pyplot.plot(history.history['accuracy'], label='train_acc')
pyplot.plot(history.history['val_accuracy'], label='val_acc')
pyplot.legend()
pyplot.show()

