from keras.applications import VGG16
from keras import models, layers

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

from keras.callbacks import ModelCheckpoint

import pickle

DATA_DIRECTORY = "data/img"
FILE_PATH = "model-{epoch:02d}-{loss:.4f}.h5"
TEST_DIR = "data/img/test"

def create_model():
    conv_base = VGG16(weights="imagenet",
                      include_top=False,
                      input_shape=(150, 150, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten()) # flatten - matrix resized to vector
    model.add(layers.Dense(256, activation="relu")) # fully connected
    model.add(layers.Dense(1, activation="sigmoid"))
    conv_base.trainable = False

    return model


def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = "data/img/train"
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary")

    validation_dir = "data/img/validate"
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary")

    return train_generator, validation_generator


if __name__ == "__main__":
    model = create_model()
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=["acc"]
                  )

    train_generator, validation_generator = get_generators()

    checkpoint = ModelCheckpoint(
        FILE_PATH, monitor="loss", verbose=1, save_best_only=True, mode="min")

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[checkpoint])

    model.save("deer_1.h5")

    with open("history.pickle", "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    test_dir = TEST_DIR
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary")

    test_loss, test_acc = model.evaluate(x=test_generator, steps=50, verbose=1)
    print("test acc:", test_acc)
