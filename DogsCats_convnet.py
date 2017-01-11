from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

img_width, img_height = 64, 64  # set image dimensions to 320x180 pixels

training_data_dir = 'input/trainSmall'  # directory for training data
test_data_dir = 'input/validation'  # directory for test data

nb_train_samples = 2002
nb_val_samples = 1001
nb_epoch = 1

# # set up checkpoints for weights
# filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath,
#                              monitor='accuracy',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='max')
# callbacks_list = [checkpoint]

# model creation: Three convolutional layers
model = Sequential()

# apply a 5x5 convolution with 32 output filters on a 180x320 image:
model.add(Convolution2D(32, 5, 5, init='normal', border_mode='same',
                        input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # converts 3D feature mapes to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))  # reset half of the weights to zero

# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# set of augments that will be applied to the training data
train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

# only color rescale for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# this generator will read pictures found in a sub folder
# it will indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')  # need categorical labels

# validation_generator = test_datagen.flow_from_directory(
#         val_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=32,
#         class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        verbose=1
        )

# model.save_weights("model_trainingWeights_final.h5")
# print("Saved model weights to disk")
#
# model.predict_generator(
#         test_generator,
#         val_samples=nb_test_samples)
