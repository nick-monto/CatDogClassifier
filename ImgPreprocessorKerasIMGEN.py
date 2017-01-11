from keras.preprocessing.image import ImageDataGenerator

trainDIR = 'input/trainSmall'  # insert training directory
saveTrainDIR = 'input/previewTrain'

testDIR = 'input/validation'  # insert test directory
saveTestDIR = 'input/previewTest'

img_width = 100
img_height = 100

train_datagen = ImageDataGenerator(
            rotation_range=20,  # random rotation in degrees
            rescale=1./255,  # rescaling of the pixel information to 0-1 format
            width_shift_range=0.1,  # percent of random shift, up to 10%
            height_shift_range=0.1,
            zoom_range=0.1,  # random percent zoom up to 10%
            horizontal_flip=True,  # random flipping on the horizontal axis
            fill_mode='nearest')  # filling of empty pixels

# only color rescale for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# this generator will read pictures found in a sub folder
# it will indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        trainDIR,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        save_to_dir=saveTrainDIR,
        save_format='jpeg')

i = 0
for batch in train_generator:
    i += 1
    if i > 20:  # generate twenty images to see what you're creating
        break

validation_generator = test_datagen.flow_from_directory(
        testDIR,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        save_to_dir=saveTestDIR,
        save_format='jpeg')

i = 0
for batch in validation_generator:
    i += 1
    if i > 20:
        break
