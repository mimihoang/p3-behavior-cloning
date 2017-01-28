import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import cv2
from keras.optimizers import Nadam
import math

size_row, size_col, channel = 64, 64, 3

def augment_brightness_camera_images(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image


def normalize_image(image):
    image = image.astype(np.float32)
    image = image/255.0 - 0.5
    return image


def preprocess_image(image):
    #image = image[55:135, :, :]
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (size_row, size_col))
    image = normalize_image(image)
    return image


def preprocess_image_data(row_data):

    y_steer = row_data['steering']

    cam_loc = np.random.choice(['center', 'left', 'right'])


    # Adjust the steering angle from the left or right camera to simulate recovery
    if cam_loc == 'left':
        y_steer += 0.25
    elif cam_loc == 'right':
        y_steer -= 0.25

    image = cv2.imread("data/" + row_data[cam_loc].strip())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    # Flip image horizontally and change steering angle accordingly to generate more data
    flip_prob = np.random.random()
 
    if flip_prob > 0.5:
        y_steer = -1*y_steer
        image = cv2.flip(image, 1)
    
    
    image = augment_brightness_camera_images(image)
    image = preprocess_image(image)
    
    return image, y_steer


def generate_data(data, batch_size):
    
    df_len = data.shape[0]
    batches = df_len // batch_size
    
    i = 0
    while 1:
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, size_row, size_col, channel), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0
        for index, row in data.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = preprocess_image_data(row)
            j += 1

        i += 1
        if i == batches - 1:
            i = 0
            
        yield X_batch, y_batch


def get_model():
    model = Sequential()
    
    model.add(Convolution2D(16, 8, 8, input_shape=(size_row, size_col, channel), subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    optimizer = Nadam()

    model.compile(optimizer=optimizer, loss="mse")

    return model


def save_model(model):
    model.save_weights('model.h5') 
    with open('model.json', 'w') as f:
        f.write(model.to_json())


if __name__ == "__main__":

    batch_size = 32

    ### Loading driving log data
    driving_log = pd.read_csv('data/driving_log.csv')

    ### Remove data with throttle below .25
    index = driving_log['throttle'] > .25
    df = driving_log[index]
    total_row = len(df)-1

    df = df.sample(frac=1).reset_index(drop=True)

    train_samples = int(.8 * total_row)

    training_data = df.loc[:train_samples]
    validation_data = df.loc[train_samples:]

    training_generator = generate_data(training_data, batch_size=batch_size)
    validation_generator = generate_data(validation_data, batch_size=batch_size)


    model = get_model()

    model.fit_generator(training_generator, validation_data=validation_generator,
                        samples_per_epoch=20000, nb_epoch=8, nb_val_samples=2000)

    save_model(model)
