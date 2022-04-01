from keras.models import load_model
from keras.optimizers import adam_v2
import cv2
import numpy as np
import os
from game import Igra


def get_state(images):
    if images.shape[0] < 4:
        image = images[0]
        image = cv2.resize(image, (80, 80))
        image = image.astype("float")
        image /= 255.0
        state = np.stack((image, image, image, image), axis=2)
    else:
        image_list = []
        for i in range(images.shape[0]):
            image = cv2.resize(images[i], (80, 80))
            image = image.astype("float")
            image /= 255.0
            image_list.append(image)
        state = np.stack((image_list[0], image_list[1], image_list[2], image_list[3]), axis=2)
    state = np.expand_dims(state, axis=0)
    return state

SIZE_OF_BATCH = 32
EPOCHS = 10
m = load_model("squirrel_model.h5")
m.compile(optimizer=adam_v2.Adam(lr=1e-6), loss="mse")
igra = Igra()
num_wins = 0
for e in range(EPOCHS):
    loss = 0.0
    igra.restart()
    action = 1
    screenshots, reward, konec_igry = igra.update(action)
    state = get_state(screenshots)
    while not konec_igry:
        q = m.predict(state)[0]
        action = np.argmax(q)
        screenshots, reward, konec_igry = igra.update(action)
        state = get_state(screenshots)
        if reward == 1:
            num_wins += 1
    print("Число партий: {:03d}, Число побед: {:03d}".format(e + 1, num_wins))
print("")
