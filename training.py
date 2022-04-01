from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import adam_v2
import cv2
import collections
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


def get_minibatch(experience, m, num_actions, gamma, size_of_minibatch):
    batch_indices = np.random.randint(low=0, high=len(experience), size=size_of_minibatch)
    batch = [experience[i] for i in batch_indices]
    X = np.zeros((size_of_minibatch, 80, 80, 4))
    Y = np.zeros((size_of_minibatch, num_actions))
    for i in range(len(batch)):
        state, action, reward, state_2, konec_igry = batch[i]
        X[i] = state
        Y[i] = m(state)[0]
        Q_sa = np.max(m(state_2)[0])
        if konec_igry:
            Y[i, action] = reward
        else:
            Y[i, action] = reward + gamma * Q_sa
    return X, Y


def create_model():
    m = Sequential()
    m.add(Conv2D(32, kernel_size=8, strides=4, kernel_initializer="normal", padding="same", input_shape=(80, 80, 4)))
    m.add(Activation("relu"))
    m.add(Conv2D(64, kernel_size=4, strides=2,
                 kernel_initializer="normal", padding="same"))
    m.add(Activation("relu"))
    m.add(Conv2D(64, kernel_size=3, strides=1,
                 kernel_initializer="normal", padding="same"))
    m.add(Activation("relu"))
    m.add(Flatten())
    m.add(Dense(512, kernel_initializer="normal"))
    m.add(Activation("relu"))

    m.add(Dense(3, kernel_initializer="normal"))
    return m


def train_model(m, initial_exp_rate, final_exp_rate, gamma,
                size_of_minibatch, observe_epochs, train_epochs):
    experience = collections.deque(maxlen=50000)
    igra = Igra()
    kolvo_pobed = 0
    epochs = observe_epochs + train_epochs
    exp_rate = initial_exp_rate
    optimizer = adam_v2.Adam()
    for e in range(epochs):
        loss = 0.0
        igra.restart()
        action_0 = 1
        image, reward_0, konec_igry = igra.update(action_0)
        state = get_state(image)
        while not konec_igry:
            state_1 = state
            m.compile(optimizer=optimizer, loss='mse')
            if e <= observe_epochs:
                action = np.random.randint(low=0, high=igra.num_actions, size=1)[0]
            else:
                if np.random.rand() <= exp_rate:
                    action = np.random.randint(low=0, high=igra.num_actions, size=1)[0]
                else:
                    q = m(state)[0]
                    action = np.argmax(q)
            image, reward, konec_igry = igra.update(action)
            state = get_state(image)

            if reward == 1:
                kolvo_pobed += 1
            experience.append((state_1, action, reward, state, konec_igry))

            if e > observe_epochs:
                X, Y = get_minibatch(experience, m, igra.num_actions, gamma, size_of_minibatch)
                loss += m.train_on_batch(X, Y)
        if exp_rate > final_exp_rate:
            exp_rate -= (initial_exp_rate - final_exp_rate) / epochs
        print('Эпоха {:04d}/{:d} | loss {:.5f} | Число побед: {:d}'.format(e + 1, epochs, loss, kolvo_pobed))
        if e % 100 == 0:
            m.save('squirrel_model.h5', overwrite=True)
    m.save('squirrel_model.h5', overwrite=True)
train_model(create_model(), 0.01, 0.02, 1, 200, 100, 200)