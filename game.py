import collections
import pygame as pgm
import numpy as np
import random
import os


class Igra:
    def __init__(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pgm.init()
        pgm.key.set_repeat(5, 10)
        self.scr_width = 640
        self.scr_height = 427
        self.floor = 350
        self.ceil = 10
        self.image_path = "background.jpg"
        self.image = pgm.image.load(self.image_path)
        self.image = pgm.transform.scale(self.image, (self.scr_width, self.scr_height))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = [0, 0]
        self.belka = Belka(self)
        self.oreh = Oreh(self)
        self.actions = {
            0: self.belka.move_left,
            1: self.belka.stay_away,
            2: self.belka.move_right
        }
        self.num_actions = len(self.actions)


    def restart(self):
        self.last_frames = collections.deque(maxlen=4)
        self.screen = pgm.display.set_mode((self.scr_width, self.scr_height), 0, 32)
        self.konec_igry = False
        self.reward = 0
        self.belka.reset()
        self.oreh.reset()


    def update(self, belka_action):
        pgm.event.pump()
        self.screen.blit(self.image, self.rect)
        self.actions[belka_action]()
        self.oreh.fall()
        self.reward = 0
        if self.oreh.rect.top >= self.floor - self.oreh.height // 2:
            if self.oreh.rect.colliderect(self.belka.rect):
                self.reward = 1
            else:
                self.reward = -1
            self.konec_igry = True
        pgm.display.flip()
        self.last_frames.append(pgm.surfarray.array2d(self.screen))
        return self.get_last_frames(), self.reward, self.konec_igry


    def get_last_frames(self):
        return np.array(list(self.last_frames), dtype='uint8')


class IgrovoyObject:
    def __init__(self, igra, width, height, image_path, velocity):
        self.igra = igra
        self.image_path = image_path
        self.width = width
        self.height = height
        self.image = pgm.image.load(self.image_path)
        self.image = pgm.transform.scale(self.image, (self.width, self.height))
        self.rect = self.image.get_rect()
        self.velocity = velocity


    def move_left(self):
        self.rect.move_ip(-self.velocity, 0)
        if self.rect.left < 0:
            self.rect.left = self.velocity
        self.igra.screen.blit(self.image, self.rect)


    def move_right(self):
        self.rect.move_ip(self.velocity, 0)
        if self.rect.left > self.igra.scr_width - self.width:
            self.rect.left = self.igra.scr_width - self.width - self.velocity
        self.igra.screen.blit(self.image, self.rect)


    def stay_away(self):
        self.igra.screen.blit(self.image, self.rect)


class Belka(IgrovoyObject):
    def __init__(self, igra):
        super().__init__(igra, 100, 80, "belka.png", 10)
        self.start_position = (200, 350)


    def reset(self):
        self.rect.left, self.rect.top = self.start_position
        self.igra.screen.blit(self.image, self.rect)


class Oreh(IgrovoyObject):
    def __init__(self, igra):
        super().__init__(igra, 30, 30, "oreh.png", 5)


    def reset(self):
        self.rect.left, self.rect.top = (random.randint(0, self.igra.scr_width - self.width), self.igra.ceil)
        self.igra.screen.blit(self.image, self.rect)


    def fall(self):
        self.rect.move_ip(0, self.velocity)
        self.igra.screen.blit(self.image, self.rect)
