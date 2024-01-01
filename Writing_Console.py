import pygame, sys 
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 630
WINDOSIZEY = 480
# Initialize pygame
pygame.init()

BOUNDARYINC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

IMAGESAVE = False
MODEL = load_model("bestmodel.h5")
LABELS = {0: "Zero", 1:"One", 
          2:"Two", 3:"Three", 
          4:"Four", 5:"Five", 
          6:"Six", 7: "Seven",
          8:"Eight", 9:"Nine"}

# FONT = pygame.font.Font("freesansbold.tff", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOSIZEY))

pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt =1
PREDICT = True
# ... (previous code remains unchanged)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_Y, rect_max_Y = max(number_ycord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_ycord[-1] + BOUNDARYINC)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("Image.png", img_arr)
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                print(label)  # Adjust this line based on where you want to display the label

                # testSurface = FONT.render(label, True, RED, WHITE)
                # textRecObj = testSurface.get_rect()
                # textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_Y
                #
                # DISPLAYSURF.blit(testSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)
            elif event.key == K_BACKSPACE:
                # Remove the last drawn line when the backspace key is pressed
                DISPLAYSURF.fill(BLACK)
                for i in range(len(number_xcord) - 1):
                    pygame.draw.line(DISPLAYSURF, WHITE, (number_xcord[i], number_ycord[i]), (number_xcord[i + 1], number_ycord[i + 1]), 4)

    pygame.display.update()
