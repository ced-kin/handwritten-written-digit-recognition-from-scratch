import neuralnetwok as nn
import numpy as np
import pygame
import matplotlib.pyplot as plt

net = nn.NeuralNetwork()
pygame.init()
screen = pygame.display.set_mode([600, 600])
screen.fill((115, 115, 115))
mouse = pygame.mouse
draw_surface = pygame.Rect(160, 200, 280, 280)
prediction_surface = pygame.Rect(160, 100, 280, 50)
clear_button = pygame.Rect(470, 315, 50, 50)
pygame.draw.rect(screen, (255, 255, 255), draw_surface)
pygame.draw.rect(screen, (255, 255, 255), prediction_surface)
pygame.draw.rect(screen, (255, 0, 0), clear_button)
font = pygame.font.Font('freesansbold.ttf', 40)
screen.blit(font.render('Cl', True, (0, 0, 0)), (470, 315))
prediction_text = "Prediction : ?"
screen.blit(font.render(prediction_text, True, (0, 0, 0)), (160, 100))
something_drawn = False


def make_matrix():
    all_pixels = np.zeros((28, 28))
    # will be a 28x28 list of how many black pixels there are in a 10x10 area from the 280x280 drawing surface
    # after creation, will need to be reshaped into a 784x1 matrix so that it can be used to make a prediction

    for y in range(28):
        y_offset = 200 + (y * 10)
        for x in range(28):
            x_offset = 160 + (x * 10)
            total = 0.0
            for i in range(10):
                for j in range(10):
                    pixel = screen.get_at((x_offset + i, y_offset + j))
                    # (255, 255, 255, 255) is white, (0, 0, 0, 255) is black
                    # need to divide by 100 to get scaled colour value
                    if pixel == (0, 0, 0, 255):
                        total += 1.0
            total /= 100.0
            all_pixels[y, x] = total

    # plt.imshow(all_pixels, cmap="gray")
    # plt.show()
    all_pixels = np.reshape(all_pixels, (784, 1))
    return all_pixels


def get_prediction(image):
    pred = net.predict([image])
    # print(pred)
    # print(np.argmax(pred))
    return pred


running = True
while running:
    left_pressed, middle_pressed, right_pressed = mouse.get_pressed()

    if left_pressed and draw_surface.collidepoint(pygame.mouse.get_pos()):
        pygame.draw.circle(screen, (0, 0, 0), (pygame.mouse.get_pos()), 10)
        something_drawn = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONUP and something_drawn:
            something_drawn = False
            image_matrix = make_matrix()
            prediction = get_prediction(image_matrix)
            pygame.draw.rect(screen, (255, 255, 255), prediction_surface)
            prediction_text = "Prediction : " + str(np.argmax(prediction))
            screen.blit(font.render(prediction_text, True, (0, 0, 0)), (160, 100))

        elif event.type == pygame.MOUSEBUTTONDOWN and clear_button.collidepoint(pygame.mouse.get_pos()):
            screen.fill((115, 115, 115))
            pygame.draw.rect(screen, (255, 255, 255), draw_surface)
            pygame.draw.rect(screen, (255, 255, 255), prediction_surface)
            pygame.draw.rect(screen, (255, 0, 0), clear_button)
            screen.blit(font.render('Cl', True, (0, 0, 0)), (470, 315))
            prediction_text = "Prediction : ?"
            screen.blit(font.render(prediction_text, True, (0, 0, 0)), (160, 100))

    pygame.display.flip()
pygame.quit()
