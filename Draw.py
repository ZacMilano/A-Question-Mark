import pygame
import sys


def display_img(img):
    screen = pygame.display.set_mode((280, 280))
    screen.fill((0, 0, 0))

    for y in range(0, 28):
        for x in range(0, len(img)//28): # Assumes image height of exactly 28
            cur_val = img[(28 * y) + x]
            cur_color = (cur_val, cur_val, cur_val)
            pygame.draw.rect(screen, cur_color, (x * 10, y * 10, 10, 10), 0)

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # sys.exit()
