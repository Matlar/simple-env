import pygame

class KeyHandler:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((100, 100))

    def get_action(self):
        action = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 0
                elif event.key == pygame.K_d:
                    action = 1
                elif event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_a:
                    action = 3
        return action
