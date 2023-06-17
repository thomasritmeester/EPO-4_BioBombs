# =============================================================================
# import pygame
# import random
# import time
# 
# output=[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
# # Initialize Pygame
# pygame.init()
# 
# # Set the window size
# window_size = (600, 400)
# screen = pygame.display.set_mode(window_size)
# clock = pygame.time.Clock()
# surface = pygame.surface.Surface(window_size)
# def text(words, color, fnt="Arial", size=100):
# 	font = pygame.font.SysFont(fnt, size)
# 	txt = font.render(words, 1, color)
# 	return txt
# 
# # Set the title of the window
# pygame.display.set_caption("stress detector")
# running = True
# stress = text("stress",(255,0,0) )
# nostress = text("no stress",(0,255,0))
# 
# def update():
#     surface.fill("BLACK")
#     surface.blit()
#     
#     
#     
# #def update():
# 	#pygame.display.update()
# 
# def show(writing, pos):
# 	screen.blit(writing, pos)
# 
# while running:
#     g=0
#     if output[0]==1:
#         stresslevel = stress
#     else:
#        stresslevel = nostress
# 
#     
# 
#     for a in output[1:]:
#         print("a: "+str(a))
#         g+=1
#         print("g: "+str(g))
#         
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#         if a==1:
#             stresslevel = stress
#         else:
#            stresslevel = nostress
#         show(stresslevel, (100, 100))
#         time.sleep(1)
#         update()
# ============================================================================

# =============================================================================
# import pygame
# import time
#  
# # Initialize Pygame
# pygame.init()
#  
# # Set the window size
# window_size = (600, 400)
# screen = pygame.display.set_mode(window_size)
# clock = pygame.time.Clock()
# surface = pygame.surface.Surface(window_size)
# output=[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
# 
# def text(words, color, fnt="Arial", size=20):
# 	font = pygame.font.SysFont(fnt, size)
# 	txt = font.render(words, 1, color)
# 	return txt
#  
# # Set the title of the window
# pygame.display.set_caption("stress detection c4")
# running = True
#  
# def update(speed):
# 	pygame.display.update()
# 	clock.tick(speed)
#  
# def show(writing, pos):
# 	screen.blit(writing, pos)
# 
# while running:
#     for x in output:
#        if output[x]== 1:
#            text = text('stress', color=(255, 0, 0))
#        else:
#            text = text('no stress', color=(0, 255, 0))
#            for event in pygame.event.get():
#                if event.type == pygame.QUIT:
#                    running = False
#                    show(text, (100, 100))
#                    update(3)
# =============================================================================


import pygame
import time


pygame.init()

output=[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

screen = pygame.display.set_mode((600, 400))
font = pygame.font.SysFont(None, 30)

def draw_txt(text, font, color, x, y):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

run = True 
while run:
    screen.fill((0,0,0))
    for x in output:
        draw_txt(str(x), font, (0, 255, 0), 220, 150)
        time.sleep(2)
        screen.fill((255,255,255))
        time.sleep(2)
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            run=False
            
        pygame.display.flip()

pygame.quit()

