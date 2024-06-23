import pygame
import random

# Constants
w = 8
width, height = 640, 240
columns, rows = width // w, height // w

# Initialize board
def create2DArray(columns, rows):
    arr = [[0 for _ in range(rows)] for _ in range(columns)]
    return arr

board = create2DArray(columns, rows)

def setup():
    for i in range(1, columns - 1):
        for j in range(1, rows - 1):
            board[i][j] = random.randint(0, 1)

def draw(window):
    next_board = create2DArray(columns, rows)

    for i in range(1, columns - 1):
        for j in range(1, rows - 1):
            neighbor_sum = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    neighbor_sum += board[i + k][j + l]
            neighbor_sum -= board[i][j]

            if board[i][j] == 1 and neighbor_sum < 2:
                next_board[i][j] = 0
            elif board[i][j] == 1 and neighbor_sum > 3:
                next_board[i][j] = 0
            elif board[i][j] == 0 and neighbor_sum == 3:
                next_board[i][j] = 1
            else:
                next_board[i][j] = board[i][j]

    for i in range(columns):
        for j in range(rows):
            color = (255 - board[i][j] * 255, 255 - board[i][j] * 255, 255 - board[i][j] * 255)
            pygame.draw.rect(window, color, (i * w, j * w, w - 1, w - 1))

    return next_board

def main():
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Game of Life")

    setup()

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill((255, 255, 255))
        global board
        board = draw(window)
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
