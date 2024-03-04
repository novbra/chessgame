"""
主界面，处理输入以及显示当前游戏状态
"""
import pygame
import Chessbasic

# 全局变量
HEIGHT = 960
WIDTH = 960
PieceSIZE = HEIGHT // 8  # 棋子尺寸
img = {}  # 棋子图片集合

pygame.init()

# 初始化图片,一个循环把所有的图片加载进img这个集合里
def Loadimages():
    pieces = ["wr", "wn", "wb", "wq", "wk", "bb", "bn", "br", "bp", "wp", "bk", "bq"]
    for piece in pieces:
        img[piece] = pygame.transform.scale(pygame.image.load("image/" + piece + ".png"),(PieceSIZE,PieceSIZE))

# 游戏初始化
def main():
    # 创建窗口
    screen = pygame.display.set_mode((HEIGHT, WIDTH))
    # 设置标题
    pygame.display.set_caption('AI Chess')
    # 加载图片
    Loadimages()


    clock = pygame.time.Clock()
    gamestate = Chessbasic.GameState()  # 棋盘状态
    validmoves = gamestate.Getvalidmove() # 合法的落子位置集合
    movemade = False # 判断是否发生合法移动
    selected = ()  # 存储被选中的方块（row，col）
    clicked = []  # 存储用户点击的方块[(4,2),(5,3)]
    running = True

    # 游戏主循环
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #  鼠标点击事件
            elif event.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos()  # 捕获鼠标点击位置
                row = location[1]//PieceSIZE  # 位置整除8，获得点击的是第几块的数据
                column = location[0]//PieceSIZE

                # selected代表当前用户点击的方块的位置，clicked代表历史点击的方块的位置的的集合
                if selected == (row, column):  # 点击了相同的方块
                    selected = ()  # 清空（取消）
                    clicked = []
                else:
                    selected = (row, column)  # 点击了不同的方块，将方块的数据存入clicked中
                    clicked.append(selected)
                if len(clicked) == 2 : # 当clicked中存在两个不同的位置时，即将发生棋子的移动
                    move = Chessbasic.Move(clicked[0], clicked[1], gamestate.board)
                    print(move.getChess())
                    if move in validmoves:  #  如果该位置属于合法落子位置集合
                        gamestate.Piecemove(move)  #  移动棋子
                        movemade = True     #  表示发生了移动
                        selected = ()  # 移动完棋子，清空记录
                        clicked = []
                    else:
                        clicked =[selected]  #把当前的点击次数设置为选择当前方块


            # 键盘事件
            elif event.type == pygame.KEYDOWN:
                # 按下Z键，撤回一步
                if event.key == pygame.K_z:
                    gamestate.Pieceundo()
                    movemade = True

        if movemade : #  如果移动发生了，重新获得可落子位置
            validmoves = gamestate.Getvalidmove()
            movemade = False

        Drawgame(screen, gamestate)
        pygame.display.flip()

# 绘制游戏
def Drawgame(screen, gamestate):
    Drawboard(screen)
    Drawpieces(screen, gamestate.board)

# 绘制棋盘
def Drawboard(screen):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for row in range(8):
        for column in range(8):
            color = colors[(row + column) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(column * PieceSIZE, row * PieceSIZE, PieceSIZE, PieceSIZE))
# 绘制棋子
def Drawpieces(screen, board):
    for row in range(8):
        for column in range(8):
            piece = board[row][column]
            if piece != "--":
                screen.blit(img[piece], pygame.Rect(column * PieceSIZE, row * PieceSIZE, PieceSIZE, PieceSIZE))





# 运行
if __name__ == '__main__':
    main()