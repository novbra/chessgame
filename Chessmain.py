"""
主界面，处理输入以及显示当前游戏状态
"""
import pygame
import pygame_menu
import Chessbasic, AI
import sys
# 线程
from multiprocessing import Process, Queue
# 定义颜色
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
# 全局变量
HEIGHT = 930
WIDTH = 930
MOVELOGHEIGHT = HEIGHT//2
MOVELOGWIDTH = 450
SCROLLBAR_WIDTH = 10
SCROLL_SPEED = 20
PieceSIZE = HEIGHT // 8  # 棋子尺寸
img = {}  # 棋子图片集合
menu = None

pygame.init()
# 菜单项
menu_items = {'difficulty': ['Simple', 'ordinary','difficult'],
              'set up': ['User', 'quit']
              }
# 当前激活的菜单
active_menu = None
# 绘制菜单栏
def draw_menu_bar(screen, active_menu, menu_items):
    font = pygame.font.Font(None, 30)
    background_color = (255, 255, 255)  # 白色背景，RGB格式
    menu_height = font.get_height()
    for index, (key, items) in enumerate(menu_items.items()):
        # 绘制菜单项背景
        menu_item_background = pygame.Rect(index * 200, 0, 200, menu_height)
        pygame.draw.rect(screen, background_color, menu_item_background)

        # 绘制菜单项文字
        text_surface = font.render(key, True, (0, 0, 0))  # 黑色文字
        screen.blit(text_surface, (index * 200, 0))

        if active_menu == key:
            for i, item in enumerate(items):
                # 绘制下拉菜单背景
                dropdown_background = pygame.Rect(index * 200, (i + 1) * menu_height, 200, menu_height)
                pygame.draw.rect(screen, background_color, dropdown_background)

                # 绘制下拉菜单项文字
                item_surface = font.render(item, True, (0, 0, 0))  # 黑色文字
                screen.blit(item_surface, (index * 200, (i + 1) * menu_height))

# 定义开始菜单中的函数
def start_the_game():
    global gamestate, validmoves, movemade, animate, gameover, running, menu
    # 初始化游戏状态
    gamestate = Chessbasic.GameState()
    # 获取所有有效的移动
    validmoves,trainvalidmoves = gamestate.Getvalidmove()
    # 设置移动是否已经发生的标志
    movemade = False
    # 设置是否应该显示动画的标志
    animate = False
    # 设置游戏是否结束的标志
    gameover = False
    #


    # 设置游戏运行状态为True，开始游戏循环
    running = True
    if menu is not None:  # 在调用之前检查menu是否已经被赋值
     # 关闭菜单
      menu.disable()
      menu.full_reset()  # 重置菜单状态，确保菜单不会阻止游戏界面的渲染


# 初始化图片,一个循环把所有的图片加载进img这个集合里
def Loadimages():
    pieces = ["wr", "wn", "wb", "wq", "wk", "bb", "bn", "br", "bp", "wp", "bk", "bq"]
    for piece in pieces:
        img[piece] = pygame.transform.scale(pygame.image.load("image/" + piece + ".png"),(PieceSIZE,PieceSIZE))

# 游戏初始化
def  main_game_loop():
    global menu,running, active_menu
    active_menu = None  # 在这里初始化active_menu变量
    # 创建窗口
    screen = pygame.display.set_mode((WIDTH + MOVELOGWIDTH,HEIGHT ))
    # 设置标题
    pygame.display.set_caption('AI Chess')
    # 加载图片
    Loadimages()
    movelogfont = pygame.font.SysFont("Arial", 24, False, False)

    clock = pygame.time.Clock()
    gamestate = Chessbasic.GameState()  # 棋盘状态
    validmoves,trainvalidmoves = gamestate.Getvalidmove() # 合法的落子位置集合
    movemade = False # 判断是否发生合法移动
    selected = ()  # 存储被选中的方块（row，col）
    clicked = []  # 存储用户点击的方块[(4,2),(5,3)]
    player1 =True# 如果是人类在操作白棋，则其值为True
    player2 =False# 如果是人类在操作黑棋，则其值为True
    AIThinking = False
    moveFinderProcess = None
    moveUndone =False
    animate = False #flag variable for when we should animate a move
    gameover = False
    scroll_offset = 0  # 初始化滚动偏移量
    button1 = Button(WIDTH + 175, HEIGHT // 2 + 100, 100, 50, "Undo")
    button2 = Button(WIDTH + 175, HEIGHT // 2 + 200, 100, 50, "Reset")
    print(trainvalidmoves)
    running = True
    # 显示菜单
    # 创建菜单
    menu = pygame_menu.Menu('welcome', WIDTH + MOVELOGWIDTH, HEIGHT,
                            theme=pygame_menu.themes.THEME_BLUE)
    menu.add.text_input('name :', default='MN Master')
    menu.add.button('play', start_the_game)
    menu.add.button('exit', pygame_menu.events.EXIT)
    menu.mainloop(screen)

    # 游戏主循环
    while running:
        humanturn = (gamestate.IswTomove and player1) or (not gamestate.IswTomove and player2)  # 是否是人类的回合
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 如果是左键点击
                    mouse_x, mouse_y = event.pos
                    menu_width = 200  # 设置菜单项的宽度
                    for index, (key, items) in enumerate(menu_items.items()):
                        # 检查鼠标点击是否在菜单项的水平范围内
                        if index * menu_width < mouse_x < (index + 1) * menu_width:
                            # 检查鼠标点击是否在菜单项的垂直范围内
                            if 0 < mouse_y < 20:
                                active_menu = key if active_menu != key else None
                                break
                    if WIDTH + 175 < mouse_x < WIDTH + 275:
                        if HEIGHT // 2 + 100 < mouse_y < HEIGHT // 2 + 150:
                            animate = False
                            gamestate.Pieceundo()
                            movemade = True
                            gameover = False
                            if not (player1 and player2):
                                gamestate.Pieceundo()
                        elif HEIGHT // 2 + 200 < mouse_y < HEIGHT // 2 + 250:
                            gamestate = Chessbasic.GameState()
                            validmoves,trainvalidmoves = gamestate.Getvalidmove()
                            selected = ()
                            clicked = []
                            movemade = True
                            animate = False
                            gameover = False
                    elif mouse_x < WIDTH and not gameover:
                        row = mouse_y // PieceSIZE
                        column = mouse_x // PieceSIZE
                        if selected == (row, column) or column >= 8:
                            selected = ()
                            clicked = []
                        else:
                            selected = (row, column)
                            clicked.append(selected)
                        if len(clicked) == 2 and humanturn:
                            move = Chessbasic.Move(clicked[0], clicked[1], gamestate.board)
                            print(move.getChess())
                            if move in validmoves:
                                gamestate.Piecemove(move)
                                movemade = True
                                animate = True
                                selected = ()
                                clicked = []
                            if not movemade:
                                clicked = [selected]

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    if (player1 == True and player2 == False) or (player2 == True and player1 == False):
                        animate = False
                        gamestate.Pieceundo()
                        gamestate.Pieceundo()
                        movemade = True
                        gameover = False
                        if AIThinking:
                            moveFinderProcess.terminate()
                            AIThinking = False
                        moveUndone = True
                    else:
                        animate = False
                        gamestate.Pieceundo()
                        movemade = True
                        gameover = False
                        if AIThinking:
                            moveFinderProcess.terminate()
                            AIThinking = False
                        moveUndone = True
                elif event.key == pygame.K_r:  # 按下r键，复原棋盘
                        gamestate = Chessbasic.GameState()
                        validmoves,trainvalidmoves = gamestate.Getvalidmove()
                        selected = ()
                        clicked = []
                        movemade = True
                        animate = False
                        gameover = False
                        if AIThinking:
                            moveFinderProcess.terminate()
                            AIThinking = False
                        moveUndone = True



        #AI 移动
        if not gameover and not humanturn and not moveUndone:
            if player1 ==False and player2==False:
                AImove = AI.findrandommove(validmoves)
                gamestate.Piecemove(AImove)
                movemade = True
                animate = False
            else:
                if not AIThinking:
                    AIThinking = True
                    print("thinking")
                    returnQuene = Queue()
                    # 在线程之间传输数据
                    moveFinderProcess = Process(target=AI.findminmaxmove, args=(gamestate, validmoves, returnQuene,AI.limittime))
                    moveFinderProcess.start()
                    # 调用findminmaxmove（gs.validMoves,returnQuene）
                if not moveFinderProcess.is_alive():
                    print("done thinking")
                    AImove = returnQuene.get()
                    if AImove is None:
                        AImove = AI.findrandommove(validmoves)
                    gamestate.Piecemove(AImove)
                    movemade = True
                    animate = False
                    AIThinking = False






        if movemade : #  如果移动发生了，重新获得可落子位置
            if animate:
                animateMove(gamestate.movelog[-1], screen, gamestate.board,clock) #移动轨迹
            validmoves,trainvalidmoves = gamestate.Getvalidmove()
            movemade = False
            animate = True
            moveUndone = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            scroll_offset += SCROLL_SPEED
        if keys[pygame.K_s]:
            scroll_offset -= SCROLL_SPEED

        scroll_offset = max(0, min(scroll_offset, MOVELOGHEIGHT))  # 确保偏移在边界内
        Drawgame(screen, gamestate,validmoves,selected,movelogfont,scroll_offset,button1,button2)

        if gamestate.checkMate == True:
            gameover = True
            if gamestate.IswTomove:
                drawEndgameText(screen,'Black wins by checkmate')
            else:
                drawEndgameText(screen,'White wins by checkmate')
        elif gamestate.staleMate:
            gameover = True
            drawEndgameText(screen,'Stalemate')



        pygame.display.flip()
        # 游戏结束后可以再次显示菜单或退出
    pygame.quit()
    sys.exit()

'''
highlight square selected and move for piece selected 突出显示选定的方块并为选定的块移动
'''

def highlightSquares(screen,gamestate,validmoves,sqSelected):
    if sqSelected!=():
        row,column=sqSelected
        if gamestate.board[row][column][0]==('w'if gamestate.IswTomove else 'b'):#sqSelected is a piece that can be moved
            #highlight selected square
            s=pygame.Surface((PieceSIZE,PieceSIZE))
            s.set_alpha(100)
            s.fill(pygame.Color('blue'))
            screen.blit(s,(column * PieceSIZE, row * PieceSIZE+20))
            #highlight moves from that square
            s.fill(pygame.Color('green'))
            for move in validmoves:
                if move.startrow==row and move.startcolumn==column:
                    screen.blit(s,(move.endcolumn*PieceSIZE,move.endrow*PieceSIZE+20))

def draw_dropdown_menu(screen, background_color):
    # 定义下拉菜单栏的位置和大小
    menu_rect = pygame.Rect(0,0,WIDTH, 30)
    # 使用背景色填充菜单栏区域
    pygame.draw.rect(screen, background_color, menu_rect)

# 绘制游戏
def Drawgame(screen, gamestate,validmoves,sqSelected,movelogfont,scroll_offset,button1,button2):
    background_color = "white"  # 红色背景
    draw_dropdown_menu(screen, background_color)
    Drawboard(screen)
    highlightSquares(screen,gamestate,validmoves,sqSelected)
    Drawpieces(screen, gamestate.board)
    draw_menu_bar(screen, active_menu,menu_items)  # 首先绘制菜单栏
    Drawmovelog(screen, gamestate,movelogfont,scroll_offset)
    Drawbuttonpart(screen)
    button1.draw(screen)
    button2.draw(screen)
# 绘制棋盘
def Drawboard(screen):
    global colors
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for row in range(8):
        for column in range(8):
            color = colors[(row + column) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(column * PieceSIZE, row * PieceSIZE+20, PieceSIZE, PieceSIZE))
# 绘制棋子
def Drawpieces(screen, board):
    for row in range(8):
        for column in range(8):
            piece = board[row][column]
            if piece != "--":
                screen.blit(img[piece], pygame.Rect(column * PieceSIZE, row * PieceSIZE+20, PieceSIZE, PieceSIZE))

def Drawmovelog(screen, gamestate, font, scroll_offset):
    # 定义移动日志的矩形区域
    movelogRect = pygame.Rect(WIDTH, 0, MOVELOGWIDTH - SCROLLBAR_WIDTH, MOVELOGHEIGHT)
    pygame.draw.rect(screen, pygame.Color("white"), movelogRect)

    # 获取移动日志并创建文本列表
    moveLog = gamestate.movelog
    moveTexts = []
    for i in range(0, len(moveLog), 2):
        moveString = str(i // 2 + 1) + ". " + moveLog[i].getChess() + " "
        if i + 1 < len(moveLog):  # 如果存在下一步，则添加到字符串中
            moveString += moveLog[i + 1].getChess() + " "
        moveTexts.append(moveString)

    # 设置每行显示的移动数和其他参数
    movesPerrow = 3
    padding = 5
    lineSpace = 2

    # 计算实际文本行数，加1是为了确保总行数不为零
    total_lines = len(moveTexts) // movesPerrow + 1
    # 根据 scroll_offset 调整 textY 的初始位置
    textY = padding - scroll_offset

    # 计算滚动条手柄的高度
    content_height = max(1, len(moveTexts) / movesPerrow * (font.get_height() + lineSpace))
    handle_height = max(10, MOVELOGHEIGHT * (MOVELOGHEIGHT / content_height))
    handle_height = min(handle_height, MOVELOGHEIGHT)

    # 计算滚动条手柄的位置
    if content_height != 0:
        handle_position = (scroll_offset / content_height) * (MOVELOGHEIGHT - handle_height)
        handle_position = max(0, min(handle_position, MOVELOGHEIGHT - handle_height))
    else:
        handle_position = 0

    # 在移动日志区域绘制文本
    for i in range(0, len(moveTexts), movesPerrow):
        text = ""
        for j in range(movesPerrow):
            if i + j < len(moveTexts):
                text += moveTexts[i + j]
        textObject = font.render(text, True, pygame.Color('Black'))
        textLocation = movelogRect.move(padding, textY)
        screen.blit(textObject, textLocation)
        textY += textObject.get_height() + lineSpace

    # 绘制滚动条
    scrollbarRect = pygame.Rect(WIDTH + MOVELOGWIDTH - SCROLLBAR_WIDTH, 0, SCROLLBAR_WIDTH, MOVELOGHEIGHT)
    pygame.draw.rect(screen, pygame.Color('gray'), scrollbarRect)

    # 绘制滚动条手柄
    scrollbar_handle = pygame.Rect(WIDTH + MOVELOGWIDTH - SCROLLBAR_WIDTH, handle_position, SCROLLBAR_WIDTH, handle_height)
    pygame.draw.rect(screen, pygame.Color('darkgray'), scrollbar_handle)

def Drawbuttonpart(screen):
    movelogRect = pygame.Rect(WIDTH, HEIGHT//2, MOVELOGWIDTH, MOVELOGHEIGHT)
    pygame.draw.rect(screen, pygame.Color("white"), movelogRect)

def drawEndgameText(screen, text):
    font =pygame.font.SysFont("Helvitca",64,True, False)
    textObject =font.render(text,0, pygame.Color('Black'))
    textLocation = pygame.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH/2 - textObject.get_width()/2, HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject,textLocation)
    textObject =font.render(text,0,pygame.Color('Black'))
    screen.blit(textObject, textLocation.move(2, 2))


'''
animating a move
'''

def animateMove(move, screen, board, clock):
    global colors

    dR = move.endrow - move.startrow
    dC = move.endcolumn - move.startcolumn

    framesPerSquare = 10  # 移动一个格子所需的帧数，调整framesPerSquare的值以控制动画的速度。
    frameCount =(abs(dR)+abs(dC))*framesPerSquare

    for frame in range(frameCount +1):
        r,c=(move.startrow + dR*frame/frameCount,move.startcolumn + dC*frame/frameCount)
        Drawboard(screen)
        Drawpieces(screen, board)
        #erase the piece moved from its ending square
        color = colors[(move.endrow + move.endcolumn)% 2]
        endSquare = pygame.Rect(move.endcolumn*PieceSIZE, move.endrow*PieceSIZE+20, PieceSIZE, PieceSIZE)
        pygame.draw.rect(screen,color,endSquare)
        if move.pieceend != '--':
            if move.isEnpassantMove:
                enPassantRow = move.endrow + 1 if move.pieceend[0] =="b" else move.endrow -1
                endSquare = pygame.Rect(move.endcolumn * PieceSIZE, enPassantRow * PieceSIZE+20, PieceSIZE, PieceSIZE)
            screen.blit(img[move.pieceend],endSquare)
        screen.blit(img[move.piecestart], pygame.Rect(c * PieceSIZE, r * PieceSIZE+20, PieceSIZE, PieceSIZE))
        pygame.display.flip()
        clock.tick(60)



class Button:
    def __init__(self, x, y, width, height, text, color=(200, 200, 200), hover_color=(150, 150, 150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.font = pygame.font.Font(None, 36)
        self.image = self.create_button_image()

    def create_button_image(self):
        button_surface = pygame.Surface((self.rect.width, self.rect.height))
        button_surface.fill(self.color)

        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(self.rect.width / 2, self.rect.height / 2))

        button_surface.blit(text_surface, text_rect)

        return button_surface

    def draw(self, screen):
        screen.blit(self.image, self.rect)



# 运行
if __name__ == '__main__':

    main_game_loop()
