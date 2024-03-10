"""
主界面，处理输入以及显示当前游戏状态
"""
import pygame
import Chessbasic, AI

# 全局变量
HEIGHT = 960
WIDTH = 960
MOVELOGHEIGHT = HEIGHT
MOVELOGWIDTH = 400
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
    screen = pygame.display.set_mode((WIDTH + MOVELOGWIDTH,HEIGHT ))
    # 设置标题
    pygame.display.set_caption('AI Chess')
    # 加载图片
    Loadimages()
    movelogfont = pygame.font.SysFont("Arial", 24, False, False)

    clock = pygame.time.Clock()
    gamestate = Chessbasic.GameState()  # 棋盘状态
    validmoves = gamestate.Getvalidmove() # 合法的落子位置集合
    movemade = False # 判断是否发生合法移动
    selected = ()  # 存储被选中的方块（row，col）
    clicked = []  # 存储用户点击的方块[(4,2),(5,3)]
    player1 =False# 如果是人类在操作白棋，则其值为True
    player2 =False # 如果是人类在操作黑棋，则其值为True
    animate = False #flag variable for when we should animate a move
    gameover = False

    running = True

    # 游戏主循环
    while running:
        humanturn = (gamestate.IswTomove and player1) or (not gamestate.IswTomove and player2)# 是否是人类的回合
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                running = False
            #  鼠标点击事件
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not gameover and humanturn:
                    location = pygame.mouse.get_pos()  # 捕获鼠标点击位置
                    row = location[1]//PieceSIZE  # 位置整除8，获得点击的是第几块的数据
                    column = location[0]//PieceSIZE

                    # selected代表当前用户点击的方块的位置，clicked代表历史点击的方块的位置的的集合
                    if selected == (row, column) or column >=8 :  # 点击了相同的方块
                        selected = ()  # 清空（取消）
                        clicked = []
                    else:
                        selected = (row, column)  # 点击了不同的方块，将方块的数据存入clicked中
                        clicked.append(selected)
                    if len(clicked) == 2 : # 当clicked中存在两个不同的位置时，即将发生棋子的移动
                        move = Chessbasic.Move(clicked[0], clicked[1], gamestate.board)
                        print(move.getChess())
                        for i in range(len(validmoves)):
                            if move == validmoves[i]:  #  如果该位置属于合法落子位置集合
                                # animateMove(validmoves[i], screen, gamestate.board, clock) #移动轨迹
                                gamestate.Piecemove(validmoves[i])  #  移动棋子
                                movemade = True     #  表示发生了移动
                                animate = True     #开启动画
                                selected = ()  # 移动完棋子，清空记录
                                clicked = []
                        if not movemade:
                            clicked =[selected]  #把当前的点击次数设置为选择当前方块


            # 键盘事件
            elif event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_z :# 按下Z键，撤回一步
                    animate = False
                    gamestate.Pieceundo()
                    movemade = True
                    gameover = False
                if event.key == pygame.K_r :# 按下r键，复原棋盘
                    gamestate = Chessbasic.GameState()
                    validmoves = gamestate.Getvalidmove()
                    selected = ()
                    clicked = []
                    movemade = True
                    animate = False
                    gameover =False





        #AI 移动
        if not gameover and not humanturn:
            AImove = AI.findminmaxmove(gamestate,validmoves)
            if AImove is None:
                AImove = AI.findrandommove(validmoves)
            gamestate.Piecemove(AImove)
            movemade = True
            animate = False






        if movemade : #  如果移动发生了，重新获得可落子位置
            if animate:
                animateMove(gamestate.movelog[-1], screen, gamestate.board,clock) #移动轨迹
            validmoves = gamestate.Getvalidmove()
            movemade = False
            animate = False

        Drawgame(screen, gamestate,validmoves,selected,movelogfont)

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
            s.fill(pygame.Color('yellow'))
            screen.blit(s,(column * PieceSIZE, row * PieceSIZE))
            #highlight moves from that square
            s.fill(pygame.Color('green'))
            for move in validmoves:
                if move.startrow==row and move.startcolumn==column:
                    screen.blit(s,(move.endcolumn*PieceSIZE,move.endrow*PieceSIZE))



# 绘制游戏
def Drawgame(screen, gamestate,validmoves,sqSelected,movelogfont):
    Drawboard(screen)
    highlightSquares(screen,gamestate,validmoves,sqSelected)
    Drawpieces(screen, gamestate.board)
    Drawmovelog(screen, gamestate,movelogfont)

# 绘制棋盘
def Drawboard(screen):
    global colors
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

def Drawmovelog(screen, gamestate, font):
    movelogRect = pygame.Rect(WIDTH,0,MOVELOGWIDTH,MOVELOGHEIGHT)
    pygame.draw.rect(screen,pygame.Color("white"),movelogRect)
    moveLog = gamestate.movelog
    moveTexts = []
    for i in range (0,len(moveLog), 2):
        moveString = str(i//2+1) + ". "+moveLog[i].getChess() + " "
        if i + 1 <len(moveLog):
            moveString +=moveLog[i+1].getChess() + " "
        moveTexts.append(moveString)

    movesPerrow = 3
    padding = 5
    lineSpace = 2
    textY = padding
    for i in range(0,len(moveTexts),movesPerrow):
        text = ""
        for j in range(movesPerrow):
            if i+j < len(moveTexts):
                text += moveTexts[i+j]
        textObject =font.render(text,True, pygame.Color('Black'))
        textLocation = movelogRect.move(padding,textY)
        screen.blit(textObject,textLocation)
        textY += textObject.get_height() + lineSpace


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
        endSquare = pygame.Rect(move.endcolumn*PieceSIZE, move.endrow*PieceSIZE, PieceSIZE, PieceSIZE)
        pygame.draw.rect(screen,color,endSquare)
        if move.pieceend != '--':
            if move.isEnpassantMove:
                enPassantRow = move.endrow + 1 if move.pieceend[0] =="b" else move.endrow -1
                endSquare = pygame.Rect(move.endcolumn * PieceSIZE, enPassantRow * PieceSIZE, PieceSIZE, PieceSIZE)
            screen.blit(img[move.pieceend],endSquare)
        screen.blit(img[move.piecestart], pygame.Rect(c * PieceSIZE, r * PieceSIZE, PieceSIZE, PieceSIZE))
        pygame.display.flip()
        clock.tick(60)






# 运行
if __name__ == '__main__':
    main()
