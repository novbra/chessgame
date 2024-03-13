"""
主界面，处理输入以及显示当前游戏状态
"""
import pygame
import Chessbasic
import AI
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
    screen = pygame.display.set_mode((WIDTH,HEIGHT ))
    # 设置标题
    pygame.display.set_caption('AI Chess')
    # 加载图片
    Loadimages()

    clock = pygame.time.Clock()

    gamestate = Chessbasic.GameState()  # 棋盘状态 调用Chessbasic.GameState构造函数生成

    validmoves = gamestate.Getvalidmove(False) # 合法的落子位置集合

    movemade = False # 判断是否发生合法移动

    selected = ()  # 存储被选中的方块（row，col）

    clicked = []  # 存储用户点击的方块[(4,2),(5,3)]

    player1 =True# 如果是人类在操作白棋，则其值为True 默认白棋是人类操作

    if player1==True:#如果白棋是玩家操作，那么AI提示默认将开启
        aihint=True

    tips_moves = {}  # AI计算后的前几位最佳走法

    player2 =False # 如果是人类在操作黑棋，则其值为True

    animate = False #flag variable for when we should animate a move

    DEFAULT_SEARCH_DEPTH=4 #默认搜索树搜索到第4层

    gameover = False

    running = True #游戏主线程循环标志

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
                    if selected == (row, column):  # 点击了相同的方块
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
                if event.key == pygame.K_r :# 按下r键，复原棋盘
                    gamestate = Chessbasic.GameState()
                    validmoves = gamestate.Getvalidmove(False)
                    selected = ()
                    clicked = []
                    movemade = True
                    animate = False

        #AI 移动

        if not gameover:
            if not humanturn: #机器轮
                AImove = AI.test_get_best_move(gamestate,4)
                if AImove is None:
                    print("AI没有提供落子方案")
                    AImove = AI.randommove(validmoves)
                gamestate.Piecemove(AImove)
                movemade = True
                animate = False

                tips_moves = {}  # 机器下完棋后，将AI提示字典清空，以便下轮玩家回合生成
            else: # 玩家轮
                # print("玩家轮，自动打印价值")
                #获得对应位置价值，交由drawgame来绘制
                #  执行这条语句的时候，对手还没走步，因此最后的AI提示是给对手提示的
                if len(tips_moves)==0:#如果AI落法字典为空，表示此次玩家本回合并未生成AI走棋结果;生成以后，在玩家下完棋后会重置tips_moves字典为空
                    tree=AI.make_game_tree(gamestate,DEFAULT_SEARCH_DEPTH,1)
                    print(gamestate.board)
                    child_list=tree.root.children
                    child_list.sort(key=lambda item: item.val) #按照降序来排序，子结点val是负数，利益越大值越小，因此从小到大排序的即可
                    for child in child_list[:1]: #列表切片
                        print("(",child.move.startrow,",",child.move.startcolumn,")","->","(",child.move.endrow,",",child.move.endcolumn,")",-child.val)
                        tips_moves[child.move]=-child.val #受negamax算法影响，depth为1的结点的val就是负向，需要取反

        if movemade : #  如果移动发生了，重新获得可落子位置
            if animate:
                animateMove(gamestate.movelog[-1], screen, gamestate.board,clock) #移动轨迹
            validmoves = gamestate.Getvalidmove(False)

            movemade = False
            animate = False

        Drawgame(screen, gamestate,validmoves,selected,tips_moves) #新增

        if gamestate.checkMate == True:
            gameover = True
            if gamestate.IswTomove:
                drawText(screen,'Black wins by checkmate')
            else:
                drawText(screen,'White wins by checkmate')
        elif gamestate.staleMate: #和棋/僵持局面判断
            gameover = True
            drawText(screen,'Stalemate')

        pygame.display.flip()

'''
highlight square selected and move for piece selected 突出显示选定的方块并为选定的块移动
'''

def highlightSquares(screen,gamestate,validmoves,sqSelected):
    if sqSelected!=():
        row,column=sqSelected # 获取被选中的行列
        if gamestate.board[row][column][0]==('w'if gamestate.IswTomove else 'b'):#sqSelected is a piece that can be moved
            #highlight selected square 高亮选中区域

            s=pygame.Surface((PieceSIZE,PieceSIZE)) #棋子起点高亮

            s.set_alpha(100)

            s.fill(pygame.Color('yellow'))

            screen.blit(s,(column * PieceSIZE, row * PieceSIZE)) #screen.blit()方法将一个图片放到一个对象上面

            #highlight moves from that square

            s.fill(pygame.Color('green'))

            for move in validmoves:
                if move.startrow==row and move.startcolumn==column: # 找到起点正是是选中的位置，然后再绘制其终点
                    screen.blit(s,(move.endcolumn*PieceSIZE,move.endrow*PieceSIZE)) #棋子落地点高亮
'''
@method: highlight AI moves
@author Yan
@time: 2024/3/11 1:30
'''
def highlight_AI_hints(screen,gamestate,moves):
    if gamestate.IswTomove and not len(moves) ==0: #弱逻辑，后期需要修改, 默认白棋为人类操纵
        #绘制AI走法提示
        color_set=["#8e44ad","#16a085","#f39c12"]
        color_index=0
        start_xy={}
        end_xy={}
        for move in moves.keys():
            color=None
            end_offset=0
            s_xy=move.startrow,move.startcolumn
            e_xy=move.endrow,move.endcolumn
            if start_xy.get(s_xy) is not None:
                #读取该点已存储的颜色
                color=start_xy.get(s_xy)
            else:
                #生成该点颜色
                color = pygame.Color(color_set[color_index])
                #存储当前坐标的颜色
                start_xy[s_xy]=color

            #落地点不同棋子预测位置图标进行错开绘画
            if end_xy.get(e_xy) is None:
                end_xy[e_xy]=0

            end_offset=end_xy[e_xy]
            end_xy[e_xy]+=1



            val = moves[move]  # 该步价值

            #终点参数
            offset_=5
            radius=10
            offset_between=end_offset*radius*4.5
            circle_center =offset_between+offset_+move.endcolumn * PieceSIZE+radius, offset_+move.endrow * PieceSIZE+radius
            text_center=offset_between+offset_+move.endcolumn * PieceSIZE+2.5*radius, offset_+move.endrow * PieceSIZE+0.5*radius
            #绘制起点格子
            color_index=(color_index+1)%3
            pygame.draw.circle(screen, color, (offset_+move.startcolumn * PieceSIZE+radius, offset_+move.startrow * PieceSIZE+radius), radius,0,True,True,True,True )

            # pygame.draw.rect(screen, color, (move.startcolumn * PieceSIZE, move.startrow * PieceSIZE,PieceSIZE,PieceSIZE),3)

            # 绘制终点圆形图案
            pygame.draw.circle(screen, color, circle_center, radius,radius-1,False,True,False,True) # 棋子终点画圆标注
            #绘制终点价值
            font = pygame.font.SysFont("宋体", 16,bold=True)
            s = font.render(str(val), True, color)
            screen.blit(s,text_center)






# 绘制游戏 绘制总函数
def Drawgame(screen, gamestate,validmoves,sqSelected,tip_moves):
    Drawboard(screen)
    highlightSquares(screen,gamestate,validmoves,sqSelected)
    #注意pieces棋子需要在最后画，因为前两个高亮应该在底下，否则会把棋子的图案遮挡住
    Drawpieces(screen, gamestate.board)
    #如果是玩家轮，需要展示AI落子结果
    highlight_AI_hints(screen,gamestate,tip_moves)

# 绘制棋盘, Drawgame的子方法
def Drawboard(screen):
    global colors
    colors = [pygame.Color("white"), pygame.Color("#2980b9")]
    for row in range(8):
        for column in range(8):
            color = colors[(row + column) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(column * PieceSIZE, row * PieceSIZE, PieceSIZE, PieceSIZE))

# 绘制棋子, Drawgame的子方法
def Drawpieces(screen, board):
    for row in range(8):
        for column in range(8):
            piece = board[row][column]
            if piece != "--":
                screen.blit(img[piece], pygame.Rect(column * PieceSIZE, row * PieceSIZE, PieceSIZE, PieceSIZE))

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
            screen.blit(img[move.pieceend],endSquare)
        screen.blit(img[move.piecestart], pygame.Rect(c * PieceSIZE, r * PieceSIZE, PieceSIZE, PieceSIZE))
        pygame.display.flip()
        clock.tick(60)

def drawText(screen, text):
    font =pygame.font.SysFont("Helvitca",64,True, False)
    textObject =font.render(text,0, pygame.Color('Black'))
    textLocation = pygame.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH/2 - textObject.get_width()/2, HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject,textLocation)
    textObject =font.render(text,0,pygame.Color('Black'))
    screen.blit(textObject, textLocation.move(2, 2))


# 运行
if __name__ == '__main__':
    main()
