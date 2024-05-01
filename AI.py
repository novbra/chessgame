"""
所有机器算法的实现，目前已经完成：
1.随机移动
2.贪婪算法（非递归、极小化极大、negamax、ab剪枝）
"""
import random
import time

pieceScore = {"k":0,"q":10,"r":8,"b":5,"n":5,"p":1}
# 评估函数：兵种位置得分
knightScore = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, 2, 2, 2, 2, 2, 2, 1],
              [1, 2, 3, 3, 3, 3, 2, 1],
              [1, 2, 3, 4, 4, 3, 2, 1],
              [1, 2, 3, 4, 4, 3, 2, 1],
              [1, 2, 3, 3, 3, 3, 2, 1],
              [1, 2, 2, 2, 2, 2, 2, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

bishopScore = [[4, 3, 2, 1, 1, 2, 3, 4],
              [3, 4, 3, 2, 2, 3, 4, 3],
              [2, 3, 4, 3, 3, 4, 3, 2],
              [1, 2, 3, 4, 4, 3, 2, 1],
              [1, 2, 3, 4, 4, 3, 2, 1],
              [2, 3, 4, 3, 3, 4, 3, 2],
              [3, 4, 3, 2, 2, 3, 4, 3],
              [4, 3, 2, 1, 1, 2, 3, 4]]

queenScore = [[1, 1, 1, 3, 3, 1, 1, 1],
              [1, 2, 3, 3, 3, 1, 1, 1],
              [1, 4, 3, 3, 3, 4, 2, 1],
              [1, 2, 3, 4, 4, 3, 2, 1],
              [1, 2, 3, 3, 3, 2, 2, 1],
              [1, 4, 3, 3, 3, 4, 2, 1],
              [1, 2, 3, 3, 3, 1, 1, 1],
              [1, 1, 1, 3, 3, 1, 1, 1]]

rookScore = [[4, 3, 4, 4, 4, 4, 3, 4],
              [4, 4, 4, 4, 4, 4, 4, 4],
              [1, 1, 2, 3, 3, 2, 1, 1],
              [1, 2, 3, 4, 4, 3, 2, 1],
              [1, 2, 3, 4, 4, 3, 2, 1],
              [1, 1, 2, 3, 3, 2, 1, 1],
              [4, 4, 4, 4, 4, 4, 4, 4],
              [4, 3, 4, 4, 4, 4, 3, 4]]

whitePawnScore = [[8, 8, 8, 8, 8, 8, 8, 8],
                  [8, 8, 8, 8, 8, 8, 8, 8],
                  [5, 6, 6, 7, 7, 6, 6, 5],
                  [2, 3, 3, 5, 5, 3, 3, 2],
                  [1, 2, 3, 4, 4, 3, 2, 1],
                  [1, 1, 2, 3, 3, 2, 1, 1],
                  [1, 1, 1, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0]]

blackPawnScore = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 1],
                  [1, 1, 2, 3, 3, 2, 1, 1],
                  [1, 2, 3, 4, 4, 3, 2, 1],
                  [2, 3, 3, 5, 5, 3, 3, 2],
                  [5, 6, 6, 7, 7, 6, 6, 5],
                  [8, 8, 8, 8, 8, 8, 8, 8],
                  [8, 8, 8, 8, 8, 8, 8, 8]]

piecePositionScores = {"n": knightScore, 'q': queenScore, "b": bishopScore, "r": rookScore,
                      "bp": blackPawnScore, "wp": whitePawnScore}

checkmate = 999
stalemate = 0
DEPTH    = 1 #控制递归版贪婪的递归深度
max_depth = 1  # Set the desired maximum depth
limittime =0.01
# 评分函数
def scoreMaterial(board):
    score = 0
    for row in board:
        for square in row :
            if square[0] == "w":
                score += pieceScore[square[1]]
            elif square[0] =="b":
                score -= pieceScore[square[1]]
    return score
# 如果得分为正，则表面对白棋有优势，为负对黑优
def scoreBoard(gamestate):
    if gamestate.checkMate:
        if gamestate.IswTomove:# 将局状态且轮到白棋走，则得分为-999
            return -checkmate
        else:
            return checkmate
    elif gamestate.staleMate:
        return stalemate

    score = 0
    for row in range(len(gamestate.board)):
        for column in range(len(gamestate.board[row])):
            square =gamestate.board[row][column]
            if square != "--":
                piecePositionScore =0
                if square[1] != "k":
                    if square[1]=="p":
                        piecePositionScore = piecePositionScores[square][row][column]
                    else:
                        piecePositionScore = piecePositionScores[square[1]][row][column]


                if square[0] == "w":
                    score += pieceScore[square[1]] +piecePositionScore * .1
                elif square[0] == "b":
                    score -= pieceScore[square[1]] +piecePositionScore * .1
    return score



#随机
def findrandommove(validmoves):
    return validmoves[random.randint(0,len(validmoves)-1)]


#非递归贪婪
def findgreedymove(gamestate,validmoves):
    turn = 1 if gamestate.IswTomove else -1 # 由turn来记录此次移动是该加分还是减分
    opponentminmaxscore = checkmate #初始值为999
    bestmove = None
    random.shuffle(validmoves) #打乱可移动的位置集合
    for playermove in validmoves:  #遍历可移动的位置集合
        gamestate.Piecemove(playermove)
        opponentsMoves,opptrainvalidmoves = gamestate.Getvalidmove()# 获取对手的移动
        if gamestate.checkMate:
            opponentmaxscore = -checkmate
        elif gamestate.staleMate:
            opponentmaxscore =stalemate
        else:
            opponentmaxscore = - checkmate #对手的最优得分
            for opponentsMove in opponentsMoves:  #计算对手的可能的移动
                gamestate.Piecemove(opponentsMove)
                gamestate.Getvalidmove()
                if gamestate.checkMate: # 如果这一步可以将王，则得满分
                    score = checkmate
                elif gamestate.staleMate:# 如果是僵局，则得0分
                    score = stalemate
                else:
                    score = -turn * scoreMaterial(gamestate.board) # 记录该次移动的得分
                if score>opponentmaxscore:
                    opponentmaxscore= score  # 记录当前最优解
                gamestate.Pieceundo()

        # 如果对方
        if opponentmaxscore<opponentminmaxscore:
            opponentminmaxscore = opponentmaxscore
            bestmove = playermove
        gamestate.Pieceundo()# 每次循环之后都撤回这一步
    return bestmove

# 调用minmax或者negamax或者αβ优化negamax
def findminmaxmove(gamestate,validmoves, returnQuene,limittime):
    global nextmove,counter
    nextmove = None
    random.shuffle(validmoves)
    start_time = time.time()
    counter = 0 #记录调用了多少次的算法函数
    # minmaxmove(gamestate,validmoves,DEPTH,gamestate.IswTomove)
    # negamaxmove(gamestate,validmoves,DEPTH, 1 if gamestate.IswTomove else -1)
    # negamaxalphabetamove(gamestate, validmoves,-checkmate, checkmate,  DEPTH, 1 if gamestate.IswTomove else -1)
    # _,nextmove =negamax_alpha_beta(gamestate, validmoves,DEPTH,-checkmate, checkmate,1 if gamestate.IswTomove else -1)
    nextmove =iterativedeepeningmove(gamestate, validmoves,-checkmate, checkmate, 1 if gamestate.IswTomove else -1, limittime)
    end_time = time.time()
    print("用时", end_time - start_time, "算法调用次数", counter)

    returnQuene.put(nextmove)

#递归版贪婪
def minmaxmove(gamestate, validmoves, depth, wTomove):
    random.shuffle(validmoves) #打乱可移动的位置集合
    global  nextmove
    if depth==0:
        return scoreBoard(gamestate)

    if wTomove:
        maxscore = -checkmate # 从最坏的情况开始
        for move in validmoves:
            gamestate.Piecemove(move)
            nextmoves,nexttrainvalidmoves = gamestate.Getvalidmove()
            score  = minmaxmove(gamestate,nextmoves, depth-1,False)
            if score > maxscore:
                maxscore = score
                if depth == DEPTH:# 如果已经查找完所有的分支之后
                    nextmove = move
            gamestate.Pieceundo()
        return maxscore
    else:
        minscore = checkmate
        for move in validmoves:
            gamestate.Piecemove(move)
            nextmoves,nexttrainvalidmoves = gamestate.Getvalidmove()
            score = minmaxmove(gamestate, nextmoves, depth - 1, True)
            if score < minscore:
                minscore = score
                if depth ==DEPTH:
                    nextmove = move
            gamestate.Pieceundo()
        return minscore


# negamax
def negamaxmove(gamestate, validmoves, depth, turn): #此方法为depth降序
    random.shuffle(validmoves) #打乱可移动的位置集合
    global nextmove,counter
    counter+=1
    if depth ==0:
        return turn*scoreBoard(gamestate)

    maxscore = -checkmate
    for move in validmoves:
        gamestate.Piecemove(move)
        nextmoves,nexttrainvalidmoves = gamestate.Getvalidmove()
        score = -negamaxmove(gamestate, nextmoves,depth-1,-turn)
        if score> maxscore:
            maxscore = score
            if depth ==DEPTH:
                nextmove = move
        gamestate.Pieceundo()
    return maxscore

# negamax+alphabeta剪枝
def negamaxalphabetamove(gamestate, validmoves,alpha,beta, depth, turn):
    random.shuffle(validmoves)  # 打乱可移动的位置集合
    global nextmove,counter
    counter +=1
    if depth == 0:
        return turn * scoreBoard(gamestate)

    maxscore = -checkmate

    #对validmoves进行历史排序

    best_move=None
    for move in validmoves:
        gamestate.Piecemove(move)
        nextmoves,nexttrainvalidmoves = gamestate.Getvalidmove()

        if not len(nextmoves)==0:
            # 根据历史库中的价值表对Move的进行价值排序，价值越高排在前面，越优先遍历，让剪枝发生的更快
            for i in range(len(nextmoves)):
                nextmoves[i].score = gamestate.history.get(nextmoves[i])
            #对nextmoves排序
            nextmoves.sort(key=lambda item: item.h_score, reverse=False)
            # 暂定第一步为好棋，然后根据搜索再确认好棋是哪部
            best_move = nextmoves[0]
            score = -negamaxalphabetamove(gamestate, nextmoves,-beta,-alpha, depth - 1, -turn)
            if score > maxscore:
                maxscore = score
                best_move= move #@Yan 新增
                if depth == DEPTH:
                    nextmove = move
                    print(move,score)
            gamestate.Pieceundo()
            if maxscore>alpha:# 剪枝开始，alpha表示当前搜索路径上已知的最佳分数，beta表示对手的最佳分数。如果某个节点的分数超出了这个范围，就可以停止搜索该节点的子树。
                alpha =maxscore
            if alpha>=beta:# 此处是因为alpha是从-999开始的，而beta是从999开始的
                break
    #循环结束后，将本结点最佳下步记录到历史库中
    if not best_move==None:
        gamestate.history.add(best_move,depth)
    return maxscore


# 加了时间限制
def negamax_alpha_beta(gamestate,validmoves, depth, alpha, beta, turn,starttime,limittime):
    global counter
    counter +=1
    if depth == 0 :
        return turn * scoreBoard(gamestate), None

    max_score = -checkmate
    best_move = None

    for move in validmoves:
        if time.time() - starttime > limittime:
            break
        gamestate.Piecemove(move)
        nextmoves,nexttrainvalidmoves = gamestate.Getvalidmove()
        score, _ = negamax_alpha_beta(gamestate,nextmoves, depth - 1, -beta, -alpha, -turn,starttime,limittime)

        score = -score
        gamestate.Pieceundo()

        if score > max_score:
            max_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return max_score, best_move

# iterativedeepening
def iterativedeepeningmove(gamestate,validmoves, alpha, beta, turn,time_limit):
    start_time = time.time()
    depth = 1
    bestmoves=[]
    while True:
        _,bestmove = negamax_alpha_beta(gamestate,validmoves, depth, alpha, beta, turn,start_time,time_limit)
        if bestmove!=None:
            bestmoves.append(bestmove)
        if time.time() - start_time > time_limit:
            break
        depth+=1
        print(depth)
    return bestmoves[-1]
