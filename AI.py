"""
所有机器算法的实现，目前已经完成：
1.随机移动(不是AI)
2.贪婪算法
"""
import random


pieceScore = {"k":0,"q":10,"r":8,"b":5,"n":5,"p":1}
checkmate = 999
stalemate = 0

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


#随机
def randommove(validmoves):
    return validmoves[random.randint(0,len(validmoves)-1)]


#非递归贪婪
def greedymove(gamestate,validmoves):
    turn = 1 if gamestate.IswTomove else -1 # 由turn来记录此次移动是该加分还是减分
    opponentminmaxscore = checkmate #初始值为999
    bestmove = None
    random.shuffle(validmoves) #打乱可移动的位置集合
    for playermove in validmoves:  #遍历可移动的位置集合
        gamestate.Piecemove(playermove)
        opponentsMoves = gamestate.Getvalidmove()# 获取对手的移动
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


#递归版贪婪
