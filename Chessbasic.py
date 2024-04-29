"""
存储当前游戏状态，日志，基本方法，规则
"""
import structure
import numpy as np
import torch

class GameState():
    def __init__(self):
        # 8*8的棋盘，b代表黑，w代表白
        self.board = [
            ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
            ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]]
        self.moveFunctions = {'p': self.getPawnMoves, 'r': self.getRookMoves,
                              'n': self.getKnightMoves, 'q': self.getQueenMoves,
                              'k': self.getKingMoves, 'b': self.getBishopMoves}
        self.IswTomove = True  #判断谁输谁赢
        self.movelog = []
        self.count = 0
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)
        self.checkMate = False #被将的状态
        self.staleMate = False #僵局的状态
        self.pawnhavemoved = False  # 和棋的一种情况，没有兵走动过
        self.enpassantPossible=()
        self.enpassantPossiblelog =[self.enpassantPossible]
        self.currentCastlingRight =CastleRights(True,True,True,True)
        self.castleRightsLog = [CastleRights(self.currentCastlingRight.wks,self.currentCastlingRight.bks,
                                             self.currentCastlingRight.wqs,self.currentCastlingRight.bqs)]
        self.history=structure.HistoryScore()

    # 棋盘状态转换为神经网络可接受的格式
    def get_state(self):
            # 这里需要将棋盘状态转换为一个适合神经网络输入的格式
            # 例如，使用一个8x8x2的numpy数组，其中2表示棋盘上的两种颜色（白和黑）
            # 1表示白棋，-1表示黑棋，0表示空格
            state = np.zeros((8, 8, 2), dtype=np.float32)
            for row in range(8):
                for col in range(8):
                    piece = self.board[row][col]
                    if piece != "--":
                        color, piece_type = piece[0], piece[1]
                        # 白棋为1，黑棋为-1
                        state[row, col, 0 if color == "w" else 1] = 1
                        # 可以根据piece_type给棋子赋予不同的值，或者只区分有无
            return torch.tensor(state)

    def copy(self):
        # ""“创建棋盘状态的深度副本”""

            copied_state = GameState()  # 创建一个新的 GameState 实例
            copied_state.board = [row[:] for row in self.board]  # 深度复制棋盘
            copied_state.IswTomove = self.IswTomove  # 复制谁的回合
            copied_state.movelog = [move.copy() for move in self.movelog]  # 深度复制走法日志
            copied_state.checkMate = self.checkMate  # 复制将死状态
            copied_state.staleMate = self.staleMate  # 复制僵局状态
            copied_state.enpassantPossible = self.enpassantPossible  # 复制en passant状态
            copied_state.enpassantPossiblelog = self.enpassantPossiblelog[:]  # 复制en passant状态日志
            copied_state.currentCastlingRight = self.currentCastlingRight  # 复制当前的王车易位权利
            copied_state.castleRightsLog = self.castleRightsLog[:]  # 复制王车易位权利日志

            # 深度复制历史评分
            copied_state.history = structure.HistoryScore()
            for i in range(len(self.history.history_score)):
                for j in range(len(self.history.history_score[0])):
                    copied_state.history.history_score[i][j] = self.history.history_score[i][j]

            return copied_state

    # # 根据神经网络的输出选择走法
    # def get_move_from_policy(self, policy):
    #         probabilities = policy.detach().numpy().flatten()
    #         action = np.random.choice(len(probabilities), p=probabilities)
    #         legal_moves = self.Getvalidmove()
    #         action_move = legal_moves[action]  # 将动作索引映射到实际走法
    #         return action_move

    # def step(self, move):
    #
    #     # 执行走法
    #     self.Piecemove(move)
    #     # 检查游戏是否结束
    #     if self.is_game_over():
    #         print("Game over!")
    #     else:
    #         # 切换回合
    #         self.IswTomove = not self.IswTomove
    #     return self  # 确保返回当前的 game state

    # 检查游戏是否结束
    def is_game_over(self):
        # 检查是否将死
        if self.inCheck() and len(self.Getvalidmove()) == 0:
            self.checkMate = True
            return True
        # 检查是否僵局
        if not self.inCheck() and len(self.Getvalidmove()) == 0:
            self.staleMate = True
            return True
        return False

    def reset(self):
        # ""“重置游戏状态到一个新的游戏”""
        # 初始化棋盘状态，通常是设置棋子到起始位置
        self.board = self.get_initial_board()
        self.moveFunctions = {'p': self.getPawnMoves, 'r': self.getRookMoves,
                              'n': self.getKnightMoves, 'q': self.getQueenMoves,
                              'k': self.getKingMoves, 'b': self.getBishopMoves}
        # 重置其他游戏状态，例如轮到谁走棋、游戏是否结束等
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)
        self.IswTomove = True
        self.movelog = []
        self.checkMate = False
        self.staleMate = False
        self.enpassantPossible = ()
        self.enpassantPossiblelog = [self.enpassantPossible]
        self.currentCastlingRight = CastleRights(True, True, True, True)
        self.castleRightsLog = [CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                             self.currentCastlingRight.wqs, self.currentCastlingRight.bqs)]
        self.history = structure.HistoryScore()


    def get_initial_board(self):
        # 返回初始棋盘布局的静态方法”"""
        # 这里应该是棋盘的初始布局，与 __init__ 方法中设置的相同
        return [
            ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
            ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]
        ]

#移动棋子
    def Piecemove(self, move):
        self.board[move.startrow][move.startcolumn] = "--"  #  把初始的方块设为空
        if move.piecestart == 'bp' or move.piecestart == 'wp':
            self.pawnhavemoved = True

        if move.pieceend == '--':
            self.count = self.count + 1
        if move.pieceend != '--':
            self.count = 0
            self.pawnhavemoved = False
        self.board[move.endrow][move.endcolumn] = move.piecestart  #  把棋子转移到选定的方块上
        self.IswTomove = not self.IswTomove  #  回合轮换
        self.movelog.append(move)  #  在日志中增加移动记录
        if move.piecestart =="wk":
            self.whiteKingLocation = (move.endrow, move.endcolumn)
        elif move.piecestart =="bk":
            self.blackKingLocation = (move.endrow, move.endcolumn)#更新双王位置
        #pawn promotion小兵晋升
        if move.isPawnPromotion:
            self.board[move.endrow][move.endcolumn] = move.piecestart[0]+'q'

        #enpassant move
        if move.isEnpassantMove:
            self.board[move.startrow][move.endcolumn]='--'
        #update enpassantPossible variable
        if move.piecestart[1]=='p' and abs(move.startrow-move.endrow)==2:
            self.enpassantPossible=((move.startrow+move.endrow)//2,move.startcolumn)
        else:
            self.enpassantPossible=()

        #castle move
        if move.isCastleMove:
            if move.endcolumn-move.startcolumn==2:
                self.board[move.endrow][move.endcolumn-1]=self.board[move.endrow][move.endcolumn+1]#moves the rook
                self.board[move.endrow][move.endcolumn + 1]='--'#erase old rook

            else:
                self.board[move.endrow][move.endcolumn + 1]=self.board[move.endrow][move.endcolumn-2]#moves the rook
                self.board[move.endrow][move.endcolumn -2] = '--'

        self.enpassantPossiblelog.append(self.enpassantPossible)

        #update cateling rights
        self.updateCastleRights(move)
        self.castleRightsLog.append(CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                             self.currentCastlingRight.wqs, self.currentCastlingRight.bqs))
        # 将走法添加到日志中
        self.movelog.append(move)

        return self

    #撤回一步
    def Pieceundo(self):
        if len(self.movelog) != 0:  #只要日志里有记录
            move = self.movelog.pop()   #弹出最后一个记录并赋给move
            self.board[move.startrow][move.startcolumn] =move.piecestart  #把棋子的位置复原
            self.board[move.endrow][move.endcolumn] = move.pieceend
            self.IswTomove = not self.IswTomove  # 回合轮换
            if move.piecestart == "wk":
                self.whiteKingLocation = (move.startrow, move.startcolumn)
            elif move.piecestart == "bk":
                self.blackKingLocation = (move.startrow, move.startcolumn)  # 更新双王位置
            if move.isEnpassantMove:
                self.board[move.endrow][move.endcolumn] = '--'
                self.board[move.startrow][move.endcolumn] = move.pieceend

            self.enpassantPossiblelog.pop()
            self.enpassantPossible = self.enpassantPossiblelog[-1]

            #undo castling rights
            self.castleRightsLog.pop()
            self.currentCastlingRight= self.castleRightsLog[-1]#set the current castle rights to the last one in the list

            #undo castle move
            if move.isCastleMove:
                if move.endcolumn-move.startcolumn==2:
                    self.board[move.endrow][move.endcolumn+1]=self.board[move.endrow][move.endcolumn-1]
                    self.board[move.endrow][move.endcolumn-1]='--'
                else:
                    self.board[move.endrow][move.endcolumn -2] = self.board[move.endrow][move.endcolumn + 1]
                    self.board[move.endrow][move.endcolumn + 1] = '--'

            self.checkMate = False
            self.staleMate = False

    '''
    update the castle rights given the move
    '''
    def  updateCastleRights(self,move):
        if move.piecestart =='wk':
            self.currentCastlingRight.wks=False
            self.currentCastlingRight.wqs = False
        elif move.piecestart=='bk':
            self.currentCastlingRight.bks = False
            self.currentCastlingRight.bqs = False
        elif move.piecestart=='wr':
            if move.startrow==7:
                if move.startcolumn==0:#left rook
                    self.currentCastlingRight.wqs=False
                elif move.startcolumn==7:#right rook
                    self.currentCastlingRight.wks=False
        elif move.piecestart=='br':
            if move.startrow==0:
                if move.startcolumn==0:#left rook
                    self.currentCastlingRight.bqs=False
                elif move.startcolumn==7:#right rook
                    self.currentCastlingRight.bks=False
        #if a rook  is captured
        if move.pieceend == "wr":
            if move.endrow ==7:
                if move.endcolumn==0:
                    self.currentCastlingRight.wqs = False
                elif move.endcolumn == 7:
                    self.currentCastlingRight.wks =False
        elif move.pieceend =="br":
            if move.endrow ==0:
                if move.endcolumn==0:
                    self.currentCastlingRight.bqs =False
                elif move.endcolumn==7:
                    self.currentCastlingRight.bks =False


# 获取合法移动集合

    def Getvalidmove(self):
        tempEnpassantPossible = self.enpassantPossible
        tempCastleRights=CastleRights(self.currentCastlingRight.wks,self.currentCastlingRight.bks,
                                      self.currentCastlingRight.wqs,self.currentCastlingRight.bqs)#copy the current castling rights
        moves = self.Get_all_possible_moves()#生成所有可能的步
        if self.IswTomove:
            self.getCastleMoves(self.whiteKingLocation[0],self.whiteKingLocation[1],moves)
        else:
            self.getCastleMoves(self.blackKingLocation[0],self.blackKingLocation[1],moves)
        for i in range(len(moves)-1, -1, -1):#倒序
            self.Piecemove(moves[i])
            self.IswTomove = not self.IswTomove
            #所有敌方可能的步及其是否会吃王由inChek（）实现
            if self.inCheck():
                moves.remove(moves[i])
            self.IswTomove = not self.IswTomove
            self.Pieceundo()
        if len(moves) == 0: #检测被将的两种状态
            if self.inCheck():
                self.checkMate = True
            else:
                self.staleMate = True
        if self.count >= 100 and self.pawnhavemoved == False:
            self.staleMate = True
        self.enpassantPossible = tempEnpassantPossible
        self.currentCastlingRight=tempCastleRights
        return moves
    #是否被将
    def inCheck(self):
        if self.IswTomove:
            return self.squarelUnderAttack(self.whiteKingLocation[0], self.whiteKingLocation[1])
        else:
            return self.squarelUnderAttack(self.blackKingLocation[0], self.blackKingLocation[1])


    def squarelUnderAttack(self,r,c):
        self.IswTomove = not self.IswTomove#切换至敌方回合
        oppMoves = self.Get_all_possible_moves()
        self.IswTomove = not self.IswTomove#换回来
        for move in oppMoves:
            if move.endrow == r and move.endcolumn == c:#此处会被吃
                return True
        return False


# 获得当前所有可移动的位置，返回一个集合
    def Get_all_possible_moves(self):
        moves = []
        for row in range(8):
            for column in range(8):
                color = self.board[row][column][0]  # 获取当前下棋方
                if (color == "w" and self.IswTomove) or (color == "b" and not self.IswTomove):
                    piece = self.board[row][column][1]  # 获取棋子种类
                    self.moveFunctions[piece](row, column, moves)
                    #根据国际象棋类型调用适当的移动函数
                    # if piece == "p":  # 当棋子是兵时
                    #     self.Pawn(row, column, moves)
                    # # elif piece == "r":  # 当棋子是车时
                    # #     self.Rook(row, column, moves)
                    # # ...
        return moves




# 所有棋子的规则

    #兵
    def getPawnMoves(self, row, column, moves):
        if self.IswTomove:
            if self.board[row-1][column] == "--":  #前一格是否为空
                moves.append(Move((row, column), (row-1, column), self.board))  #前移一格
                if row == 6 and self.board[row - 2][column] == "--":  # 前2格是否为空且在第六行
                    moves.append(Move((row, column), (row - 2, column), self.board))  #前移俩格
            if column-1 >= 0: #不检测-1列
                if self.board[row-1][column-1][0] =="b":
                    moves.append(Move((row, column), (row - 1, column-1), self.board))  #吃左上
                elif (row-1,column-1)==self.enpassantPossible:
                    moves.append(Move((row,column),(row-1,column-1),self.board,isEnpassantMove=True))
            if column+1 <= 7: #不检测9列
                if self.board[row-1][column+1][0] =="b":
                    moves.append(Move((row, column), (row - 1, column+1), self.board))#吃右上
                elif (row - 1, column + 1) == self.enpassantPossible:
                    moves.append(Move((row, column), (row - 1, column + 1), self.board, isEnpassantMove=True))
        else:
            if self.board[row+1][column] == "--":  #前一格是否为空
                moves.append(Move((row, column), (row+1, column), self.board))  #前移一格
                if row == 1 and self.board[row + 2][column] == "--":  # 前2格是否为空且在第六行
                    moves.append(Move((row, column), (row + 2, column), self.board))  #前移俩格
            if column-1 >= 0:  #不检测-1列
                if self.board[row+1][column-1][0] == "w":
                    moves.append(Move((row, column), (row + 1, column-1), self.board))#吃左下
                elif (row + 1, column - 1) == self.enpassantPossible:
                    moves.append(Move((row, column), (row + 1, column - 1), self.board, isEnpassantMove=True))
            if column+1 <= 7:  #不检测9列
                if self.board[row+1][column+1][0] == "w":
                    moves.append(Move((row, column), (row + 1, column+1), self.board))#吃右下
                elif (row + 1, column + 1) == self.enpassantPossible:
                    moves.append(Move((row, column), (row + 1, column + 1), self.board, isEnpassantMove=True))

    #车
    def getRookMoves(self, row, column, moves):
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))  #上左下右
        enemyColor = "b" if self.IswTomove else "w"
        for d in directions:
            for i in range(1,8):
                endRow = row + d[0]*i
                endCol = column + d[1]*i
                if 0 <= endRow < 8 and 0 <= endCol < 8: #在棋盘上
                    endPiece = self.board[endRow][endCol]
                    if endPiece == "--": #有空位
                        moves.append(Move((row, column), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor:
                        moves.append(Move((row, column), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break

    def getKnightMoves(self, row, column, moves):#马
        knightMoves = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
        allyColor = "w" if self.IswTomove else "b"
        for m in knightMoves:
            endRow = row + m[0]
            endCol = column + m[1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:  # 在棋盘上
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor:
                    moves.append(Move((row, column), (endRow, endCol), self.board))
    def getBishopMoves(self, row, column, moves):#主教
        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        enemyColor = "b" if self.IswTomove else "w"
        for d in directions:
            for i in range(1, 8):
                endRow = row + d[0] * i
                endCol = column + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:  # 在棋盘上
                    endPiece = self.board[endRow][endCol]
                    if endPiece == "--":  # 有空位
                        moves.append(Move((row, column), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor:
                        moves.append(Move((row, column), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break
    def getQueenMoves(self, row, column, moves):#王后=主教+车
        self.getRookMoves(row, column, moves)
        self.getBishopMoves(row,column,moves)
    def getKingMoves(self, row, column, moves):#王
        kingMoves = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        allyColor = "w" if self.IswTomove else "b"
        for i in range(8):
            endRow = row + kingMoves[i][0]
            endCol = column + kingMoves[i][1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:  # 在棋盘上
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor:
                    moves.append(Move((row, column), (endRow, endCol), self.board))


    def getCastleMoves(self,row,column,moves):
        if self.squarelUnderAttack(row,column):
            return
        if (self.IswTomove and self.currentCastlingRight.wks) or (not self.IswTomove and self.currentCastlingRight.bks):
            self.getKingsideCastleMoves(row,column,moves)
        if (self.IswTomove and self.currentCastlingRight.wqs)or (not self.IswTomove and self.currentCastlingRight.bqs):
            self.getQueensideCastleMoves(row, column, moves)

    def getKingsideCastleMoves(self, row, column, moves):
        if self.board[row][column+1]=='--' and self.board[row][column+2]=='--':
            if not self.squarelUnderAttack(row,column+1) and not self.squarelUnderAttack(row,column+2):
                moves.append(Move((row,column),(row,column+2),self.board,isCastleMove=True))

    def getQueensideCastleMoves(self, row, column, moves):
        if self.board[row][column - 1] == '--' and self.board[row][column - 2] == '--'and self.board[row][column-3] == '--':
            if not self.squarelUnderAttack(row, column - 1) and not self.squarelUnderAttack(row, column - 2):
                moves.append(Move((row, column), (row, column - 2), self.board, isCastleMove=True))




class CastleRights():
    def __init__(self,wks,bks,wqs,bqs):
        self.wks=wks
        self.bks=bks
        self.wqs = wqs
        self.bqs = bqs



# 关于移动和坐标
class Move():
    #  转化国际象棋的标准坐标
    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,
                   "5": 3, "6": 2, "7": 1, "8": 0}
    rowsToRanks = {v: k for k, v in ranksToRows.items()}
    filesTocols = {"a": 0, "b": 1, "c": 2, "d": 3,
                   "e": 4, "f": 5, "g": 6, "h": 7}
    colsToFiles = {v: k for k, v in filesTocols.items()}

    def __init__(self, start, end, board,isEnpassantMove =False,isCastleMove =False):
        #  记录开始位置和最终位置
        self.startrow = start[0]
        self.startcolumn = start[1]
        self.endrow = end[0]
        self.endcolumn = end[1]
        self.piecestart = board[self.startrow][self.startcolumn]    #  记录棋子起始位置,=moved
        self.pieceend = board[self.endrow][self.endcolumn]         #  记录棋子最终位置,=captured
        #PawnPromotion
        self.isPawnPromotion= (self.piecestart =='wp'and self.endrow ==0) or (self.piecestart=='bp'and self.endrow ==7)
        #Enpassant
        self.isEnpassantMove = isEnpassantMove
        if self.isEnpassantMove:
            self.pieceend='wp'if self.piecestart=='bp'else 'bp'
        #castle move
        self.isCastleMove=isCastleMove

        self.moveID = self.startrow*1000+self.startcolumn*100+self.endrow*10+self.endcolumn  # 给每次移动建立一个唯一ID
        # print(self.moveID)
        self.h_score=-9999
        self.board = board

    def copy(self):
        # 创建并返回当前走法的深拷贝
        return Move(
            (self.startrow, self.startcolumn),
            (self.endrow, self.endcolumn),
            self.board,
            isEnpassantMove=self.isEnpassantMove,
            isCastleMove=self.isCastleMove
        )

    # 重写equal方法
    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    #  获得标准坐标棋盘的移动记录
    def getChess(self):
        return self.getRank(self.startrow, self.startcolumn) + self.getRank(self.endrow, self.endcolumn)

    def getRank(self, row, column):
        return self.colsToFiles[column] + self.rowsToRanks[row]

