"""
存储当前游戏状态，日志，基本方法，规则
"""
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
        self.IswTomove = True
        self.movelog = []
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)
        self.checkMate = False #被将的状态
        self.staleMate = False #僵局的状态


#移动棋子
    def Piecemove(self, move):
        self.board[move.startrow][move.startcolumn] = "--"  #  把初始的方块设为空
        self.board[move.endrow][move.endcolumn] = move.piecestart  #  把棋子转移到选定的方块上
        self.IswTomove = not self.IswTomove  #  回合轮换
        self.movelog.append(move)  #  在日志中增加移动记录
        if move.moveID =="wk":
            self.whiteKingLocation = (move.endrow, move.endcolumn)
        elif move.moveID =="bk":
            self.blackKingLocation = (move.endrow, move.endcolumn)#更新双王位置


#撤回一步
    def Pieceundo(self):
        if len(self.movelog) != 0:  #只要日志里有记录
            move = self.movelog.pop()   #弹出最后一个记录并赋给move
            self.board[move.startrow][move.startcolumn] =move.piecestart  #把棋子的位置复原
            self.board[move.endrow][move.endcolumn] = move.pieceend
            self.IswTomove = not self.IswTomove  # 回合轮换
            if move.moveID == "wk":
                self.whiteKingLocation = (move.startrow, move.startcolumn)
            elif move.moveID == "bk":
                self.blackKingLocation = (move.startrow, move.startcolumn)  # 更新双王位置

# 获取合法移动集合
    def Getvalidmove(self):
        moves = self.Get_all_possible_moves()#生成所有可能的步
        for i in range(len(moves)-1, -1, -1):#倒序
            self.Piecemove(moves[i])
            self.IswTomove = not self.IswTomove
            #所有敌方可能的步及其是否会吃王由inChek（）实现
            if self.inCheck():
                moves.remove(self.moves[i])
            self.IswTomove = not self.IswTomove
            self.Pieceundo()
        if len(moves) == 0: #检测被将的两种状态
            if self.inCheck():
                self.checkMate = True
            else:
                self.staleMate = True
        else:
            self.staleMate = False
            self.checkMate = False

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
            if column+1 <= 7: #不检测9列
                if self.board[row-1][column+1][0] =="b":
                    moves.append(Move((row, column), (row - 1, column+1), self.board))#吃右上

        else:
            if self.board[row+1][column] == "--":  #前一格是否为空
                moves.append(Move((row, column), (row+1, column), self.board))  #前移一格
                if row == 1 and self.board[row + 2][column] == "--":  # 前2格是否为空且在第六行
                    moves.append(Move((row, column), (row + 2, column), self.board))  #前移俩格
            if column-1 >= 0:  #不检测-1列
                if self.board[row+1][column-1][0] == "w":
                    moves.append(Move((row, column), (row + 1, column-1), self.board))#吃左下
            if column+1 <= 7:  #不检测9列
                if self.board[row+1][column+1][0] == "w":
                    moves.append(Move((row, column), (row + 1, column+1), self.board))#吃右下


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












# 关于移动和坐标
class Move():
    #  转化国际象棋的标准坐标
    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,
                   "5": 3, "6": 2, "7": 1, "8": 0}
    rowsToRanks = {v: k for k, v in ranksToRows.items()}
    filesTocols = {"a": 0, "b": 1, "c": 2, "d": 3,
                   "e": 4, "f": 5, "g": 6, "h": 7}
    colsToFiles = {v: k for k, v in filesTocols.items()}

    def __init__(self, start, end, board):
        #  记录开始位置和最终位置
        self.startrow = start[0]
        self.startcolumn = start[1]
        self.endrow = end[0]
        self.endcolumn = end[1]
        self.piecestart = board[self.startrow][self.startcolumn]    #  记录棋子起始位置
        self.pieceend = board[self.endrow][self.endcolumn]         #  记录棋子最终位置
        self.moveID = self.startrow*1000+self.startcolumn*100+self.endrow*10+self.endcolumn  # 给每次移动建立一个唯一ID
        # print(self.moveID)

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

