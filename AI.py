"""
所有机器算法的实现，目前已经完成：
1.随机移动(不是AI)
2.贪婪算法
"""
import random
from structure import Node, GameTree, Queue, PriorityQueue
import time
pieceScore = {"k": 20000, "q": 900, "r": 500, "b": 330, "n": 320, "p": 100}  # k:王 n:马 b:相 p:兵
checkmate = 999
stalemate = 0

POSITION_WHITE_PAWN = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [0, 0, 0, 0, 0, 0, 0, 0]]

POSITION_WHITE_KNIGHT = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]]

POSITION_WHITE_BISHOP = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]]

POSITION_WHITE_ROOK = [
    [0, 0, 0, 5, 5, 0, 0, 0],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]]

POSITION_WHITE_QUEEN = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]]

POSITION_WHITE_KING = [
    [20, 30, 10, 0, 0, 10, 30, 20],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30]]

POSITION_BLACK_PAWN = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]]

POSITION_BLACK_KNIGHT = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]]

POSITION_BLACK_BISHOP = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]]

POSITION_BLACK_ROOK = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0]]

POSITION_BLACK_QUEEN = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]]

POSITION_BLACK_KING = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20]]
piecePositionScores = {"wp": POSITION_WHITE_PAWN,
                       "wn": POSITION_WHITE_KNIGHT,
                       "wb": POSITION_WHITE_BISHOP,
                       "wr": POSITION_WHITE_ROOK,
                       "wq": POSITION_WHITE_QUEEN,
                       "wk": POSITION_WHITE_KING,
                       "bp": POSITION_BLACK_PAWN,
                       "bn": POSITION_BLACK_KNIGHT,
                       "bb": POSITION_BLACK_BISHOP,
                       "br": POSITION_BLACK_ROOK,
                       "bq": POSITION_BLACK_QUEEN,
                       "bk": POSITION_BLACK_KING
                       }

# 静态评估函数 @Yan 返回白棋棋力-黑棋棋力
def static_evaluate(board):
    white_score = 0
    black_score = 0
    f = 0
    i = 0
    j = 0
    for row in board:  # 二维数组board中的行
        j=0
        for column in row:  # 二维数组board中的列
            if column[0] == "w":
                # 子力评估
                white_score += pieceScore[column[1]]
                # 位置评估
                white_score += piecePositionScores[column[:2]][i][j]
                # 灵活性评估
                # 其他
            elif column[0] == "b":
                # 子力评估
                black_score += pieceScore[column[1]]
                # 位置评估
                black_score += piecePositionScores[column[:2]][i][j]
            j+=1
        i+=1
                # 灵活性评估
                # 其他
    f = white_score - black_score
    return f


# 随机
def randommove(validmoves):
    return validmoves[random.randint(0, len(validmoves) - 1)]

#黑子
def test_get_best_move(gamestate,depth):

    tree=make_game_tree(gamestate,depth,-1)
    child_list = tree.root.children
    child_list.sort(key=lambda item: item.val)  # 按照升序来排序
    return child_list[0].move
# 生成博弈树 depth为探索层次，暂且为4层, 特地将gamestate.board记录下,方便调试调取, 同时也带来了占用内存过大的问题 40*40*40*40 初步可能为400MB, 可以改为记录步
def make_game_tree(gamestate, depth,player_code:int) -> GameTree:
    gamestate.node_count=0
    start_time = time.time()
    root = Node(None, -9999,9999,-9999, None, 0)
    tree = GameTree(root)
    dfs(gamestate, root, depth,player_code)
    end_time=time.time()
    print("用时",end_time-start_time,"遍历结点数",gamestate.node_count)
    return tree

# dfs是make_game_tree 内置函数 评估函数默认为 白棋-黑棋 player_code,白棋为1，黑棋为-1
def dfs(gamestate, current_node: Node, depth: int,player_code:int)->int:
    gamestate.node_count+=1
    if current_node.depth == depth:
        # Terminal
        # 计算底层叶子结点局面价值
        current_node.val = player_code*static_evaluate(gamestate.board)
        if current_node.depth%2: #黑棋为负
            current_node.val=-current_node.val #使用negamax算法，需要对敌方叶子结点取负
    else:
        moves: list = gamestate.Getvalidmove(True)
        # print(current_node, current_node.depth, len(moves), "个落子可能")
        if len(moves)==0:
            #计算非底层叶子结点局面价值
            current_node.val = player_code*static_evaluate(gamestate.board)
            if current_node.depth % 2:  # 敌方
                print("黑方被将死" if player_code == 1 else "白方被将死")
            else:  # 我方
                # 使用negamax算法，需要对我方叶子节点执行取负
                current_node.val = -current_node.val
                print("白方被将死" if player_code==1 else "黑方被将死")
        else:
            #根据历史库中的价值表对Move的进行价值排序，价值越高排在前面，越优先遍历，让剪枝发生的更快
            for i in range(len(moves)):
                moves[i].score=gamestate.history.get(moves[i])
            moves.sort(key=lambda item: item.score,reverse=True) #降序，分高的在前
            #暂定第一步为好棋，然后根据搜索再确认好棋是哪部
            best_move=moves[0]
            # 结点仍有子结点
            for each in moves:
                # 模拟走棋
                gamestate.Piecemove(each)
                # 建立子结点
                new_node = Node(current_node,-current_node.beta,-max(current_node.alpha,current_node.val),-9999, each, current_node.depth + 1)
                current_node.add_child(new_node)
                # negamax
                r_value=-dfs(gamestate, new_node, depth,player_code) #子节点返回的价值

                if(r_value>current_node.val):
                    current_node.val=r_value
                    best_move=each #更新最佳步
                if current_node.val>=current_node.beta: #negamax AB pruning，取大于号即可， 把那些等于的也算进来供决策者决策
                    break
            #添加该走法到历史库，而该走法的价值是由该走法搜索的深度决定的
            gamestate.history.add(best_move,depth-current_node.depth)

    if not current_node.depth==0: #对于非根节点需要执行棋盘回溯
        gamestate.Pieceundo()

    return current_node.val # 返回本结点的评估值

