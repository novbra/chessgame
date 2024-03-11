"""
所有机器算法的实现，目前已经完成：
1.随机移动(不是AI)
2.贪婪算法
"""
import random
from structure import Node, GameTree, Queue, PriorityQueue

pieceScore = {"k": 1000, "q": 90, "r": 50, "b": 20, "n": 20, "p": 10}  # k:王 n:马 b:相 p:兵
checkmate = 999
stalemate = 0


# 评分函数
def scoreMaterial(board):
    score = 0
    for row in board:
        for square in row:
            if square[0] == "w":
                score += pieceScore[square[1]]
            elif square[0] == "b":
                score -= pieceScore[square[1]]
    return score


# 静态评估函数 @Yan 返回白棋棋力-黑棋棋力
def static_evaluate(board):
    white_score = 0
    black_score = 0
    f = 0
    for row in board:  # 二维数组board中的行
        for column in row:  # 二维数组board中的列
            if column[0] == "w":
                # 子力评估
                white_score += pieceScore[column[1]]
                # 位置评估
                # 灵活性评估
                # 其他
            elif column[0] == "b":
                # 子力评估
                black_score += pieceScore[column[1]]
                # 位置评估
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
    child_list.sort(key=lambda item: -item.val)  # 按照降序来排序
    return child_list[0].move
# 生成博弈树 depth为探索层次，暂且为4层, 特地将gamestate.board记录下,方便调试调取, 同时也带来了占用内存过大的问题 40*40*40*40 初步可能为400MB, 可以改为记录步
def make_game_tree(gamestate, depth,player_code:int) -> GameTree:
    root = Node(None, 0, None, 0)
    tree = GameTree(root)
    dfs(gamestate, root, depth,player_code)

    return tree

# dfs是make_game_tree 内置函数 评估函数默认为 白棋-黑棋 player_code,白棋为1，黑棋为-1
def dfs(gamestate, current_node: Node, depth: int,player_code:int)->int:
    if current_node.depth == depth:
        # print("该结点已到达设定的深度")
        # 计算底层叶子结点局面价值
        current_node.val = player_code*static_evaluate(gamestate.board)

        if current_node.depth%2:#敌方
            print("敌方不需要修正")
        elif current_node.depth%2==0: #我方
            # 使用negamax算法，需要对我方叶子节点执行取负
            current_node.val =-current_node.val

    else:
        moves: list = gamestate.Getvalidmove(True)
        # print(current_node, current_node.depth, len(moves), "个落子可能")

        if len(moves)==0:
            print("该结点已无棋可走：某方被将死")
            #计算非底层叶子结点局面价值
            current_node.val = player_code*static_evaluate(gamestate.board)
            if current_node.depth % 2:  # 敌方
                print("敌方不需要修正")
            else:  # 我方
                # 使用negamax算法，需要对我方叶子节点执行取负
                current_node.val = -current_node.val
        else:
            #结点仍有子结点
            for each in moves:
                gamestate.Piecemove(each)
                new_node = Node(current_node, -9999, each, current_node.depth + 1)


                current_node.add_child(new_node)
                #negamax
                current_node.val=max(dfs(gamestate, new_node, depth,player_code),current_node.val)

            #对于非叶子结点, 直接取负即可
            current_node.val=-current_node.val

    if not current_node.depth==0: #对于非根节点需要执行棋盘回溯
        gamestate.Pieceundo()

    return current_node.val # 返回本结点的评估值

