import heapq


class Node:

    def __init__(self, parent, alpha:int,beta:int,val: int, move, depth: int) -> None:
        self.parent = parent
        self.alpha=alpha
        self.beta=beta
        self.val = val
        self.move = move
        self.depth = depth
        self.children:list=[]

    def add_child(self, child) -> None:
        self.children.append(child)
        child.parent = self


class GameTree:
    root: Node = None

    def __init__(self, root: Node) -> None:
        self.root = root


class Queue(object):
    def __init__(self, size):
        '''队列属性初始化'''
        self.__li = []
        self.__size = size
        self.__cur = 0

    def isFull(self):
        '''判断队列是否满'''
        return self.__cur == self.__size

    def isEmpty(self):
        '''判断队列是否空'''
        return self.__cur == 0

    def enQueue(self, data):
        '''入队'''
        if not self.isFull():
            self.__li.append(data)
            self.__cur += 1
        else:
            print('the Queue is full')

    def deQueue(self):
        '''出队'''
        if not self.isEmpty():
            return self.__li.pop(0)
        else:
            print('the Queue is empty')

    def Queue_head(self):
        '''返回对头的元素'''
        return self.__li(0)

    def showQueue(self):
        '''输出队列全部数据'''
        print(self.__li)

    def size(self):
        '''返回队列的大小'''
        return self.__cur


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.index = 0

    # 入队元素
    def push(self, item, priority):
        heapq.heappush(self.queue, (-priority, self.index, item))
        self.index += 1

    def pop(self):
        return heapq.heappop(self.queue)[-1]

class HistoryScore:
    def __init__(self):
        self.reset()
    def reset(self):
        self.history_score=[[0 for x in range(64)] for y in range(64)]# [0 for x in range(64)]生成了一个含有64个0的列表
    def add(self,move,depth):
        i=move.startrow*8+move.startcolumn
        j=move.endrow*8+move.endcolumn
        self.history_score[i][j]+= 2 << depth# 根据该结点搜索深度计算其价值
        #这里加的权值是J.Schaeffer建议的2^depth，离叶子节点越近则越小，可以理解为越上面的节点是经过更多次计算挑选的，所以价值越大（不一定准确）
    def get(self,move):
        i = move.startrow * 8 + move.startcolumn
        j = move.endrow * 8 + move.endcolumn
        return self.history_score[i][j]