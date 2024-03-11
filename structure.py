import heapq


class Node:

    def __init__(self, parent, val: int, move, depth: int) -> None:
        self.parent = parent
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