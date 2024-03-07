"""
所有机器算法的实现，目前已经完成：
1.随机移动(不是AI)
2.贪婪算法
"""
import random


#随机
def randommove(validmoves):
    return validmoves[random.randint(0,len(validmoves)-1)]
