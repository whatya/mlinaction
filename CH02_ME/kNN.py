#!/usr/bin/python
#coding:utf-8
# 以上两行代码解决Python不支持中文的问题
from numpy import *
import operator


# 创建测试数据 4*2 矩阵（4行*2列）
# @return： group数据点，labels对应每个点所属的分类，相当于测试数据（x，y）
def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


# k-近邻算法：计算 输入点 与 输入数据集 中每个点的距离，然后返回距离最近的前k个数据
# 距离计算公式：A(x,y,z) B(x,y,z) => distance = 根号下((A.x-B.x)的平方 + (A.y-B.y)的平方 + (A.z-B.z)的平方 )
# @parameters：inX输入数据点x，dataSet训练数据集m*n矩阵，labels对应dataSet的y值m*1矩阵
# @return
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]												# 获取数据总行数
	diffMat = tile(inX, (dataSetSize,1)) - dataSet								# 用数据集中每一行数据与输入数据的每个特征值相减，形成新矩阵差
	sqDiffMat = diffMat ** 2													# 将上述矩阵每个元素平方
	sqDistances = sqDiffMat.sum(axis=1)											# 将上述矩阵每行的平方差相加
	distances = sqDistances ** 0.5												# 最后对每个平方差开放就得到输入点与数据集每个点的距离
	sortedDistIndicies = distances.argsort()									# 从小到大排序
	classCount = {}
	for i in range(k):															# 以下逻辑为返回距离最小的点
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

# 测试近邻算法
def classify0Test(inX):
	k = 3
	group, labels = createDataSet()
	result = classify0(inX, group, labels, 3)
	print(result)


# 将文本数据处理为正确的数据
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         								# 获取总行数
    returnMat = zeros((numberOfLines,3))										# 生成占位矩阵（值都为0）m（文件行数）* n （固定3列），因为有3个特征量
    classLabelVector = []                       								# 标签矩阵 m * 1
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')											# 生成每一行的一维数组 如 [0,0,0,1]，最后一个元素应该为字符串，为了好处理，用数字代表标签
        returnMat[index,:] = listFromLine[0:3]									# 生成特征量数组（x值）
        classLabelVector.append(int(listFromLine[-1]))							# 生成标签数组（y值）
        index += 1
    return returnMat,classLabelVector
