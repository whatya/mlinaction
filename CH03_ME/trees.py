#!/usr/bin/python
#coding:utf-8
# 以上两行代码解决Python不支持中文的问题

from math import log
import operator

# 计算信息熵：熵定义为信息期望值, 被分类后的各数据之间相似程度越高（越有序）熵越低, 数据之间的
# 相似程度越低（越杂乱、无序）熵越高
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)                                                   # 获取训练数据总数
    labelCounts = {}                                                            # 记录每个标签（分类）出现的总次数如：{‘yes’:2}
    for featVec in dataSet:                                                     # 以下逻辑统计每个标签出现的总次数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0                                                            # 熵值初始化
    for key in labelCounts:                                                     # 以下逻辑计算熵值
        prob = float(labelCounts[key])/numEntries                               # 计算每个标签出现次数的占比（即该分类的概率）
        shannonEnt -= prob * log(prob,2)                                        # 根据公式计算熵值
    return shannonEnt


# 生成测试数据集
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 划分数据: 表示过滤出每一行中第axis个元素值为value的元素然后只返回每行axis后面的值，如下：
# axis=0,value=0,data=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
# data ->oper1 ->[[0,1,'no'],[0,1,'no']] ->oper2 ->[[1,'no'],[1,'no']]
# oper1操作过滤出所有第一个元素为0的行，oper2操作只保留第一个元素意外的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                                     # data=[1,2,3,4] data[:0]=[] data[:1]=[1] data[:3]=[1,2,3]
            reducedFeatVec.extend(featVec[axis+1:])                             # data[0:]=[1,2,3,4] temp[2:]=[3,4]
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                                           # 获取每行特征总数（最后一个为标签结果）
    baseEntropy = calcShannonEnt(dataSet)                                       # 由于训练数据为正确答案，因此训练数据的熵值肯定是最大的
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:                                                # 以下逻辑计算按每列特征分类的熵值
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy                                     # 找出距离正确分类熵值最小的分类
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature



# 投票表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),\
    key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,\
        bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel

#树持久化
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
