#!/usr/bin/python
#coding:utf-8
# 以上两行代码解决Python不支持中文的问题
from numpy import *
from os import listdir
import operator

# k-近邻算法是分类数据最简单最有效的算法，本章通过两个例子讲述了如何使用k-近邻算法 构造分类器。
# k-近邻算法是基于实例的学习，使用算法时我们必须有接近实际数据的训练样本数 据。k-近邻算法必须
# 保存全部数据集，如果训练数据集的很大，必须使用大量的存储空间。此外， 由于必须对数据集中的每个
# 数据计算距离值，实际使用时可能非常耗时。
# k-近邻算法的另一个缺陷是它无法给出任何数据的基础结构信息，因此我们也无法知晓平均 实例样本和
# 典型实例样本具有什么特征。下一章我们将使用概率测量方法处理分类问题，该算法 可以解决这个问题。

# 创建测试数据 4*2 矩阵（4行*2列）
# @return： group数据点，labels对应每个点所属的分类，相当于测试数据（x，y）
def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

# 机器学习算法是k-近邻算法(kNN)，它的工作原理是:存在一个样本数 据集合，也称作训练样本集，
# 并且样本集中每个数据都存在标签，即我们知道样本集中每一数据 与所属分类的对应关系。输入没
# 有标签的新数据后，将新数据的每个特征与样本集中数据对应的 特征进行比较，然后算法提取样本
# 集中特征最相似数据(最近邻)的分类标签。一般来说，我们 只选择样本数据集中前k个最相似的数
# 据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。 最后，选择k个最相似数据中出现次
# 数最多的分类，作为新数据的分类。
# 距离计算公式：A(x,y,z) B(x,y,z) =>
# distance = 根号下((A.x-B.x)的平方 + (A.y-B.y)的平方 + (A.z-B.z)的平方 )
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
	sortedClassCount = sorted(classCount.iteritems(), key = \
	operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]



# 测试近邻算法
def classify0Test(inX):
	k = 3
	group, labels = createDataSet()
	result = classify0(inX, group, labels, 3)
	print(result)



# 将文本数据处理为符合格式的数据
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



# 数据归一化
# 公式： newValue = (oldValue-min)/(max-min)
# 用输入数据 减去 数据集中最小值 除以 数据集中最大值 与 最小值 的差
def autoNorm(dataSet):
	minVals = dataSet.min(0)													# 参数0表示取出矩阵每列的最小值 => [0, 3, 4]
	maxVals = dataSet.max(0)
	ranges  = maxVals - minVals
	normDataSet = zeros(shape(dataSet))											# 按输入参数生成值都为0的占位矩阵
	m = dataSet.shape[0]														# 获取行数
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = dataSet / tile(ranges, (m, 1))
	return normDataSet, ranges, minVals



# 分类器针对约会网站数据测试
def datingClassTest():
	hotRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hotRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classfierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], \
		datingLabels[numTestVecs:m],3)
		print "the classifier came back with: %d, the real answer is: %d" % \
		(classfierResult, datingLabels[i])
		if (classfierResult != datingLabels[i]): errorCount += 1.0
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


# 使用分类算法
def classifyPerson():
	resultList = ['不适合', '适合', '非常适合']									  # 不适合、适合、非常适合
	percentTats = float(\
	raw_input("percentage of time spent playing video games?"))					# 接受用户输入
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	normInX = (inArr-minVals)/ranges
	classifierResult = classify0(normInX, normMat, datingLabels, 3)
	print "You will probably like this person:",resultList[classifierResult - 1]



# 将图像转为测试向量 32 * 32 => 1 * (32 * 32) => 1 * 1024
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect



# 手写数字识别系统的测试代码
def handwritingClassTest():
																				# 处理训练数据开始
	hwLabels = []
	trainingFileList = listdir('trainingDigits')								# 文件名数组
	m = len(trainingFileList)													# txt文件总个数
	trainingMat = zeros((m,1024))												# m * 1024 矩阵
	for i in range(m):
		fileNameStr = trainingFileList[i]										# 0_0.txt
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])								# 获取数字0
		hwLabels.append(classNumStr)											# 生成数字标签矩阵
		trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)		# 生成数据矩阵
																				# 处理训练数据结束

																				# 处理测试数据开始
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):														# 循环测试每个测试数据与训练数据的距离
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print "the classifier came back with: %d, the real answer is: %d" \
		% (classifierResult, classNumStr)
		if (classifierResult != classNumStr): errorCount += 1.0
		print "\nthe total number of errors is: %d" % errorCount
		print "\nthe total error rate is: %f" % (errorCount/float(mTest))
