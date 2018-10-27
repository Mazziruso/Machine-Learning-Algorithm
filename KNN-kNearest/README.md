# KNN的Java实现

## KdKNN.java
利用最小堆来存储kdTree的k个近邻点

## KdNearest.java
相比于kdKNN.java，不需要用最小堆存储k个数据点，只需要找到距离最小的那个结点即可，搜索花费时间O(lgN)

## TestData.dat
KdKNN.java与KdNearest.java的训练数据集

## TrainData.dat
KdKNN.java与KdNearest.java的测试数据集
