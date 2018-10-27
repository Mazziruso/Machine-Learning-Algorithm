package KDTreeKNN;

import java.util.Arrays;
import java.io.*;
/*
 * 最近邻搜索
 * 利用kd tree
 * 通过先创建查找二叉树
 * 在利用中序遍历查找
 */

public class KdKNN {
	
	public static void main(String[] args) {
		
		//Textbook Example
//		double[][] A = {{2,3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}};
//		double[] point = {4.5, 5.5};
//		KTree kt = new KTree(A, 2);
//		KTree.preorder(kt.root);
//		ResultNode[] res = KTree.findTree(kt, point, 2);
//		printResNode(res);
		
//******************************************************//
//      数据封装格式
//		M x N数组形式，共M个数据，（1：N-1）是数据特征输入，最后一列是分类结果
		int dimN = 2;//特征数据维度
		
//		读取训练数据
		int trainN = 5000;
		double[][] TrainData = new double[trainN][dimN+1];
		
		String dir = "E:\\JavaWorkspace\\KNN\\src\\KdTree\\TrainData.dat";
		FileReader fr = null;
		BufferedReader br = null;
		
		try {
			fr = new FileReader(dir);
			br = new BufferedReader(fr);
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		String s = null;
		String[] sr;
		int numData = 0;
		
		try {
			while((s=br.readLine()) != null) {
				sr = s.split(" ");
				TrainData[numData][0] = Double.valueOf(sr[0]);
				TrainData[numData][1] = Double.valueOf(sr[1]);
				TrainData[numData][2] = Double.valueOf(sr[2]);
				numData++;
			}
			br.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("Train Data Reading Successfully");
		
		double[] testData = {3, 1};
		
		//根据训练序列构建kd树
		KTree kt = new KTree(TrainData, dimN);
		
		//遍历查找k个近邻值，并以一定数据格式保存显示
		//k大小设置，k过小容易过拟合，过大易欠拟合，一般选择训练数据集大小的平方根左右
		//保存格式：Node, Distance, VisitedNodes
		int k = (int)Math.sqrt(trainN);
		ResultNode[] res = KTree.findTree(kt, testData, k);
		
		//投票表决，输入k近邻点数组以及各点分类结果
		ResultVote ballot = new ResultVote(res, dimN);
		
		//结果显示
		System.out.println();
		System.out.println("KNN input point: ");
		System.out.println(Arrays.toString(testData));
		System.out.println();
		System.out.println("KNN prediction output: ");
		System.out.println(ballot.voteLabel());
		
		
		//读取测试数据集
		int testN = 500;
		double[][] TestData = new double[testN][dimN+1];
		
		dir = "E:\\JavaWorkspace\\KNN\\src\\KdTree\\TestData.dat";
		fr = null;
		br = null;
		
		try {
			fr = new FileReader(dir);
			br = new BufferedReader(fr);
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		s = null;
		numData = 0;
		
		try {
			while((s=br.readLine()) != null) {
				sr = s.split(" ");
				TestData[numData][0] = Double.valueOf(sr[0]);
				TestData[numData][1] = Double.valueOf(sr[1]);
				TestData[numData][2] = Double.valueOf(sr[2]);
				numData++;
			}
			br.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
		
		System.out.println();
		System.out.println("Test Data Reading Successfully");
		
		//测试数据集进行模型验证
		int error = 0;
		for(int i=0; i<testN; i++) {
			ballot = new ResultVote(KTree.findTree(kt, TestData[i], k), dimN);
			if(ballot.voteLabel() != TestData[i][dimN]) {
				error++;
			}
		}
		System.out.println();
		System.out.println("size of train dataset: " + trainN + " size of test dataset: " + testN + " k: " + k);
		System.out.println("Classification error rate: ");
		System.out.println(error/testN);
		
	}
	
	//打印二维数组
	public static void printA(double[][] A) {
		for(int i=0; i<A.length; i++) {
			System.out.println(Arrays.toString(A[i]));
		}
	}
	
	//打印k个近邻点封装结果数组
	public static void printResNode(ResultNode[] res) {
		for(int i=0; i<res.length; i++) {
			System.out.println(res[i]);
		}
	}

}

class Node {
	double[] element;
	int dim;//划分的维度
	Node left;
	Node right;
	
	Node(double[] e, int dim, Node left, Node right) {
		this.element = e;
		this.dim = dim;
		this.left = left;
		this.right = right;
	}
	
	Node() {
		this(null, 0, null, null);
	}
	
}

//遍历返回结果的形式
class ResultNode implements Comparable<ResultNode>{
	Node nearestPoint;
	double nearestDist;
	int nodesVisited;
	
	ResultNode(Node np, double nd, int nv) {
		this.nearestPoint = np;
		this.nearestDist = nd;
		this.nodesVisited = nv;
	}
	
	ResultNode(double nd) {
		this(null, nd, 0);
	}
	
	public String toString() {
		return (nearestPoint!=null) ? 
				("k nearest node= " + Arrays.toString(nearestPoint.element) + ", nearestDist= " + nearestDist + 
				", nodes_visited= " + nodesVisited) : 
				("k nearest node= null" + ", nearestDist= " + nearestDist + ", nodes_visited= " + nodesVisited);
	}
	
	public int compareTo(ResultNode node) {
		return (this.nearestDist<node.nearestDist) ? -1 : (this.nearestDist==node.nearestDist) ? 0 : 1;
	}
}

//投票表决类，多数表决策略
//输入k个近邻结果数组以及分类输出的维度，以判断封装的数据帧结构中特征维度与分类输出y的维度
class ResultVote {
	private int[] resultLabel;
	
	public ResultVote(ResultNode[] result, int dimLabel) {
		int k = result.length;
		this.resultLabel = new int[100];
		for(int i=0; i<k; i++) {
			this.resultLabel[(int) result[i].nearestPoint.element[dimLabel]]++;
		}
	}
	
	public int voteLabel()  {
		return findMax(this.resultLabel);
	}
	
	public static int findMax(int[] A) {
		int index = 0;
		int max = Integer.MIN_VALUE;
		
		for(int i=0; i<A.length; i++) {
			if(A[i]>max) {
				max = A[i];
				index = i;
			}
		}
		
		return index;
	}
}

class KTree {
	Node root;
	int K;
	
	KTree(double[][] A, int K) {
		this.K = K;
		this.root = KTree.createTree(0, A, 0, (A.length-1), K);
	}
	
	KTree(double[][] A) {
		this(A, A[0].length);
	}
	
	//创建kd树
	public static Node createTree(int dim, double[][] A, int start, int end, int K) {
		if(start>end) {
			return null;
		}
		
		//选择算法，选择中位数
		int split_pos = QuickSort.randomSelect(A, dim, start, end, (end-start+1)/2+1);
		double[] median = A[split_pos];
		
		int dim_upgrade = (dim+1)%K;
		
		return new Node(median, dim, 
				createTree(dim_upgrade, A, start, (split_pos-1), K),
				createTree(dim_upgrade, A, (split_pos+1), end, K));
	}
	
	//查找kd树的最近邻点
	public static ResultNode[] findTree(KTree kt, double[] point, int k) {
		ResultNode[] res = new ResultNode[k];
		for(int i=0; i<k; i++) {
			res[i] = new ResultNode(Double.MAX_VALUE);
		}
		
		int dimK = point.length;
		KTree.travel(kt.root, point, Double.MAX_VALUE, dimK, res, k);
		return res;
	}
	
	//中序遍历当前结点的子树，并记录该子树的最近点、最小距离、遍历过的点
	//输入当前遍历的结点，目标点， 最大距离， 特征维度数，结果优先队列，结果近邻点数
	public static ResultNode travel(Node kdNode, double[] target, double maxDist, int dimK, ResultNode[] result, int k) {
		ResultNode resNode;
		//base-case
		if(kdNode == null) {
			resNode = new ResultNode(null, Double.MAX_VALUE, 0);
			if(result[0].nearestPoint == null) {
				result[0] = resNode;
				MaxHeap.maxHeapify(result, 0, k);
			}
			return resNode;
		}
		
		int nodes_visited = 1;
		int s = kdNode.dim; //当前结点划分的维度
		double[] pivot = kdNode.element; //当前结点的元素
		
		//先向下深度遍历,找到最近的叶结点
		Node nearNode;
		Node farNode;
		if(target[s] <= pivot[s]) {
			nearNode = kdNode.left;
			farNode = kdNode.right;
		} else {
			nearNode = kdNode.right;
			farNode = kdNode.left;
		}
		
		//递归调用最近结点，直达叶结点
		ResultNode tempNodeNearest = KTree.travel(nearNode, target, maxDist, dimK, result, k);
		Node nearest = tempNodeNearest.nearestPoint;
		double dist = tempNodeNearest.nearestDist;
		nodes_visited += tempNodeNearest.nodesVisited;
		
		//时时更新当前最大距离，默认最大距离是inf
		if(result[0].nearestDist<maxDist) {
			maxDist = result[0].nearestDist;
		}
		
		//计算目标点与当前结点的距离
		double tempDist = 0;
		for(int i=0; i<dimK; i++) {
			tempDist += Math.pow((pivot[i] - target[i]), 2);
		}
		tempDist = Math.sqrt(tempDist);
		
		//目标点与分离超平面之间的距离
		//若最大距离圈小于目标点与超平面的距离，则直接返回上一级根节点比较
		if((maxDist < Math.abs(pivot[s]-target[s])) & (result[0].nearestPoint!=null)) {
			return new ResultNode(nearest, dist, nodes_visited);
		}
		
		//若当前结点的距离比之前最近点还近，则将最近点换成当前点
		if(tempDist<dist) {
			nearest = kdNode;
			dist = tempDist;
			maxDist = dist;
		}
		//保存当前结点放进优先队列
		if(tempDist<result[0].nearestDist) {
			resNode = new ResultNode(kdNode, tempDist, nodes_visited);
			result[0] = resNode;
			MaxHeap.maxHeapify(result, 0, k);
		}
		
		//再遍历右结点
		ResultNode tempNodeFurther = KTree.travel(farNode, target, maxDist, dimK, result, k);
		
		nodes_visited += tempNodeFurther.nodesVisited;
		if(tempNodeFurther.nearestDist < dist) {
			nearest = tempNodeFurther.nearestPoint;
			dist = tempNodeFurther.nearestDist;
		}
		
		return new ResultNode(nearest, dist, nodes_visited);
	}

	//先序遍历
	public static void preorder(Node root) {
		if(root != null) {
			System.out.println(Arrays.toString(root.element));
			preorder(root.left);
			preorder(root.right);
		}
	}
}

class QuickSort {
	
	public static void sort(double[][] A, int dim, int start, int end) {
		if(start<end) {
			int q = randomPartition(A, dim, start, end);
			QuickSort.sort(A, dim, start, q-1);
			QuickSort.sort(A, dim, q+1, end);
		}
		
	}
	
	public static int partition(double[][] A, int dim, int start, int end) {
		double[] x = A[end];
		int i = start - 1;
		double[] temp;
		
		for(int j=start; j<end; j++) {
			if(A[j][dim]<=x[dim]) {
				i++;
				temp = A[j];
				A[j] = A[i];
				A[i] = temp;
			}
		}
		
		i++;
		A[end] = A[i];
		A[i] = x;
		
		return i;
	}
	
	public static int randomPartition(double[][] A, int dim, int start, int end) {
		if(start>end) {
			return -1;
		}
		
		int i = (int)(Math.random() * (end-start)) + start;
		double[] temp = A[end];
		A[end] = A[i];
		A[i] = temp;
	
		return partition(A, dim, start, end);
		
	}
	
//	中位数选择算法
	public static int randomSelect(double[][] A, int dim, int start, int end, int index) {
		if(start == end) {
			return start;//要与实际代码中的数组下标对应
		}
		
		int q = QuickSort.randomPartition(A, dim, start, end);
		if(q<start || q>end) {
			return -1;
		}
		
		//base-case
		int k = q-start+1;
		if(k == index) {
			return q;//要与实际代码中的数组下标对应
		} else if(index < k) {
			return QuickSort.randomSelect(A, dim, start, q-1, index);
		} else {
			return QuickSort.randomSelect(A, dim, q+1, end, index-k);
		}
	}
}


//利用最大堆来构造优先队列， 存储k个近邻点
class MaxHeap {
	
	public static int parent(int i) {
		return (i-1)/2;
	}
	
	public static int lChild(int i) {
		return 2*i+1;
	}
	
	public static int rChild(int i) {
		return 2*i+2;
	}
	
	public static void maxHeapify(ResultNode[] t, int i, int heap_size) {
		int l = lChild(i);
		int r = rChild(i);
		int largest = i;
		
		if(l < heap_size) {
			largest = (t[l].compareTo(t[i])>0) ? l : i;
		}
		if(r < heap_size) {
			largest = (t[r].compareTo(t[largest])>0) ? r : largest;
		}
		
		if(largest != i) {
			ResultNode tmp = t[i];
			t[i] = t[largest];
			t[largest] = tmp;
			maxHeapify(t, largest, heap_size);
		}
	}
	
	public static void buildMaxHeap(ResultNode[] t, int heap_size) {
		for(int i=(heap_size/2)-1; i>=0; i--) {
			maxHeapify(t, i, heap_size);
		}
	}
}
