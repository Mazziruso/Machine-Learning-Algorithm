import java.util.Arrays;
import java.io.*;
/*
 * 最近邻搜索
 * 利用kd tree
 * 通过先创建查找二叉树
 * 在利用中序遍历查找
 */

public class KdNearest {
	
	public static void main(String[] args) {
		
		//训练数据
		int N = 5000;
		double[][] A = new double[N][3];
		
		String dir = "TrainData.dat";
		FileReader fr = null;
		BufferedReader br = null;
		
		try {
			fr = new FileReader(dir);
			br = new BufferedReader(fr);
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		try {
			String s = null;
			String[] sr;
			int i = 0;
			while((s=br.readLine()) != null) {
				sr = s.split(" ");
				A[i][0] = Double.valueOf(sr[0]);
				A[i][1] = Double.valueOf(sr[1]);
				A[i][2] = Double.valueOf(sr[2]);
				i++;
			}
			br.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("Reading Successfully");

		
		KTree kt = new KTree(A, 2);
		
		KTree.preorder(kt.root);
	
		double[] target = {1, 1};
	
		ResultNode res = KTree.findTree(kt, target);
		System.out.println(res);
		
	}
	
	public static void printA(double[][] A) {
		for(int i=0; i<A.length; i++) {
			System.out.println(Arrays.toString(A[i]));
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
	
}

//遍历返回结果的形式
class ResultNode {
	Node nearestPoint;
	double nearestDist;
	int nodesVisited;
	
	ResultNode(Node np, double nd, int nv) {
		this.nearestPoint = np;
		this.nearestDist = nd;
		this.nodesVisited = nv;
	}
	
	public String toString() {
		return "nearestPoint= " + Arrays.toString(nearestPoint.element) + ", nearestDist= " + nearestDist + 
				", nodes_visited= " + nodesVisited;
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
	public static ResultNode findTree(KTree kt, double[] point) {
		int K = point.length;
		return KTree.travel(kt.root, point, Double.MAX_VALUE, K);
	}
	
	//中序遍历当前结点的子树，并记录该子树的最近点、最小距离、遍历过的点
	public static ResultNode travel(Node kdNode, double[] target, double maxDist, int K) {
		//base-case
		if(kdNode == null) {
			return new ResultNode(null, Double.MAX_VALUE, 0);
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
		ResultNode tempNodeNearest = KTree.travel(nearNode, target, maxDist, K);
		Node nearest = tempNodeNearest.nearestPoint;
		double dist = tempNodeNearest.nearestDist;
		nodes_visited += tempNodeNearest.nodesVisited;
		
		//时时更新当前最大距离，默认最大距离是inf
		if(dist<maxDist) {
			maxDist = dist;
		}
		
		//目标点与分离超平面之间的距离
		//若最大距离圈小于目标点与超平面的距离，则直接返回上一级根节点比较
		if(maxDist < Math.abs(pivot[s]-target[s])) {
			return new ResultNode(nearest, maxDist, nodes_visited);
		}
		
		//计算目标点与当前结点的距离
		double tempDist = 0;
		for(int i=0; i<K; i++) {
			tempDist += Math.pow((pivot[i] - target[i]), 2);
		}
		tempDist = Math.sqrt(tempDist);
		
		//若当前结点的距离比之前最近点还近，则将最近点换成当前点
		if(tempDist<dist) {
			nearest = kdNode;
			dist = tempDist;
			maxDist = dist;
		}
		
		//再遍历右结点
		ResultNode tempNodeFurther = KTree.travel(farNode, target, maxDist, K);
		
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
