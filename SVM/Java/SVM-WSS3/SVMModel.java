package SVM;

public class SVMModel {
    public SVMParameter param;
    public int nr_class;
    public int l; //total SVs
    public int N; //total Examples
    public SVMNode[][] SV;
    public double[] ySV; //SVs label

    public double[] alpha;
    public double b;

    public int[] SV_indices;
    public int[] labels;
    public int[] nSV;

    //first order information KKT condition
    public double mL;
    public double mU;
}

