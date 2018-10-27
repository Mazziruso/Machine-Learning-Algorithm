package svm;

public class SVMProblem {
    public int l; //樣本數
    public int d; //維數
    public SVMNode[][] x;
    public double[] y;

    public SVMProblem() {
        this.l = 0;
        this.x = null;
        this.y = null;
    }
}
