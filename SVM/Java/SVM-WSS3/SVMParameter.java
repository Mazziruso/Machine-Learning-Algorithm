package SVM;

public class SVMParameter {

    //Kernel Type
    public static final int LINEAR = 0;
    public static final int POLY = 1;
    public static final int RBF = 2;
    public static final int SIGMOID = 3;

    //SVM Parameter
    public int kernel_type;
    public double degree; //for POLY
    public double sigma; //for POLY/RBF/SIGMOID
    public double coef0; //for POLY/SIGMOID
    public double eps; //|L-H|相等誤差
    public double C;
    public double tao; //Quad Coefficient(aij)

    public SVMParameter() {
        this.kernel_type = 2;
        this.degree = 3;
        this.sigma = 1;
        this.coef0 = 0;
        this.eps = 1e-3D;
        this.C = 1.0D;
        this.tao = 1e-12D;
    }

}
