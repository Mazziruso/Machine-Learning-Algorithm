package SVM;

public class SVMNode {
    public int index;
    public double value;

    public SVMNode(int index, double value) {
        this.index = index;
        this.value = value;
    }

    public SVMNode() {
        this(0, 0.0D);
    }
}
