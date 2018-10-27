package SVM;

import java.io.*;
import java.util.StringTokenizer;
import java.util.Vector;

public class SVM {

    private static double[] G;
    private static double[] G_bar;
    private static double[][] Q;
    private static int[] A; //存储起作用训练数据向量

    public static SVMModel svmTrain(SVMProblem prob, SVMParameter param) {
        SVMModel model = new SVMModel();

        //初始化模型参数
        SVM.initialize(model, param, prob);

        //进行模型训练
        long times = SVM.examineExampleWSS2(model, prob, param);


        //构造并存储训练完成后的模型及参数
        model.l = 0;
        model.nSV = new int[2];
        Vector<Integer> SVIndices = new Vector<Integer>();
        Vector<SVMNode[]> SVNode = new Vector<SVMNode[]>();
        Vector<Double> SVLabels = new Vector<Double>();
        for(int i=0; i<prob.l; i++) {
            if(model.alpha[i] > 0) {
                model.l++;
                SVIndices.addElement(i);
                SVNode.addElement(prob.x[i]);
                SVLabels.addElement(prob.y[i]);

                if(prob.y[i] > 0) {
                    model.nSV[1]++;
                } else {
                    model.nSV[0]++;
                }
            }
        }
        model.SV = new SVMNode[model.l][];
        model.SV_indices = new int[model.l];
        model.ySV = new double[model.l];
        for(int i=0; i<model.l; i++) {
            model.SV[i] = SVNode.elementAt(i);
            model.SV_indices[i] = SVIndices.elementAt(i);
            model.ySV[i] = SVLabels.elementAt(i);
        }

        SVNode.clear();
        SVLabels.clear();
        SVIndices.clear();

        SVM.info("*");
        SVM.info("Iterator times: " + times);
        SVM.modelInfo(model);

        return model;
    }

    private static long examineExampleWSS2(SVMModel model, SVMProblem prob, SVMParameter param) {
        int[] B;
        int index_i;
        int index_j;
        long times = 0L;

        B = SVM.variablesSelect_alpha_WSS2(model, prob, param);
        while(model.mU-model.mL >= param.eps) {
            index_i = B[0];
            index_j = B[1];

            QPsolver(index_i, index_j, model, prob, param);
            B = SVM.variablesSelect_alpha_WSS2(model, prob, param);

            times++;
        }

        //calculate b
        SVM.bSolver(model, prob, param);

        return times;
    }

    //利用目标函数的一阶信息量进行起作用集选择
    private static int[] variablesSelect_alpha_WSS2(SVMModel model, SVMProblem prob, SVMParameter param) {
        double mL_temp = Double.MAX_VALUE;
        double mU_temp = -Double.MAX_VALUE;

        int index_i = 0;
        int index_j = 0;

        double temp;
        for(int i=0; i<prob.l; i++) {
            if(SVM.examineIup(i, model, prob, param)) {
                temp = -1.0 * prob.y[i] * SVM.G[i];
                if(temp > mU_temp) {
                    mU_temp = temp;
                    index_i = i;
                }
            }
            if(SVM.examineIlow(i, model, prob, param)) {
                temp = -1.0 * prob.y[i] * SVM.G[i];
                if(temp < mL_temp) {
                    mL_temp = temp;
                    index_j = i;
                }
            }
        }

        model.mU = mU_temp;
        model.mL = mL_temp;

        return new int[]{index_i, index_j};

    }

    private static boolean examineIup(int index, SVMModel model, SVMProblem prob, SVMParameter param) {
        if(model.alpha[index]<param.C && prob.y[index]>0) {
            return true;
        }
        if(model.alpha[index]>0 && prob.y[index]<0) {
            return true;
        }

        return false;
    }

    private static boolean examineIlow(int index, SVMModel model, SVMProblem prob, SVMParameter param) {
        if(model.alpha[index]<param.C && prob.y[index]<0) {
            return true;
        }
        if(model.alpha[index]>0 && prob.y[index]>0) {
            return true;
        }

        return false;
    }

//    private static double cal

    private static void QPsolver(int index_i, int index_j, SVMModel model, SVMProblem prob, SVMParameter param) {
        double yi = prob.y[index_i];
        double yj = prob.y[index_j];
        double Kii = SVM.Q[index_i][index_i];
        double Kjj = SVM.Q[index_j][index_j];
        double Kij = SVM.Q[index_i][index_j] * yi * yj;

        //Quad Coef, aij
        double aij = Kii + Kjj - 2*Kij;
        if(aij <= param.tao) {
            aij = param.tao;
        }

        //bij
        double bij = -yi * SVM.G[index_i] + yj * SVM.G[index_j];

        //new alpha i, j(uncutted)
        double alpha_i;
        double alpha_j;
        double dj = -bij / aij;
        alpha_i = model.alpha[index_i] - yi * dj; //calculate uncutted alpha
        alpha_j = model.alpha[index_j] + yj * dj;

        //delta alpha
        double delta_alpha_i;
        double delta_alpha_j;

        //
        double diff = alpha_i - alpha_j;
        double summ = alpha_i + alpha_j;

        //cut alpha into constraint conditions
        if(Math.abs(yi-yj) > param.eps) { //yi!=yj due to they are Double
            if(diff > 0) {
                if(alpha_i > param.C) { //Region I
                    alpha_i = param.C;
                    alpha_j = param.C - diff;
                }
                if(alpha_j < 0) { //Region III
                    alpha_i = diff;
                    alpha_j = 0;
                }
            } else {
                if(alpha_j > param.C) { //Region II
                    alpha_i = diff + param.C;
                    alpha_j = param.C;
                }
                if(alpha_i < 0) { //Region IV
                    alpha_i = 0;
                    alpha_j = -diff;
                }
            }
        } else { //yi==yj
            if(summ > param.C) {
                if(alpha_i > param.C) { //Region I
                    alpha_i = param.C;
                    alpha_j = summ - param.C;
                }
                if(alpha_j > param.C) { //Region II
                    alpha_i = summ - param.C;
                    alpha_j = param.C;
                }
            } else {
                if(alpha_i < 0) { //Region IV
                    alpha_i = 0;
                    alpha_j = summ;
                }
                if(alpha_j < 0) { //Region III
                    alpha_i = summ;
                    alpha_j = 0;
                }
            }
        }

        //compute delta alpha
        delta_alpha_i = alpha_i - model.alpha[index_i];
        delta_alpha_j = alpha_j - model.alpha[index_j];

        //update alpha
        model.alpha[index_i] = alpha_i;
        model.alpha[index_j] = alpha_j;

        //Gradient Update
        for(int k=0; k<prob.l; k++) {
            SVM.G[k] = SVM.G[k] + SVM.Q[k][index_i] * delta_alpha_i + SVM.Q[k][index_j] * delta_alpha_j;
        }

    }

    private static void bSolver(SVMModel model, SVMProblem prob, SVMParameter param) {

        double sum = 0.0D;
        int numsBounds = 0;

        for(int i=0; i<prob.l; i++) {
            if(model.alpha[i]>0 && model.alpha[i]<param.C) {
                sum += -prob.y[i] * SVM.G[i];
                numsBounds++;
            }
        }

        if(numsBounds > 0) {
            model.b = sum / numsBounds;
        } else {
            model.b = (model.mU + model.mL)/2;
        }
    }

    //两个数据向量内积
    private static double dotProduct(SVMNode[] xi, SVMNode[] xj) {
        double dot = 0.0D;

        int k1 = 0;
        int k2 = 0;
        while(k1<xi.length && k2<xj.length) {
            if(xi[k1].index < xj[k2].index) {
                k1++;
            } else if(xi[k1].index > xj[k2].index) {
                k2++;
            } else {
                dot += xi[k1].value * xj[k2].value;
                k1++;
                k2++;
            }
        }

        return dot;
    }

    private static double dotProduct(int index1, int index2, SVMProblem prob) {
        return SVM.dotProduct(prob.x[index1], prob.x[index2]);
    }

    private static double kernelFunc(int index1, int index2, SVMProblem prob, SVMModel model) {
        /*
        0: linear kernel
        1: polynomial kernel
        2: gaussian kernel
        3: sigmoid kernel
         */

        int kernelType = model.param.kernel_type;
        double d = model.param.degree;
        double sigma = model.param.sigma;
        double coef0 = model.param.coef0;
        double val; //Output Value
        double val1;
        double dot;

        switch (kernelType) {
            case 0 :
                val = SVM.dotProduct(index1, index2, prob);
                break;

            case 1:
                dot = SVM.dotProduct(index1, index2, prob);
                val1 = sigma*dot + coef0;
                val = Math.pow(val1, d);
                break;

            case 2:
                double dot1 = SVM.dotProduct(index1, index1, prob);
                double dot2 = SVM.dotProduct(index2, index2, prob);
                double dot3 = SVM.dotProduct(index1, index2, prob);
                val1 = sigma * (dot1+dot2-2*dot3);
                val = Math.exp(-1*val1);
                break;

            case 3:
                dot = SVM.dotProduct(index1, index2, prob);
                val1 = sigma*dot + coef0;
                val = Math.tanh(val1);
                break;

            default:
                val = 0.0;
                break;
        }

        return val;
    }

    private static double kernelFunc(SVMNode[] xi, SVMNode[] xj, SVMModel model) {
        int kernelType = model.param.kernel_type;
        double d = model.param.degree;
        double sigma = model.param.sigma;
        double coef0 = model.param.coef0;
        double val; //Output Value
        double val1;
        double dot;

        switch (kernelType) {
            case 0 :
                val = SVM.dotProduct(xi, xj);
                break;

            case 1:
                dot = SVM.dotProduct(xi, xj);
                val1 = sigma*dot + coef0;
                val = Math.pow(val1, d);
                break;

            case 2:
                double dot1 = SVM.dotProduct(xi, xi);
                double dot2 = SVM.dotProduct(xj, xj);
                double dot3 = SVM.dotProduct(xi, xj);
                val1 = sigma * (dot1+dot2-2*dot3);
                val = Math.exp(-1*val1);
                break;

            case 3:
                dot = SVM.dotProduct(xi, xj);
                val1 = sigma*dot + coef0;
                val = Math.tanh(val1);
                break;

            default:
                val = 0.0;
                break;
        }

        return val;
    }

    private static void shrinking(SVMModel model, SVMProblem problem) {

    }

    //初始化模型和训练起始参数
    private static void initialize(SVMModel model, SVMParameter param, SVMProblem prob) {
        model.param = param;
        model.N = prob.l;
        model.nr_class = 2;
        model.alpha = new double[model.N];

        for(int i=0; i<model.N; i++) {
            model.alpha[i] = 0.0D;
        }

        model.b = 0.0D;
        model.labels = new int[2];
        model.labels[0] = -1;
        model.labels[1] = 1;

        model.mU = 1.0D;
        model.mL = -1.0D;

        //Q initialization Q = yi*yj*K(xi,xj)
        SVM.Q = new double[prob.l][prob.l];
        for(int i=0; i<prob.l; i++) {
            for(int j=0; j<prob.l; j++) {
                SVM.Q[i][j] = prob.y[i] * prob.y[j] * kernelFunc(i, j, prob, model);
            }
        }

        //G initialization
        SVM.G = new double[prob.l];
        for(int i=0; i<prob.l; i++) {
            SVM.G[i]  = -1;
        }
    }

    public static SVMProblem svmReadProblem(String probPath) throws IOException {
        SVMProblem prob = new SVMProblem();
        BufferedReader br = new BufferedReader(new FileReader(probPath));

        int d_max = 0; //记录最大的维度
        String s;
        SVMNode[] nodes;
        Vector<Double> labelsVec = new Vector<Double>();
        Vector<SVMNode[]> nodesVec = new Vector<SVMNode[]>();
        StringTokenizer sval;
        while((s=br.readLine()) != null) {
            sval = new StringTokenizer(s, " \n\t\f\r:");
            labelsVec.addElement(Double.valueOf(sval.nextToken()));
            nodes = new SVMNode[sval.countTokens()/2];
            for(int i=0; i<nodes.length; i++) {
                nodes[i] = new SVMNode();
                nodes[i].index = Integer.valueOf(sval.nextToken());
                nodes[i].value = Double.valueOf(sval.nextToken());

                if(nodes[i].index > d_max) {
                    d_max = nodes[i].index;
                }
            }
            nodesVec.addElement(nodes);
        }

        br.close();

        prob.l = labelsVec.size();
        prob.d = d_max;
        prob.y = new double[prob.l];
        prob.x = new SVMNode[prob.l][];

        for(int i=0; i<prob.l; i++) {
            prob.y[i] = labelsVec.elementAt(i);
            prob.x[i] = nodesVec.elementAt(i);
        }
        labelsVec.clear();
        nodesVec.clear();

        return prob;
    }

    private static final String[] kernel_type_table =
            {"LINEAR", "POLYNOMIAL", "RBF", "SIGMOID"};


    //训练模型保存及加载
    public static void svmSaveModel(String path, SVMModel model) throws IOException {
        DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)));

        SVMParameter param = model.param;

        dos.writeBytes("SVM_type: C-SVC\n");
        dos.writeBytes("Kernel_type: " + SVM.kernel_type_table[param.kernel_type] + "\n");

        if(param.kernel_type == 1) {
            dos.writeBytes("degree: " + param.degree + "\n");
        }
        if(param.kernel_type != 0) {
            dos.writeBytes("sigma: " + param.sigma + "\n");
        }
        if(param.kernel_type==1 || param.kernel_type==3) {
            dos.writeBytes("coef0: " + param.coef0 + "\n");
        }

        dos.writeBytes("number_of_class: " + model.nr_class + "\n");
        dos.writeBytes("total_support_vectors(SV): " + model.l + "\n");
        dos.writeBytes("b: " + model.b + "\n");
        dos.writeBytes("label: " + model.labels[0] + " " + model.labels[1] + "\n");
        dos.writeBytes("nr_SV: " + model.nSV[0] + " " + model.nSV[1] + "\n");

        SVMNode[] p;
        int index;
        for(int i=0; i<model.l; i++) {
            index = model.SV_indices[i];
            dos.writeBytes(String.valueOf(model.alpha[index]) + " | ");

            dos.writeBytes(String.valueOf(model.ySV[i]) + " ");

            p = model.SV[i];
            for(int j=0; j<p.length; j++) {
                dos.writeBytes(p[j].index + ":" + p[j].value + " ");
            }
            dos.writeBytes("\n");
        }

        dos.close();
    }

    public static SVMModel svmLoadModel(String path) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(path));
        SVMModel model = new SVMModel();
        model.param = new SVMParameter();

        //temporary variables
        String s;
        StringTokenizer sMod;

        //SVM TYPE
        br.readLine();

        //KERNEL TYPE
        sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
        sMod.nextToken();
        s = sMod.nextToken();
        model.param.kernel_type = s.equals("LINEAR") ? 0 : s.equals("POLYNOMIAL") ? 1 : s.equals("RBF") ? 2 : 3;

        //degree, sigma and coef0
        if(model.param.kernel_type == 1) {
            sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
            sMod.nextToken();
            s = sMod.nextToken();
            model.param.degree = Double.valueOf(s);
        }
        if(model.param.kernel_type != 0) {
            sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
            sMod.nextToken();
            s = sMod.nextToken();
            model.param.sigma = Double.valueOf(s);
        }
        if(model.param.kernel_type == 1 || model.param.kernel_type == 3) {
            sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
            sMod.nextToken();
            s = sMod.nextToken();
            model.param.coef0 = Double.valueOf(s);
        }

        //nr_class
        sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
        sMod.nextToken();
        s = sMod.nextToken();
        model.nr_class = Integer.valueOf(s);

        //total SVs, l
        sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
        sMod.nextToken();
        s = sMod.nextToken();
        model.l = Integer.valueOf(s);

        //b
        sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
        sMod.nextToken();
        s = sMod.nextToken();
        model.b = Double.valueOf(s);

        //labels
        sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
        sMod.nextToken();
        s = sMod.nextToken();
        model.labels = new int[2];
        model.labels[0] = Integer.valueOf(s);
        s = sMod.nextToken();
        model.labels[1] = Integer.valueOf(s);

        //nr_SV, nSV
        sMod = new StringTokenizer(br.readLine(), " :\n\t\r\f");
        sMod.nextToken();
        s = sMod.nextToken();
        model.nSV = new int[2];
        model.nSV[0] = Integer.valueOf(s);
        s = sMod.nextToken();
        model.nSV[1] = Integer.valueOf(s);

        //alpha, y SV
        int index = 0;
        int length;
        model.alpha = new double[model.l];
        model.SV = new SVMNode[model.l][];
        model.ySV = new double[model.l];
        while((s=br.readLine()) != null) {
            sMod = new StringTokenizer(s, " :|\n\t\r\f");
            model.alpha[index] = Double.valueOf(sMod.nextToken());
            model.ySV[index] = Double.valueOf(sMod.nextToken());
            length = sMod.countTokens()/2;
            model.SV[index] = new SVMNode[length];
            for(int i=0; i<length; i++) {
                model.SV[index][i] = new SVMNode();
                model.SV[index][i].index = Integer.valueOf(sMod.nextToken());
                model.SV[index][i].value = Double.valueOf(sMod.nextToken());
            }
            index++;
        }
        br.close();


        return model;
    }

    //预测
    public static double svmPredict(SVMModel model, SVMProblem prob, String path) throws IOException {
        DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)));
        double acc;
        long correct;
        long total = prob.l;

        dos.writeBytes("label: -1 1\n");

        correct = 0L;
        double v;
        for(int i=0; i<total; i++) {
            v = SVM.svmPredictValue(model, prob.x[i]);

            if(v == prob.y[i]) {
                correct++;
            }

            dos.writeBytes(String.valueOf(v) + "\n");
        }
        dos.close();

        acc = (double)correct/(double)total;

        SVM.info("Accuracy = "+acc*100+
                "% ("+correct+"/"+total+") (classification)\n");

        return acc;
    }

    private static double svmPredictValue(SVMModel model, SVMNode[] x) {
        double val = 0.0D;

        double yi;
        double alpha;
        int index;
        for(int i=0; i<model.l; i++) {
            if(model.SV_indices != null) {
                index = model.SV_indices[i];
            } else {
                index = i;
            }
            yi = model.ySV[i];
            alpha = model.alpha[index];

            val += alpha*yi*SVM.kernelFunc(x, model.SV[i], model);
        }

        val += model.b; //样本点与分离超平面之间的函数间隔

        double y = (val>=0) ? 1 : -1;

        return y;
    }

    public static void modelInfo(SVMModel model) {
        SVMParameter param = model.param;

        SVM.info("*************");
        SVM.info("svm_type: c-svc");
        SVM.info("kernel_type: " + SVM.kernel_type_table[param.kernel_type]);
        SVM.info("nr_class: 2");
        SVM.info("total_sv: " + model.l);
        if(param.kernel_type == 1) {
            SVM.info("degree: " + param.degree);
        }
        if(param.kernel_type != 0) {
            SVM.info("sigma: " + param.sigma);
        }
        if(param.kernel_type==1 || param.kernel_type==3) {
            SVM.info("coef0: " + param.coef0);
        }
        SVM.info("label: -1 1");
        SVM.info("nr_SV: " + model.nSV[0] + " " + model.nSV[1]);
        SVM.info("*************");
    }

    private static void info(String s) {
        System.out.println(s);
    }
}
