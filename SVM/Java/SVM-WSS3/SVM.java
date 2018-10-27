package SVM;

import java.io.*;
import java.util.StringTokenizer;
import java.util.Vector;

public class SVM {

    //stopping criteria
    private static double mU; //m(a)
    private static double mL; //M(a)

    private static double[] G;
    private static double[] G_bar;
    private static double[][] Q;
    private static int[] A; //存储起作用训练数据向量
    private static int counter; //count iterations so as to shrinking

    public static SVMModel svmTrain(SVMProblem prob, SVMParameter param, String matdir) throws IOException {
        SVMModel model = new SVMModel();

        //初始化模型参数
        SVM.initialize(model, param, prob);

        //进行模型训练
        long times = SVM.svmLearningWSS3(model, prob, param, matdir);


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

    private static long svmLearningWSS3(SVMModel model, SVMProblem prob, SVMParameter param, String matdir) throws IOException {
        int[] B;
        int index_i;
        int index_j;
        long times = 0L;
        boolean shrinkFirst = true;

        //
        double val;
        DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(matdir)));

        B = SVM.variablesSelect_alpha_WSS3(model, prob, param);
        while(true) {
            index_i = B[0];
            index_j = B[1];

            //while j exits, indicating that KKT condition is not hold for sub-problem
            if(index_i<0 || index_j<0) {
                if(SVM.A.length == prob.l) {
                    break;
                } else {
                    SVM.reconstructG(model, prob);
                    //重置A
                    SVM.A = new int[prob.l];
                    for(int i=0; i<prob.l; i++) {
                        SVM.A[i] = i;
                    }
                    //在全集上判断kkt条件
                    B = SVM.variablesSelect_alpha_WSS3(model, prob, param);
                    if (B[0]<0 || B[1]<0) { //全集也满足停止条件
                        break;
                    } else {
                        SVM.counter = 1;

                        index_i = B[0];
                        index_j = B[1];
                    }
                }
            }

            //开始缩减
            if(counter == 0) {

                //第一次缩减达到精度要求
                if(shrinkFirst) {
                    if (SVM.mU - SVM.mL <= (10 * param.eps)) {
                        SVM.reconstructG(model, prob);
                        shrinkFirst = false;
                    }
                }

                SVM.shrinking(model, prob);
                counter = Math.min(1000, prob.l);
            }

            //solve quadratic problem
            SVM.QPsolver(index_i, index_j, model, prob, param);

            //evaluate object function value
            val = SVM.objFuncCal(model);
            dos.writeBytes(val + " ");

            //working set selection
            B = SVM.variablesSelect_alpha_WSS3(model, prob, param);

            times++;
            counter--;
        }

        //dos close
        dos.close();

        //calculate b
        SVM.bSolver(model, prob, param);

        return times;
    }

    //更新m(a) 和 M(a)，在缩减集上处理
    private static void updateM_alpha(SVMModel model, SVMProblem prob, SVMParameter param) {
        double mU_temp = -Double.MAX_VALUE;
        double mL_temp = Double.MAX_VALUE;
        int index;

        double temp;
        for(int i=0; i<A.length; i++) {
            index = A[i];
            if(examineIup(index, model, prob, param)) {
                temp = -1.0 * prob.y[index] * SVM.G[index];
                if(temp > mU_temp) {
                    mU_temp = temp;
                }
            }
            if(examineIlow(index, model, prob, param)) {
                temp = -1.0 * prob.y[index] * SVM.G[index];
                if(temp < mL_temp) {
                    mL_temp = temp;
                }
            }
        }

        SVM.mL = mL_temp;
        SVM.mU = mU_temp;
    }

    //在缩减集上处理
    private static int[] variablesSelect_alpha_WSS3(SVMModel model, SVMProblem prob, SVMParameter param) {
        double mU_temp = -Double.MAX_VALUE;
        double mL_temp = Double.MAX_VALUE;

        int index_i = -1;
        int index_j = -1;

        int index; //shrinking data index

        double temp;
        //select alpha i
        for(int i=0; i<A.length; i++) {
            index = A[i];
            if(SVM.examineIup(index, model, prob, param)) {
                temp = -1.0 * prob.y[index] * SVM.G[index];
                if(temp > mU_temp) {
                    mU_temp = temp;
                    index_i = index;
                }
            }
        }
        //select alpha j
        double bit;
        double ait;
        double valTemp;
        double valObjFunc = Double.MAX_VALUE;
        for(int t=0; t<A.length; t++) {
            index = A[t];
            if(SVM.examineIlow(index, model, prob, param)) {
                temp = -1.0 * prob.y[index] * SVM.G[index];
                if((mU_temp-temp) > param.eps) { //进行KKT判断，宜进行相减运算，否则会得到无穷序列
                    //calculate quad coef and bij
                    bit = mU_temp + prob.y[index] * SVM.G[index];
                    ait = SVM.Q[index_i][index_i] + SVM.Q[index][index] - 2*SVM.Q[index_i][index]*prob.y[index_i]*prob.y[index];
                    if(ait <= param.tao) { //Q is a non positive semi-define
                        ait = param.tao;
                    }

                    valTemp = -bit*bit/ait;
                    if(valTemp < valObjFunc) {
                        valObjFunc = valTemp;
                        index_j = index;
                    }
                }
                if(temp < mL_temp) {
                    mL_temp = temp;
                }
            }
        }

        SVM.mU = mU_temp;
        SVM.mL = mL_temp;

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

        //update G_bar
        SVM.updateGbar(index_i, index_j, alpha_i, alpha_j, model, prob);

        //update alpha
        model.alpha[index_i] = alpha_i;
        model.alpha[index_j] = alpha_j;

        //Gradient Update
        int index;
        for(int k=0; k<A.length; k++) {
            index = A[k];
            SVM.G[index] = SVM.G[index] + SVM.Q[index][index_i] * delta_alpha_i + SVM.Q[index][index_j] * delta_alpha_j;
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

    //目标函数值计算
    private static double objFuncCal(SVMModel model) {
        double[] alpha = model.alpha;
        int N = model.N;

        double val = 0.0D;
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                val += alpha[i]*alpha[j]*SVM.Q[i][j];
            }
        }

        for(int i=0; i<N; i++) {
            val -= alpha[i];
        }

        return val;
    }

    //缩减有效数据集
    private static void shrinking(SVMModel model, SVMProblem prob) {
        Vector<Integer> A_temp = new Vector<Integer>();
        int index;
        for(int i=0; i<SVM.A.length; i++) {
            index = SVM.A[i];
            if(!SVM.isShrink(index, model, prob)) {
                A_temp.addElement(index);
            }
        }

        SVM.A = new int[A_temp.size()];
        for(int i=0; i<SVM.A.length; i++) {
            SVM.A[i] = A_temp.elementAt(i);
        }
        A_temp.clear();
    }

    //判断当前点是否需要缩减
    private static boolean isShrink(int index, SVMModel model, SVMProblem prob) {
        boolean flag = false;

        double temp = -1.0 * prob.y[index] * SVM.G[index];

        if((prob.y[index]>0&&model.alpha[index]>=model.param.C) || (prob.y[index]<0&&model.alpha[index]<=0)) {
            if(temp > SVM.mU) {
                flag = true;
            }
        } else if((prob.y[index]>0&&model.alpha[index]<=0) || (prob.y[index]<0&&model.alpha[index]>=model.param.C)) {
            if(temp < SVM.mL) {
                flag = true;
            }
        }

        return flag;
    }

    //梯度重建
    private static void reconstructG(SVMModel model, SVMProblem prob) {
        int N = prob.l;
        int A_pointer = 0;
        int index;

        for(int i=0; i<N; i++) {
            if(i != A[A_pointer]) {
                SVM.G[i] = SVM.G_bar[i] - 1;
                for(int j=0; j<SVM.A.length; j++) {
                    index = SVM.A[j];
                    if(model.alpha[index]>0 && model.alpha[index]<model.param.C) {
                        SVM.G[i] += model.alpha[index] * SVM.Q[i][index];
                    }
                }
            } else {
                if(A_pointer<A.length-1) {
                    A_pointer++;
                }
            }
        }
    }

    //更新G_bar
    private static void updateGbar(int index_i, int index_j, double alpha_i_new, double alpha_j_new, SVMModel model, SVMProblem prob) {
        double alpha_i_old = model.alpha[index_i];
        double alpha_j_old = model.alpha[index_j];
        SVMParameter param = model.param;

        if(alpha_i_new>=param.C && alpha_i_old<param.C) {
            for(int i=0; i<prob.l; i++) {
                SVM.G_bar[i] += param.C * SVM.Q[i][index_i];
            }
        } else if(alpha_i_new<param.C && alpha_i_old>=param.C) {
            for(int i=0; i<prob.l; i++) {
                SVM.G_bar[i] -= param.C * SVM.Q[i][index_i];
            }
        }

        if(alpha_j_new>=param.C && alpha_j_old<param.C) {
            for(int i=0; i<prob.l; i++) {
                SVM.G_bar[i] += param.C * SVM.Q[i][index_j];
            }
        } else if(alpha_j_new<param.C && alpha_j_old>=param.C) {
            for(int i=0; i<prob.l; i++) {
                SVM.G_bar[i] -= param.C * SVM.Q[i][index_j];
            }
        }
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

        //G_bar initialization
        SVM.G_bar = new double[prob.l];

        //
        SVM.A = new int[prob.l];
        for(int i=0; i<prob.l; i++) {
            SVM.A[i] = i;
        }

        //
        SVM.mU = 1.0D;
        SVM.mL = -1.0D;

        //counter initialization
        SVM.counter = Math.min(1000, prob.l);
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
