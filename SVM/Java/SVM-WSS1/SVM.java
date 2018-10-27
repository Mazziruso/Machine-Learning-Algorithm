package svm;

import java.io.*;
import java.util.StringTokenizer;
import java.util.Vector;

public class SVM {

    public static SVMModel svmTrain(SVMProblem prob, SVMParameter param) {
        SVMModel model = new SVMModel();

        //初始化模型参数
        SVM.initialize(model, param, prob);

        SVM.variablesSelct_alpha(model, prob, param);

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

        SVM.modelInfo(model);

        return model;
    }

    private static void variablesSelct_alpha(SVMModel model, SVMProblem prob, SVMParameter param) {
        int numChanged = 0;
        boolean examineAll = true;
        /*
        启发式寻找优化变量
        以下两层循环，开始时检查所有样本，选择不符合KKT条件的两个乘子进行优化，选择成功，返回true，否则，返回false
        所以成功了，numChanged必然>0，从第二遍循环时，不从整个样本中去寻找不符合KKT条件的两个乘子进行优化，
        而是从边界間隔乘子中去寻找，因为边界間隔样本需要调整的可能性更大，非边界間隔样本往往不被调整而始终停留在非边界上。
        如果没找到，再从整个样本中去找，直到整个样本中再也找不到需要改变的乘子为止，此时算法结束。
        */
        while(numChanged>0 || examineAll) {
            numChanged = 0;
            if(examineAll) {
                for(int i=0; i<prob.l; i++) {
                    numChanged += examineExample(i, prob, model);
                }
            } else {
                for(int i=0; i<prob.l; i++) {
                    if(model.alpha[i]>0 && model.alpha[i]<param.C) {
                        numChanged += examineExample(i, prob, model);
                    }
                }
            }

            if(examineAll) {
                examineAll = false;
            } else if(numChanged == 0) {
                examineAll = true;
            }
        }
    }

    private static int examineExample(int index1, SVMProblem prob, SVMModel model) {
        /*
        y1: 第一个优化变量的类别
        alpha1: 第一个优化变量的旧值
        E1: 第一个变量多对应的函数误差值，g(x1)-y1
        r1: 用来判别是否满足KKT条件
         */
        double y1;
        double alpha1; //old alpha1 value
        double E1;
        double r1; //r1 = y1 * E1 = y1*g(x)-1

        alpha1 = model.alpha[index1];
        y1 = prob.y[index1];
        E1 = model.E[index1];

        r1 = y1 * E1; // r1 = y*g(x)-1

        /*
        KKT条件:
            1: alpha1=00   <--> y1*g(x1)-1>=0 <--> r1>=0
            2: 0<alpha2<C  <--> y1*g(x1)-1==0 <--> r1==0
            3: alpha1==C   <--> y1*g(x1)-1<=0 <--> r1<=0
        当满足((alpha1==0 or r1<=0) and (alpha1==C or r1>=0))时，满足KKT条件。
        以tolerance为判别误差，将该式改为:
        ((alpha1==0 or r1<=-tolerance) and (alpha1==C or r1>=tolerance))
        上式的逆否条件是:
        ((alpha1>0 or r1>tolerance) and (alpha1<C or r1<-tolerance)), 此时不满足KKT条件
         */
        /*
        若第一个优化变量alpha1违反KKT条件时，通过以下方式选取第二个优化变量alpha2
            1: 在边界间隔(支持向量)上找寻|E1-E2|最大的点，将它作为第二个优化变量
            2: 若1找不到，那么遍历随机找寻一个边界间隔(支持向量)上的点作为第二个优化变量
            3: 若2的优化结果不成功，那么在所有的训练集中随机找寻一个点作为第二个优化变量
            4: 若都优化结果没有太大进展，则返回0，表明alpha1当前已是最优
         */
        if((alpha1>0 && r1>model.param.tolerance) || (alpha1<model.param.C && r1<(-1*model.param.tolerance))) {

            //方式1，在边界间隔上找寻|E1-E2|最大的点
            if(examineFirstChoice(index1, E1, prob, model)) {
                return 1;
            }

            //方式2，在边界间隔上随机找寻可优化点
            if(examineBound(index1, prob, model)) {
                return 1;
            }

            //方式3，在整个训练集上随机找寻可优化点
            if(examineNonBound(index1, prob, model)) {
                return 1;
            }
        }

        return 0;
    }

    private static boolean examineFirstChoice(int index1, double E1, SVMProblem prob, SVMModel model) {
        int index2 = -1;
        double fmax = 0.0D;
        double E2;

        double temp;
        for(int k=0; k<prob.l; k++) {

            //间隔边界上的点
            if(model.alpha[k]>0 && model.alpha[k]<model.param.C) {
                E2 = model.E[k];

                temp = Math.abs(E1-E2);
                if(temp > fmax) {
                    fmax = temp;
                    index2 = k;
                }
            }
        }

        //如果找到这样一个点alpha2,那么进行优化，并判断是否优化成功
        if(index2 >= 0) {
            if(SVM.takeStep(index1, index2, prob, model)) {
                return true;
            }
        }

        return false;
    }

    private static boolean examineBound(int index1, SVMProblem prob, SVMModel model) {
        int index2 = (int)(Math.random()*(prob.l-1));

        for(int k=0; k<prob.l; k++) {

            //边界间隔上随机找寻可行点，找到即可优化退出
            if(model.alpha[index2]>0 && model.alpha[index2]<model.param.C) {
                if (SVM.takeStep(index1, index2, prob, model)) {
                    return true;
                }
            }
            index2 = (index2+1) % prob.l;
        }

        return false;
    }

    private static boolean examineNonBound(int index1, SVMProblem prob, SVMModel model) {
        int index2 = (int)(Math.random()*(prob.l-1));

        //在整个训练集上随机找寻可行点，找出并优化退出
        for(int k=0; k<prob.l; k++) {
            if(SVM.takeStep(index1, index2, prob, model)) {
                return true;
            }
            index2 = (index2+1) % prob.l;
        }
        return false;
    }

    private static boolean takeStep(int index1, int index2, SVMProblem prob, SVMModel model) {
        //C
        double C = model.param.C;
        //函数误差
        double E1 = model.E[index1];
        double E2 = model.E[index2];
        //alpha2更新后的取值范围
        double L;
        double H;
        //alpha1和alpha2更新后的值
        double a1;
        double a2;
        //alpha1和alpha2旧值
        double alpha1 = model.alpha[index1];
        double alpha2 = model.alpha[index2];
        //
        double eta;
        //
        double y1 = prob.y[index1];
        double y2 = prob.y[index2];
        //阈值b以及b的更新前后差
        double b_old = model.b;
        double b;
        double delta_b;

        //计算L和H
        double gamma;
        if(y1 == y2) {
            gamma = alpha1 + alpha2;
            L = Math.max(0, gamma-C);
            H = Math.min(C, gamma);
        } else {
            gamma = alpha2-alpha1;
            L = Math.max(0, gamma);
            H = Math.min(C, C+gamma);
        }

        //compute eta
        double k11 = SVM.kernelFunc(index1, index1, prob, model);
        double k22 = SVM.kernelFunc(index2, index2, prob, model);
        double k12 = SVM.kernelFunc(index1, index2, prob, model);
        eta = k11 + k22 - 2*k12;

        //compute a2
        double Lobj; //当a2=L, W(a2)的值，书p128
        double Hobj; //当a2=H, W(a2)的值，书p128
        double val1;
        if(eta>0) {
            a2 = alpha2 + y2*(E1-E2)/eta;

            //剪切以满足取值范围要求
            if(a2 > H) {
                a2 = H;
            } else if(a2 < L) {
                a2 = L;
            }
        }
        else {
            //若eta<=0时，不符合特征空间模平方的特性，需另做处理
            val1 = y2*(E2-E1)-eta*alpha2;
            Lobj = eta*L*L/2 + L*val1;
            Hobj = eta*H*H/2 + H*val1;
            a2 = (Lobj>Hobj) ? H : L;
        }

        //判断alpha2更新前后的差是否满足停止条件
        if(Math.abs(a2-alpha2) < model.param.eps) {
            return false;
        } //若alpha2更新小于设定值，说明优化没有进展

        //compute a1
        a1 = alpha1 + y1*y2*(alpha2-a2);

        //更新阈值b
        if(a1>0 && a1<C && a2>0 && a2<C) {
            b = b_old - E1 - y1*k11*(a1-alpha1) - y2*k12*(a2-alpha2);
        } else {
            double b1 = b_old - E1 - y1*k11*(a1-alpha1) - y2*k12*(a2-alpha2);
            double b2 = b_old - E2 - y1*k12*(a1-alpha1) - y2*k22*(a2-alpha2);
            b = (b1+b2)/2;
        }
        //b更新前后差
        delta_b = b-b_old;

        //更新所有函数误差Ei
        double Ei_old;
        double k1i;
        double k2i;
        double delta_alpha1 = a1 - alpha1;
        double delta_alpha2 = a2 - alpha2;
        for(int i=0; i<prob.l; i++) {
            if(i==index1 || i==index2) {
                model.E[i] = 0.0D;
            } else {
                Ei_old = model.E[i];
                k1i = SVM.kernelFunc(index1, i, prob, model);
                k2i = SVM.kernelFunc(index2, i, prob, model);
                model.E[i] = Ei_old+y1*k1i*delta_alpha1+y2*k2i*delta_alpha2+delta_b;
            }
        }

        //更新alpha值
        model.alpha[index1] = a1;
        model.alpha[index2] = a2;

        //更新阈值b
        model.b = b;

        return true; //优化取得进展
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

    //初始化模型和训练起始参数
    private static void initialize(SVMModel model, SVMParameter param, SVMProblem prob) {
        model.param = param;
        model.N = prob.l;
        model.nr_class = 2;
        model.alpha = new double[model.N];
        model.E = new double[model.N];

        for(int i=0; i<model.N; i++) {
            model.alpha[i] = 0.0D;
            model.E[i] = -1*prob.y[i];
        }

        model.b = 0.0D;
        model.labels = new int[2];
        model.labels[0] = -1;
        model.labels[1] = 1;
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
