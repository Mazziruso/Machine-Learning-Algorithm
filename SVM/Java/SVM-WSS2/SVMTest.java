package SVM;

import java.io.*;

/*
Require: label needs -1 or +1
 */

public class DemoTest {
    public static void main(String[] args) throws IOException {
        String[] dir = {"SVMData\\mySVMTestWSS2\\svmguide1.scale", //训练数据集
                "SVMData\\mySVMTestWSS2\\svmguide1.model",   //存放SVM模型
                "SVMData\\mySVMTestWSS2\\svmguide1t.scale",        //测试数据集
                "SVMData\\mySVMTestWSS2\\svmguide1t.output"};//测试集label输出

        SVMProblem prob = SVM.svmReadProblem(dir[0]);

        SVMParameter param = new SVMParameter();
        //SVM Parameter setup
        param.kernel_type = 2;
        param.degree = 3;
        param.sigma = 0.25D;
        param.coef0 = 0.0D;
        param.eps = 1e-3D;
        param.tolerance = 0.001;
        param.C = 2.0D;
        param.tao = 1e-12;

        //SVM TRAIN
        SVMModel model;

        model = SVM.svmTrain(prob, param);

        //SVM MODEL SAVE
        SVM.svmSaveModel(dir[1], model);

        //SVM MODEL LOAD
//        model = SVM.svmLoadModel(dir[1]);

//        SVM.modelInfo(model);

        //SVM PREDICT
        //test datum
        SVMProblem testX = SVM.svmReadProblem(dir[2]);

        double acc = SVM.svmPredict(model, testX, dir[3]);

//        System.out.println(acc);

    }
}

