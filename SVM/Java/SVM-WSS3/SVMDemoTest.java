package SVM;

import java.io.*;

/*
Require: label needs -1 or +1
 */

public class SVMDemoTest {
    public static void main(String[] args) throws IOException {
        String[] dir = {"SVMData\\mySVMTestWSS3\\a1a.txt", //训练数据集
                "SVMData\\mySVMTestWSS3\\a1a.model",   //存放SVM模型
                "SVMData\\mySVMTestWSS3\\a1aT.txt",        //测试数据集
                "SVMData\\mySVMTestWSS3\\a1aT.output"};//测试集label输出

        String matdir = "E:\\JavaWorkspace\\SVMData\\mySVMTest_delta\\a1aVal.txt";

        SVMProblem prob = SVM.svmReadProblem(dir[0]);

        SVMParameter param = new SVMParameter();
        //SVM Parameter setup
        param.kernel_type = 2;
        param.degree = 3;
        param.sigma = 1.0/prob.d;
        param.coef0 = 0.0D;
        param.eps = 1e-3D;
        param.C = 2.0D;
        param.tao = 1e-12;

        //SVM TRAIN
        SVMModel model;

        model = SVM.svmTrain(prob, param, matdir);

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


