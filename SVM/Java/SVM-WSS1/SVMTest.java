package svm;
import java.io.*;
import java.util.StringTokenizer;
import java.math.BigDecimal;

public class SVMTest {
    public static void main(String[] args) throws IOException {
        String[] dir = {"E:\\JavaWorkspace\\SVMData\\mySVMTest\\a1a.txt",
                        "E:\\JavaWorkspace\\SVMData\\mySVMTest\\a1a.model",
                        "E:\\JavaWorkspace\\SVMData\\mySVMTest\\a1aT.txt",
                        "E:\\JavaWorkspace\\SVMData\\mySVMTest\\a1aT.output"};

        SVMProblem prob = SVM.svmReadProblem(dir[0]);
        SVMParameter param = new SVMParameter();
        //SVM Parameter setup
        param.kernel_type = 2;
        param.degree = 3;
        param.sigma = 0.031D;
        param.coef0 = 0.0D;
        param.eps = 1e-3D;
        param.tolerance = 0.001;
        param.C = 2.0D;

        //SVM TRAIN
        SVMModel model;

//        model = SVM.svmTrain(prob, param);

        //SVM MODEL SAVE
//        SVM.svmSaveModel(dir[1], model);

        //SVM MODEL LOAD
        model = SVM.svmLoadModel(dir[1]);

        SVM.modelInfo(model);

        //SVM PREDICT
        //test datum
        SVMProblem testX = SVM.svmReadProblem(dir[2]);

        double acc = SVM.svmPredict(model, testX, dir[3]);

    }
}
