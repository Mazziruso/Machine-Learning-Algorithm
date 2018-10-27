import sys
sys.path.append("E:\\Python workspace\\MachineLearning\\libsvm-master\\python")
sys.path.append("E:\\Python workspace\\MachineLearning\\SVM")

import matplotlib.pyplot as plt
import numpy as np

from svm import *
from svmutil import *

import gridSearch

#Construct Data Structure
train_labels, train_values = svm_read_problem("E:\\Python workspace\\MachineLearning\\LIBSVMData\\A1A\\a1a")
predict_labels, predict_values = svm_read_problem("E:\\Python workspace\\MachineLearning\\LIBSVMData\\A1A\\a1a.t")

#GridSearch
options = "-log2c -5,5,1 -svmtrain \"E:\\Python workspace\\MachineLearning\\libsvm-master\\windows\\svm-train.exe\" -gnuplot \"C:\\GNUPlot\\gnuplot\\bin\\gnuplot.exe\" -v 10"
gridSearch.find_parameters("E:\\Python workspace\\MachineLearning\\LIBSVMData\\A1A\\a1a", options)

##SVM Train and Test
acc = [0]*11
i = 0
for C in range(20, 41, 2):
	C /= 10.0
	options = '-s 0 -t 2 -g 0.031 -c ' + str(C)
	n = 20

	acc[i] = 0
	while n>0:
		model = svm_train(train_labels, train_values, options)

# svm_save_model("E:\\Python workspace\\MachineLearning\\LIBSVMData\\A1A\\a1a.model", model)

		output_labels, output_acc, output_values = svm_predict(predict_labels, predict_values, model)
		acc[i] += output_acc[0]
		n -= 1
	acc[i] /= 20
	i += 1

gamma = [i/10.0 for i in range(20, 41, 2)]
print(acc)

x = np.array(gamma)
y = np.array(acc)

plt.figure()
plt.stem(x, y)
plt.show()

acc, mse, scc = evaluations(predict_labels, output_labels)

print("result: ")
print(output_acc)
print(acc)
print(mse)
print(scc)
