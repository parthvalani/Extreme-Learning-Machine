//Libraries for cuda runtime
#include <cublas_v2.h>
#include <curand.h>

//Standard C libraries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Display_Matrix.h"
#include "Random_Matrix.h"
#include "Inverse_Matrix.h"
#include "PInverse_Matrix.h"
#include "Load_Matrix.h"
#include "train.h"
#include "test.h"

//Main function
int main()
{	
	
	float *Mat_f;
	float *beta;
	float *op_matrix;
	unsigned long training_samples = 50000;
	unsigned long testing_samples = 10000;
	unsigned long ip_num  = 64;
	unsigned long op_num = 10;
	unsigned long hid_num = 20;
	Mat_f= (float *)malloc(ip_num* hid_num* sizeof(float));
	beta= (float *)malloc(hid_num* op_num* sizeof(float));
	op_matrix= (float *)malloc(testing_samples* op_num* sizeof(float));
	 
	
	float *X_Train = (float *)malloc(training_samples	* ip_num	* sizeof(float));
	float *Y_Train = (float *)malloc(training_samples	* op_num	* sizeof(float));
	float *X_Test = (float *)malloc(testing_samples 	* ip_num	* sizeof(float));
	float *Y_Test = (float *)malloc(testing_samples	* op_num	* sizeof(float));
	
	Import_Fromfile(X_Train,"features_cifar10/train_features.csv");
	Import_Fromfile(Y_Train,"features_cifar10/train_labels.csv");
	Import_Fromfile(X_Test,"features_cifar10/test_features.csv");
	Import_Fromfile(Y_Test,"features_cifar10/test_labels.csv");
	

	//// Calling a training function of ELM
	Train_elm(X_Train,Y_Train,Mat_f,beta,ip_num,hid_num,op_num,training_samples);
	
	//// Calling a testing function of ELM
	Test_elm(X_Test,Y_Test,Mat_f,beta,op_matrix,ip_num,hid_num,op_num,testing_samples);
	
	/// Output Matrix and Accuracy
	Display_Matrix(op_matrix,Y_Test,testing_samples,op_num);
	printf("\n");
	
	return 0;
}