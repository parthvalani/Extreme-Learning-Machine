void Test_elm(float *input, float *lbl, float *Mat_f, float *beta, float *op_matrix, unsigned long ip_num, unsigned long hid_num, unsigned long op_num, unsigned long testing_samples)
{
	//Matrix mul
	cublasHandle_t handle;
    cublasCreate(&handle);
	const float alf = 1;
	const float bet = 0;
	
	//Device pointers
	float *dev_ip, *dev_Mat_f, *dev_H, *dev_op, *dev_beta ;
	
	//Allocate device memory
	cudaMalloc((void **)&dev_ip, testing_samples * ip_num* sizeof(float));
	cudaMalloc((void **)&dev_Mat_f, ip_num* hid_num	* sizeof(float));
	cudaMalloc((void **)&dev_H, testing_samples	* hid_num* sizeof(float));
	cudaMalloc((void **)&dev_op, testing_samples* op_num* sizeof(float));
	cudaMalloc((void **)&dev_beta, hid_num* op_num* sizeof(float));
	
	//////// Testing of ELm has started..........
	cudaMemcpy(dev_ip, input, testing_samples * ip_num	* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Mat_f, Mat_f	, ip_num * hid_num	* sizeof(float), cudaMemcpyHostToDevice);
	
	// calculate I multiply with Mat_f
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, testing_samples, hid_num, ip_num, &alf, dev_ip, testing_samples, dev_Mat_f, ip_num, &bet, dev_H, testing_samples);
	
	//using activation function
	matrixSigmoid<<<150,1000>>>(dev_H);
	cudaMemcpy(dev_beta	, beta, hid_num	* op_num* sizeof(float), cudaMemcpyHostToDevice);
	
	// calculate H X B
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, testing_samples, op_num, hid_num, &alf, dev_H, testing_samples, dev_beta, hid_num, &bet, dev_op, testing_samples);
	cudaMemcpy(op_matrix, dev_op, testing_samples * op_num * sizeof(float), cudaMemcpyDeviceToHost); // copy back result from device to host
	
	//destroy handle and fee rest of the memory
	cublasDestroy(handle);
	cudaFree(dev_H);
	cudaFree(dev_op);
	cudaFree(dev_beta);
	cudaFree(dev_ip);
	cudaFree(dev_Mat_f);

}
	