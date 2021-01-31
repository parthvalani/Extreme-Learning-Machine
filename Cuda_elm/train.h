__global__ void matrixRandomBalance(float *a)
{
	unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	a[x] = a[x]*2.0 -1.0;
}

__global__ void matrixSigmoid(float *a)
{
	unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;
	a[x] = tanh(a[x]);
}
void Train_elm(float *input, float *lbl, float *Mat_f, float*beta, unsigned long ip_num, unsigned long hid_num, unsigned long op_num, unsigned long training_samples)
{
	//creating handle 
	cublasHandle_t handle;
    cublasCreate(&handle);
	const float Alpha = 1;
	const float Beta = 0;
	
	//Host pointers and allocate memory to them
	float *h_H = (float *)malloc(training_samples 	* hid_num	*sizeof(float));
	float *h_H_pinv = (float *)malloc(hid_num 	* training_samples		*sizeof(float));
	
	// defining device pointers and allocate device memory
	float *dev_ip, *dev_Mat_f, *dev_H, *dev_H_pinv, *dev_op, *dev_beta ;
	cudaMalloc((void **)&dev_ip, training_samples* ip_num* sizeof(float));
	cudaMalloc((void **)&dev_Mat_f, ip_num * hid_num* sizeof(float));
	cudaMalloc((void **)&dev_H, training_samples* hid_num* sizeof(float));
	cudaMalloc((void **)&dev_H_pinv, hid_num* training_samples* sizeof(float));
	cudaMalloc((void **)&dev_op, training_samples* op_num* sizeof(float));
	cudaMalloc((void **)&dev_beta, hid_num* op_num* sizeof(float));
	
	//generating random weights values. 
	Random_Matrix(Mat_f,ip_num,hid_num);
	
	////////////////////  Training for ELM training has started ................
	cudaMemcpy(dev_ip, input, training_samples* ip_num* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Mat_f, Mat_f	, ip_num * hid_num	* sizeof(float), cudaMemcpyHostToDevice);
	matrixRandomBalance<<<4,1000>>>(dev_Mat_f);
	cudaMemcpy(Mat_f, dev_Mat_f	, ip_num * hid_num	* sizeof(float), cudaMemcpyDeviceToHost);
	
	// calculate I multiply with Mat_f
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, training_samples, hid_num, ip_num, &Alpha, dev_ip, training_samples, dev_Mat_f, ip_num, &Beta, dev_H, training_samples);
	
	// storing the result from device to host(h_H)
	cudaMemcpy(h_H, dev_H, training_samples *hid_num * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Use of activation function.
	matrixSigmoid<<<150,1000>>>(dev_H);
	cudaMemcpy(h_H, dev_H, training_samples *hid_num * sizeof(float), cudaMemcpyDeviceToHost);
	
	//calling inverse function for h_H
	matrix_pinv(h_H,h_H_pinv,training_samples,hid_num);
	
	//tranferring H and Labels to device for beta count 
	cudaMemcpy(dev_H_pinv, h_H_pinv	, hid_num * training_samples* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_op, lbl, training_samples * op_num* sizeof(float), cudaMemcpyHostToDevice);
	
	// calculate B = Hi O and copy back results to B
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hid_num, op_num, training_samples, &Alpha, dev_H_pinv, hid_num, dev_op, training_samples, &Beta, dev_beta, hid_num);
	cudaMemcpy(beta, dev_beta, hid_num *op_num * sizeof(float), cudaMemcpyDeviceToHost);
	
	//destroy handle and fee rest of the memory
	cublasDestroy(handle);
	cudaFree(dev_beta);
	cudaFree(dev_Mat_f);
	cudaFree(dev_ip);
	cudaFree(dev_H);
	cudaFree(dev_H_pinv);
	cudaFree(dev_op);
	
	free(h_H);
	free(h_H_pinv);

}
