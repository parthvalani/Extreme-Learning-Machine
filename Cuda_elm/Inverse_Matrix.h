void Inverse_Matrix(float *matrix, float *Inv_matrix, unsigned long dim)
{
	
	cublasHandle_t handle;
    cublasCreate(&handle);
	
	// defining host pointers for input matrix, output matrix and allocating memory for them
	float *f_outcome = (float *)malloc(dim*dim*sizeof(float));
    float **outcome = (float **)malloc(sizeof(float *));
    float **ip_matrix = (float **)malloc(sizeof(float *));
	outcome[0] = f_outcome;
    ip_matrix[0]  = matrix;
	
	// defining pivot number and allocating memory
	int *p_num, *temp;
	cudaMalloc(&p_num, dim * 1 * sizeof(int));
	cudaMalloc(&temp, 1 * sizeof(int));
	
	//Assign memories to main input matrix and resulting Inverse matrix
	float **h_RF = (float **)malloc(1*sizeof(float *));
    float **dev_RF, *dev_RF_f;
    cudaMalloc(&dev_RF,1*sizeof(float *));
    cudaMalloc(&dev_RF_f, dim*dim*1*sizeof(float));
    h_RF[0] = dev_RF_f;
	cudaMemcpy(dev_RF,h_RF,1*sizeof(float *),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_RF_f, ip_matrix[0], dim*dim*sizeof(float), cudaMemcpyHostToDevice);
	
	// finding pivot num matrix
	cublasSgetrfBatched(handle,dim,dev_RF,dim,p_num,temp,1);
	
	//Again assign memories to main input matrix and resulting Invrse matrix
	float **h_RI = (float **)malloc(1*sizeof(float *));
    float **dev_RI, *dev_RI_f;
    cudaMalloc(&dev_RI,1*sizeof(float *));
    cudaMalloc(&dev_RI_f, dim*dim*1*sizeof(float));
    h_RI[0] = dev_RI_f;
	cudaMemcpy(dev_RI,h_RI,1*sizeof(float *),cudaMemcpyHostToDevice);
	
	//inverse from pivot num matrix
    cublasSgetriBatched(handle,dim,(const float **)dev_RF,dim,p_num,dev_RI,dim,temp,1);
	
	//Copy outcome to Inverse matrix
	cudaMemcpy(Inv_matrix, dev_RI_f, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);
	
	//Destroy handle and free memories
	free(f_outcome);
	free(outcome);
	free(ip_matrix);
	free(h_RF);
	free(h_RI);
	
	cublasDestroy(handle);
	cudaFree(p_num);
	cudaFree(temp);
	cudaFree(dev_RF);
	cudaFree(dev_RF_f);
	cudaFree(dev_RI);
	cudaFree(dev_RI_f);
	
}