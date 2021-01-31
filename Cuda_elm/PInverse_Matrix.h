void matrix_pinv(float *mat, float *inv_mat, unsigned long dim_x, unsigned long dim_y)
{
	
	//initiale cublas handles and other stuff
	cublasHandle_t handle;
    cublasCreate(&handle);
	const float Alpha = 1;
	const float Beta = 0;
	//Initialize pointers for arrays and allocate host memory for above pointers
	float *T_mat = (float *)malloc(dim_y*dim_y*sizeof(float));;
	float *T_mat_inv = (float *)malloc(dim_y*dim_y*sizeof(float));
	
	//Initialize device pointers and allocate device memory to array
	float *dev_mat, *dev_T_mat,*dev_T_mat_inv,*dev_mat_inv;
	cudaMalloc((void **)&dev_mat		, dim_x * dim_y * sizeof(float));
	cudaMalloc((void **)&dev_T_mat		, dim_y * dim_y * sizeof(float));
	cudaMalloc((void **)&dev_T_mat_inv	, dim_y * dim_y * sizeof(float));
	cudaMalloc((void **)&dev_mat_inv	, dim_y * dim_x * sizeof(float));
	/////////////////   A'A
	
	cudaMemcpy(dev_mat, mat, dim_x * dim_y * sizeof(float), cudaMemcpyHostToDevice); // copy memory from host(mat) to device(dev_mat)
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim_y, dim_y, dim_x, &Alpha, dev_mat, dim_x, dev_mat, dim_x, &Beta, dev_T_mat, dim_y);  //matrix multiplication
	cudaMemcpy(T_mat, dev_T_mat, dim_y * dim_y * sizeof(float), cudaMemcpyDeviceToHost); //copy back from device to host and save result to T_mat
	//////////////// (A'A)inv
	Inverse_Matrix(T_mat,T_mat_inv,dim_y); // calling the function of matrix multiplication 
	cudaMemcpy(dev_T_mat_inv, T_mat_inv, dim_y * dim_y * sizeof(float), cudaMemcpyHostToDevice); // copy the result from host(T_mat_inv) to device(dev_T_mat_inv)
	/////////////// (A'A)inv A'
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_y, dim_x, dim_y, &Alpha, dev_T_mat_inv, dim_y, dev_mat, dim_x, &Beta, dev_mat_inv, dim_y); // matrix multiplication
	cudaMemcpy(inv_mat, dev_mat_inv, dim_y * dim_x * sizeof(float), cudaMemcpyDeviceToHost); // copy back from device to host and save result to inv_mat
	
	//destroy handle and fee rest of the memory
	cublasDestroy(handle);
	cudaFree(dev_mat);
	cudaFree(dev_mat_inv);
	cudaFree(dev_T_mat);
	cudaFree(dev_T_mat_inv);
	free(T_mat);
	free(T_mat_inv);
}

