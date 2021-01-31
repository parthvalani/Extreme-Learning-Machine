// for generating random weights for elm.
void Random_Matrix(float *Mat, unsigned long dim_x, unsigned long dim_y)
{
	// Create a pseudo-random number generator
	curandGenerator_t rg;
	curandCreateGenerator(&rg, CURAND_RNG_PSEUDO_DEFAULT);
	// defining the variable.
	unsigned long size = dim_x * dim_y;
	
	//define device pointer and allocate memory
	float *dev_Mat;
	cudaMalloc((void **)&dev_Mat, size * sizeof(float));

	curandSetPseudoRandomGeneratorSeed(rg, (unsigned long long) clock());
	curandGenerateUniform(rg, dev_Mat, size); //Getting randomly generated data for weights matrix in device memory
	cudaMemcpy(Mat, dev_Mat, size * sizeof(float), cudaMemcpyDeviceToHost); //Copy back assigned numbers to host
	
	//Clear memory and destroy generator
	cudaFree(dev_Mat);
	curandDestroyGenerator(rg);
}