int min(float *array,int num)
{
    int c, location = 0;
	for (c = 0; c < num; c++){
		if (array[c] < array[location]){
		  location = c;
		}
	}
	//printf("%d \n", location);
	return location;
}

//Function to display matrix in proper formate
void Display_Matrix(float *a, float *b, unsigned long m, unsigned long n)
{
	float f[100],c[100],d=1.0;
	unsigned long sum=0; 
	float v = m;
	for(unsigned long i=0;i<m;i++)
	{
		for(unsigned long j=0;j<n;j++)
		{
			f[j] = fabs(d-a[i+j*m]);
			c[j] = fabs(d-b[i+j*m]);
		}
		if (min(f,n)==min(c,n)){
			sum = sum+1;
		}
	}
	printf("Testing accuracy is : %f\n",(sum/v)*100);
}




	
		
		