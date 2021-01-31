
void Import_Fromfile(float *i, const char *File_name)
{
	FILE *file;
	int a=0;
	// open the file in read only mode.
	file = fopen(File_name, "r");
	if (file == NULL) {
		printf("Failed to open file\a");
	}
	
	while (fscanf(file, "%file", &i[a++]) == 1) {
		fscanf(file, ",");
	}
	
	fclose(file);
}