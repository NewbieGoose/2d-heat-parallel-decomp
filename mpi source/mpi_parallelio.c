#include "mpi_parallelio.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#define MIN(arg1,arg2) ((arg1)<(arg2) ? (arg1):(arg2))

MPI_Datatype subarray,file_view;
MPI_Offset disp;
int small_array_size_y;
int small_array_size_x;


void parallelIO_Init(
    int coords[2],
    int scheme[2], 
    int big_array_size_y, 
    int big_array_size_x,
    int local_size_y,
    int local_size_x){
    
    printf("parallel IO init\n");
    
    MPI_Datatype temp;

    small_array_size_y = local_size_y + 2;
    small_array_size_x = local_size_x + 2;
	
	/* Displacement */
	MPI_Offset y_offset = (big_array_size_y / scheme[0]) * coords[0] + MIN(coords[0],big_array_size_y % scheme[0]);
	MPI_Offset x_offset = (big_array_size_x / scheme[1]) * coords[1] + MIN(coords[1],big_array_size_x % scheme[1]);


    int small_array_sizes[2] = {small_array_size_y,small_array_size_x};
    int subarray_sizes[2] = {local_size_y,local_size_x};
    int start_coords[2] = {1,1};

    MPI_Type_create_subarray(2,small_array_sizes,subarray_sizes,start_coords,MPI_ORDER_C,MPI_FLOAT,&temp);
    MPI_Type_create_resized(temp,0,sizeof(float),&subarray);
    MPI_Type_commit(&subarray);

	MPI_Type_vector(local_size_y, local_size_x, big_array_size_x, MPI_FLOAT, &file_view);
	MPI_Type_commit(&file_view);

	disp = (big_array_size_x * y_offset + x_offset) * sizeof(float);
}

void parallelIO_Finalize(){
    MPI_Type_free(&subarray);
    MPI_Type_free(&file_view);
}

void parallelIO_in(
    const char* input_file, /* INPUT */
    float** array /* OUTPUT */
    ){
    
    MPI_File Fh;
	float *temp_array = calloc( sizeof(float) , 2 * small_array_size_x * small_array_size_y );
	MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &Fh);	/* open input file */
	MPI_File_set_view(Fh, disp, MPI_FLOAT, file_view, "native", MPI_INFO_NULL);
   	MPI_File_read(Fh, temp_array, 1 , subarray, MPI_STATUS_IGNORE);
   	MPI_File_close(&Fh);

    *array = temp_array;
}

void parallelIO_output(
    float* array /* INPUT */,
    const char* output_file /* INPUT */
    ){
        MPI_File Fh;
        printf("parallel IO out\n");

        MPI_File_open(MPI_COMM_WORLD,output_file,MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&Fh);
        MPI_File_set_view(Fh, disp, MPI_FLOAT, file_view, "native", MPI_INFO_NULL);
        MPI_File_write(Fh,array,1,subarray,MPI_STATUS_IGNORE);
        MPI_File_close(&Fh);
}