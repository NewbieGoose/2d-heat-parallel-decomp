#ifndef __PARALLELIO_G__
#define __PARALLELIO_G__

void parallelIO_Init(
    int coords[2], /* Coordinates in Cartesian Communicator */
    int scheme[2], /* Cartesian Comm dimensions */
    int big_array_size_y,   /* Shared array */
    int big_array_size_x,   /* dimensions   */
    int local_size_y,       /* Own subarray */
    int local_size_x);      /* dimensions   */

void parallelIO_Finalize();

void parallelIO_in(
    const char* input_file, /* INPUT */
    float** array            /* OUTPUT */
    );

void parallelIO_output(
    float* array,           /* INPUT */
    const char* output_file /* INPUT */
    );
#endif