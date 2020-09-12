/*filename: mpi_egkefaliko_test.c , original file mpi_heat2D.c*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
//#include "mpi_parallelio.h"

#include "grid_sizes.h" /* Includes sizes */

#define MAX_TEMP (X_SIZE*Y_SIZE)*(X_SIZE*Y_SIZE)/8      /* bounds for the  */
#define MIN_TEMP 10                                    /* rand function   */

#define TIMESTEPS 1000     /* number of "update" iterations */

#define ERROR_CODE -666         /* label for defining errors                     */
#define OK 1                    /* label for defining that everything went OK :) */

#define NORTH 0        /* indicates        */
#define SOUTH 1        /* the coordinates  */
#define WEST 2         /* of neighbouring  */
#define EAST 3         /* processes        */

#define CONV_ERROR .05f

#define MASTER 0

/* Parametres for temperature equation */
struct Parms {
  float cx;
  float cy;
} parms = {0.1, 0.1};

MPI_Comm MPI_CART_COMM;      /*create new communicator in order to change the topology*/

/*  start of functions prototypes */
int init_array(float** array, int y_size,int x_size, int neighbors[4]);
void prtdat(int nx, int ny, float *u1, char *fnam);
void print_array(float* array, int y_size,int x_size);
void update(int start_x, int end_x, int start_y, int end_y, int y_size, float *u1, float *u2);
void update_canvas(int left_border_x, int right_border_x, int up_border_y, int down_border_y, int x_size, int neighbors[4], float *u1, float *u2);
/* end of functions prototypes */

int my_rank;

int main(void){
    int rank_size;                      /* rank info                                   */
    int local_size_x, local_size_y;     /* local dimensions                            */
    int processor_scheme[2] = {0,0};    /* coordinates of tasks over the heating table */
    int my_coords[2];
    int ix,iy;
#ifdef REDUCE_PROGRAM
    int convergence_condition[2];
#endif
    float* sub_array;

    int local_sum;
    int sum;
    int periods[2]={0,0}; /* initializing period parameter to avoid data circulation */

    int neighbors[4];                                       /* max_number of neighbors                                             */
    MPI_Request sending_requests[2][4], receiving_requests[2][4]; /* requests for non-blocking communication, both receiving and sending */
    MPI_Offset x_offset, y_offset, displacement;
    double time_start, time_end;                            /* variables to save starting and ending time                          */

    MPI_Datatype vertical_vector, vertical_vector_temp, horizontal_vector, horizontal_vector_temp; /* Vectors for sending and receiving the vertical halos */


    MPI_Init(NULL,NULL);                          /* initialize       */
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);       /* the              */
    MPI_Comm_size(MPI_COMM_WORLD,&rank_size);     /* mpi_enviroment   */

    MPI_Barrier(MPI_COMM_WORLD);    /* sychronize all processors before starting computing time needed */
    time_start = MPI_Wtime();
    MPI_Dims_create(rank_size, 2, processor_scheme);                                    /* Create the processor scheme, how they are going to be organized */
    MPI_Cart_create(MPI_COMM_WORLD, 2, processor_scheme, periods, 1, &MPI_CART_COMM);   /* Create cartesian topology for the 2D grid                       */

    if(my_rank == MASTER){
        printf("Processor scheme : %dx%d\n", processor_scheme[0],processor_scheme[1]);
	printf("Grid per processor : %dx%d\n", Y_SIZE/processor_scheme[0], X_SIZE/processor_scheme[1]);
#if  defined(PARALLELIO)
	printf("PARALLELIO\n");
#elif defined(REDUCE_PROGRAM)
	printf("REDUCE\n");
#else
	printf("SIMPLE\n");
#endif
    }

    MPI_Cart_coords(MPI_CART_COMM,my_rank,2,my_coords);

    MPI_Cart_shift(MPI_CART_COMM,0,1,&neighbors[NORTH], &neighbors[SOUTH]); /* Y axis */
    MPI_Cart_shift(MPI_CART_COMM,1,1,&neighbors[WEST], &neighbors[EAST]);   /* X axis */


    /* Decide how much load every process will get */
    local_size_y = Y_SIZE / processor_scheme[0] + ( (my_coords[0] < Y_SIZE % processor_scheme[0]) ? 1 : 0);
    local_size_x = X_SIZE / processor_scheme[1] + ( (my_coords[1] < X_SIZE % processor_scheme[1]) ? 1 : 0);

    //printf("I am task %d (y,x)=(%d,%d) and I got N: %d, S:%d, W:%d, E:%d with local_size_x %d and local_size_y %d\n",my_rank,my_coords[0],my_coords[1],neighbors[NORTH], neighbors[SOUTH], neighbors[WEST], neighbors[EAST], local_size_x, local_size_y );
#ifndef PARALLELIO
    if(init_array(&sub_array, local_size_y, local_size_x, neighbors)==ERROR_CODE) {
      MPI_Abort(MPI_COMM_WORLD, ERROR_CODE);
      exit(1);
    }
#else
    parallelIO_Init(my_coords,processor_scheme,Y_SIZE,X_SIZE,local_size_y,local_size_x);
    parallelIO_in("input.dat",&sub_array);
#endif
    /* Initializing datatypes */

    MPI_Type_vector(local_size_y, 1, local_size_x+2 , MPI_FLOAT, &vertical_vector_temp);      /* Create datatype for sending/receiving column as one entity */
    MPI_Type_create_resized(vertical_vector_temp, 0, sizeof(float), &vertical_vector);        /* Resize as one float                                        */
    MPI_Type_commit(&vertical_vector);                                                        /* Commit type                                                */

    MPI_Type_contiguous(local_size_x, MPI_FLOAT, &horizontal_vector_temp);                    /* Create datatype for sending/receiving row as one entity */
    MPI_Type_create_resized(horizontal_vector_temp, 0, sizeof(float), &horizontal_vector);    /* Resize as one float                                     */
    MPI_Type_commit(&horizontal_vector);                                                      /* Commit type                                             */

    MPI_Barrier(MPI_COMM_WORLD);
    float *current_array, *future_array;
    int time_step = 1, array_size = (local_size_x+2)*(local_size_y+2), iz = 0;

    current_array = sub_array;
    future_array = sub_array + array_size;

#define GET_OFFSET(y_pos,x_pos) ( (x_pos) + (y_pos)*(local_size_x+2) )
    MPI_Recv_init(current_array + GET_OFFSET(local_size_y+1,1), 1, horizontal_vector, neighbors[SOUTH], SOUTH, MPI_CART_COMM, &receiving_requests[0][SOUTH]);   /* receive the southern halo, practically waiting to receive  */
    MPI_Recv_init(current_array + GET_OFFSET(0,1), 1, horizontal_vector, neighbors[NORTH], NORTH, MPI_CART_COMM, &receiving_requests[0][NORTH]);                                 /* receive the northern halo, practically waiting to receive  */
    MPI_Recv_init(current_array + GET_OFFSET(1,local_size_x+1), 1, vertical_vector, neighbors[EAST], EAST, MPI_CART_COMM, &receiving_requests[0][EAST]);                      /* receive the eastern halo, practically waiting to receive   */
    MPI_Recv_init(current_array + GET_OFFSET(1,0), 1, vertical_vector, neighbors[WEST], WEST, MPI_CART_COMM, &receiving_requests[0][WEST]);                  

    MPI_Recv_init(future_array + GET_OFFSET(local_size_y+1,1), 1, horizontal_vector, neighbors[SOUTH], SOUTH, MPI_CART_COMM, &receiving_requests[1][SOUTH]);   /* receive the southern halo, practically waiting to receive  */
    MPI_Recv_init(future_array + GET_OFFSET(0,1), 1, horizontal_vector, neighbors[NORTH], NORTH, MPI_CART_COMM, &receiving_requests[1][NORTH]);                                 /* receive the northern halo, practically waiting to receive  */
    MPI_Recv_init(future_array + GET_OFFSET(1,local_size_x+1), 1, vertical_vector, neighbors[EAST], EAST, MPI_CART_COMM, &receiving_requests[1][EAST]);                      /* receive the eastern halo, practically waiting to receive   */
    MPI_Recv_init(future_array + GET_OFFSET(1,0), 1, vertical_vector, neighbors[WEST], WEST, MPI_CART_COMM, &receiving_requests[1][WEST]);                  

    MPI_Rsend_init(current_array + GET_OFFSET(1,1), 1, horizontal_vector, neighbors[NORTH], SOUTH, MPI_CART_COMM, &sending_requests[0][NORTH]);                   /* send data to the halo of the northern neighbor */
    MPI_Rsend_init(current_array + GET_OFFSET(local_size_y,1), 1, horizontal_vector, neighbors[SOUTH], NORTH, MPI_CART_COMM, &sending_requests[0][SOUTH]);  /* send data to the halo of the southern neighbor */
    MPI_Rsend_init(current_array + GET_OFFSET(1,1), 1, vertical_vector, neighbors[WEST], EAST, MPI_CART_COMM, &sending_requests[0][WEST]);                       /* send data to the halo of the western neighbor  */
    MPI_Rsend_init(current_array + GET_OFFSET(1,local_size_x), 1, vertical_vector, neighbors[EAST], WEST, MPI_CART_COMM, &sending_requests[0][EAST]);                     /* send data to the halo of the eastern neighbor  */

    MPI_Rsend_init(future_array + GET_OFFSET(1,1), 1, horizontal_vector, neighbors[NORTH], SOUTH, MPI_CART_COMM, &sending_requests[1][NORTH]);                   /* send data to the halo of the northern neighbor */
    MPI_Rsend_init(future_array + GET_OFFSET(local_size_y,1), 1, horizontal_vector, neighbors[SOUTH], NORTH, MPI_CART_COMM, &sending_requests[1][SOUTH]);  /* send data to the halo of the southern neighbor */
    MPI_Rsend_init(future_array + GET_OFFSET(1,1), 1, vertical_vector, neighbors[WEST], EAST, MPI_CART_COMM, &sending_requests[1][WEST]);                       /* send data to the halo of the western neighbor  */
    MPI_Rsend_init(future_array + GET_OFFSET(1,local_size_x), 1, vertical_vector, neighbors[EAST], WEST, MPI_CART_COMM, &sending_requests[1][EAST]);                     /* send data to the halo of the eastern neighbor  */


    for(; time_step <= TIMESTEPS; time_step++){       /* main for that updates the values of the subarrays for TIMESTEPS times */

      current_array = sub_array + iz * array_size;
      future_array  = sub_array + (1-iz) * array_size;

      
      MPI_Startall(4,receiving_requests[iz]);
      /*convesion that tag indicates the position of the receiving process refering to the sender*/
      MPI_Startall(4,sending_requests[iz]);

      /* update only the independent "white" slots of the sub_array */
      update(
        2,                            /* Left x axis bound for update  */
        local_size_x - 1,             /* Right x axis bound for update */
        2,                            /* Upper y axis bound for update */
        local_size_y - 1,             /* Lower y axis bound for update */
        local_size_x + 2, current_array, future_array);

      MPI_Waitall(4, receiving_requests[iz], MPI_STATUSES_IGNORE);

      update_canvas(
        1,
        local_size_x,
        1,
        local_size_y,
        local_size_x + 2, neighbors, current_array, future_array);



        MPI_Waitall(4, sending_requests[iz], MPI_STATUSES_IGNORE);

#ifdef REDUCE_PROGRAM
        convergence_condition[0] = 1;
        for(iy=1; iy<=local_size_y; iy++){
          for(ix=1 ; ix < local_size_x+1; ix++){
              if((*(future_array + GET_OFFSET(iy,ix)) - *(current_array + GET_OFFSET(iy,ix)) / *(current_array + GET_OFFSET(iy,ix)) ) > CONV_ERROR){
                convergence_condition[0] = 0;
                iy = local_size_y + 1;
                break;
            }
          }
          if(convergence_condition[0] == 0){
            break;
          }
        }
        MPI_Allreduce(&convergence_condition[0], &convergence_condition[1] , 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD); /* first position has the sending condition and the second has receiving condition */

        if(convergence_condition[1])
          break;
#endif
      iz = 1 - iz;           /* change status from past to current array */
    }
       

    /* free the custom defined data types */
    MPI_Type_free(&vertical_vector);
    MPI_Type_free(&horizontal_vector);

#ifdef PARALLELIO
    parallelIO_output(sub_array + iz*array_size,"output.dat");
    parallelIO_Finalize();
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    if(my_rank == rank_size-1 ) {         //why it computes the time correctly with rank_size -1 and not with master????????????
    time_end=MPI_Wtime();
    printf("total time is: %lf\n", time_end - time_start );        /*  computing and printing the total time used to calculate final result */
    }

    free(sub_array);    /* free the allocated memory of the matrix */

    MPI_Finalize();   /* End the mpi_enviroment and terminate the programm */
    return 0;
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
for (iy = ny-1; iy >= 0; iy--) {
  for (ix = 0; ix <= nx-1; ix++) {
    fprintf(fp, "%6.1f", *(u1+ix*ny+iy));
    if (ix != nx-1)
      fprintf(fp, " ");
    else
      fprintf(fp, "\n");
    }
  }
fclose(fp);
}

/*******************************************************************************
  * subroutine, that initializes the array with random nubers between the limits
  given in the "defined" sector
******************************************************************************/

int init_array(float** array, int y_size,int x_size, int neighbors[4]){
  int ix,iy;
  float *temp_array = calloc(sizeof(float) , 2*(y_size+2) * (x_size+2));
  if (temp_array==NULL){
    printf("Error, cannot allocate memory\n");
    return ERROR_CODE;
  }

  for(iy = 1; iy < y_size+1; iy++){
    if( !( neighbors[NORTH] == MPI_PROC_NULL && iy == 1 ) && !(neighbors[SOUTH] == MPI_PROC_NULL && iy == y_size) ){
      for(ix = 1; ix < x_size+1 ; ix++){
        if(  !(neighbors[WEST] == MPI_PROC_NULL && ix == 1) && !(neighbors[EAST] == MPI_PROC_NULL && ix == x_size) ){
          *(temp_array + ix + iy*(x_size+2)) = (float)(my_rank); //MIN_TEMP + rand()%(MAX_TEMP - MIN_TEMP);
        }
      }
    }
  }

  *array = temp_array;
  return OK;
}

/******************************************************************************
  *subroutine for printing subarray
*******************************************************************************/
void print_array(float* array, int y_size,int x_size){
    int ix,iy;
    for(iy=0;iy<y_size;iy++){
      for(ix=0;ix<x_size;ix++){
        printf("%6.1f ",*(array + ix + iy * x_size));
      }
      printf("\n");
    }
}

/******************************************************************************
  *  subroutine update
  ******************************************************************************/
 void update(int start_x, int end_x, int start_y, int end_y, int x_size, float *u1, float *u2)
 {
    int ix, iy;
     for(iy = start_y; iy<=end_y;iy++){
       for (ix = start_x; ix <= end_x; ix++){
         *(u2+ix+iy*x_size) = *(u1+ix+iy*x_size)  +
                           parms.cx * (*(u1+(ix+1)+iy*x_size) +
                           *(u1+(ix-1)+iy*x_size) -
                           2.0 * *(u1+ix+iy*x_size)) +
                           parms.cy * (*(u1+ix+(iy+1)*x_size) +
                          *(u1+ix+(iy-1)*x_size) -
                           2.0 * *(u1+ix+iy*x_size));
       }
     }
 }
 /******************************************************************************
   subroutine update_canvas that computes the values of the dependent elements
   of the sub_array ("green slots")
 *******************************************************************************/
 void update_canvas(int left_border_x, int right_border_x, int up_border_y, int down_border_y, int x_size, int neighbors[4], float *u1, float *u2){
   int ix,iy,end,start;
   if(neighbors[NORTH] != MPI_PROC_NULL){
     iy = up_border_y;
     start = left_border_x + ((neighbors[WEST] == MPI_PROC_NULL) ? 1 : 0);
     end = right_border_x - ((neighbors[EAST] == MPI_PROC_NULL) ? 1 : 0);
     for(ix = start ; ix <= end; ix++){       /* updating northern border of green slots */
       *(u2+ix+iy*x_size) = *(u1+ix+iy*x_size)  +
                         parms.cx * (*(u1+(ix+1)+iy*x_size) +
                         *(u1+(ix-1)+iy*x_size) -
                         2.0 * *(u1+ix+iy*x_size)) +
                         parms.cy * (*(u1+ix+(iy+1)*x_size) +
                         *(u1+ix+(iy-1)*x_size) -
                         2.0 * *(u1+ix+iy*x_size));
     }
   }
   if(neighbors[SOUTH] != MPI_PROC_NULL){
     iy = down_border_y;
     start = left_border_x + ((neighbors[WEST] == MPI_PROC_NULL) ? 1 : 0);
     end = right_border_x - ((neighbors[EAST] == MPI_PROC_NULL) ? 1 : 0);
     for(ix = start ; ix <= end; ix++){       /* updating southern border of green slots */
       *(u2+ix+iy*x_size) = *(u1+ix+iy*x_size)  +
                         parms.cx * (*(u1+(ix+1)+iy*x_size) +
                         *(u1+(ix-1)+iy*x_size) -
                         2.0 * *(u1+ix+iy*x_size)) +
                         parms.cy * (*(u1+ix+(iy+1)*x_size) +
                        *(u1+ix+(iy-1)*x_size) -
                         2.0 * *(u1+ix+iy*x_size));
     }
   }
   if(neighbors[WEST] != MPI_PROC_NULL){
     ix = left_border_x;
     start = up_border_y + ((neighbors[NORTH] == MPI_PROC_NULL) ? 1 : 0);
     end = down_border_y - ((neighbors[SOUTH] == MPI_PROC_NULL) ? 1 : 0);
     for(iy = start; iy <= down_border_y ; iy++){          /* updating western border of green slots */
       *(u2+ix+iy*x_size) = *(u1+ix+iy*x_size)  +
                         parms.cx * (*(u1+(ix+1)+iy*x_size) +
                         *(u1+(ix-1)+iy*x_size) -
                         2.0 * *(u1+ix+iy*x_size)) +
                         parms.cy * (*(u1+ix+(iy+1)*x_size) +
                        *(u1+ix+(iy-1)*x_size) -
                         2.0 * *(u1+ix+iy*x_size));
     }
   }
   if(neighbors[EAST] != MPI_PROC_NULL){
     ix = right_border_x;
     start = up_border_y + ((neighbors[NORTH] == MPI_PROC_NULL) ? 1 : 0);
     end = down_border_y - ((neighbors[SOUTH] == MPI_PROC_NULL) ? 1 : 0);
     for(iy = start; iy <= end; iy++){          /* updating eastern border of green slots */
       *(u2+ix+iy*x_size) = *(u1+ix+iy*x_size)  +
                         parms.cx * (*(u1+(ix+1)+iy*x_size) +
                         *(u1+(ix-1)+iy*x_size) -
                         2.0 * *(u1+ix+iy*x_size)) +
                         parms.cy * (*(u1+ix+(iy+1)*x_size) +
                        *(u1+ix+(iy-1)*x_size) -
                         2.0 * *(u1+ix+iy*x_size));
     }
   }
 }
