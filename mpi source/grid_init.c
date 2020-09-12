#include "grid_init.h"

#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <stdlib.h>

#include "grid_sizes.h" /* Includes sizes */


void serial_init(const char* file);
void heat_init(const char* file);
void random_init(const char* file, int max);

void grid_init(const char* file, enum init_choice choice){
    switch (choice)
    {
        case SERIAL:
            serial_init(file);
            break;
        case RANDOM:
            random_init(file,2000);
            break;
        case HEAT:
            heat_init(file);
            break;
    }
}

void heat_init(const char* file){
    int ix, iy, fd;
    fd = open(file,O_WRONLY | O_CREAT, 00777);
    for (iy = 0; iy <= Y_SIZE-1; iy++) 
        for (ix = 0; ix <= X_SIZE-1; ix++){
            float temp = (float)(ix * (X_SIZE - ix - 1) * iy * (Y_SIZE - iy - 1));
            write(fd,&temp,sizeof(float));
        }
}

void serial_init(const char* file){
    float count = 100.0f;
    float zero = .0f;
    int x,y,fd;
    fd = open(file,O_WRONLY | O_CREAT, 00777);
    for( y = 0; y < Y_SIZE ; y++){
        for( x = 0; x < X_SIZE ; x++){
            if(x == 0 || x == X_SIZE-1 || y == 0 || y == Y_SIZE-1)
                write(fd,&zero,sizeof(float));
            else{
                write(fd,&count,sizeof(float));
                count++;
            }
        }
    }
}

void random_init(const char* file, int max){
    srand(time(NULL));

    float random_float = .0f;
    float zero = .0f;
    int x,y,fd;
    fd = open(file,O_WRONLY | O_CREAT, 00777);
    for( y = 0; y < Y_SIZE ; y++){
        for( x = 0; x < X_SIZE ; x++){
            if(x == 0 || x == X_SIZE-1 || y == 0 || y == Y_SIZE-1)
                write(fd,&zero,sizeof(float));
            else{
                random_float = rand()%max;
                write(fd,&random_float,sizeof(float));
            }
        }
    }
}

int main(int argc, char* argv[]){
    enum init_choice choice;
    if(argc < 0)
        choice = RANDOM;
    else{
        if(strcmp(argv[1],"-s") == 0)
            choice = SERIAL;
        else if(strcmp(argv[1],"-r") == 0)
            choice = RANDOM;
        else if(strcmp(argv[1],"-t") == 0)
            choice = HEAT;
        else
            choice = RANDOM;
    }
    grid_init("input.dat",choice);
    return 0;
}
