#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define X_SIZE 10240
#define Y_SIZE 16384

#define ARRAY_SIZE (X_SIZE*Y_SIZE)

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#define TIMESTEPS 1000

const char* input_file_name = "input.dat";
const char* output_file_name = "output.dat";

void prtdat(int nx, int ny, float *current, const char *fnam);
void inidat(int nx, int ny, float *u);

void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %u or %uKB or %uMB\n", devProp.totalGlobalMem, devProp.totalGlobalMem/1024, devProp.totalGlobalMem / (1024*1024), devProp.totalGlobalMem / 1024 / 1024 / 1024);
	printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

__global__ void kernelCalculateNewGenerationWithSharedMemory(float* current, float* next, int ny, int nx) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const float cx = 0.1;
	const float cy = 0.1;

	int me = ix + iy * nx,
		east = ix + 1 + iy * nx,
		west = ix - 1 + iy * nx,
		north = ix + (iy - 1) * nx,
		south = ix + (iy + 1) * nx;

	// INIT SHARED MEMORY
	__shared__ float dev_sharedMem[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	dev_sharedMem[threadIdx.y][threadIdx.x] = current[me];

	__syncthreads();
	/* The point to update doesn't need an element that's "included" in this block */
	if ((threadIdx.x > 0) && (threadIdx.x < (BLOCK_SIZE_X - 1)) &&
		(threadIdx.y > 0) && (threadIdx.y < (BLOCK_SIZE_Y - 1))
		) {
		next[me] = cx * (dev_sharedMem[threadIdx.y][threadIdx.x-1] + dev_sharedMem[threadIdx.y][threadIdx.x+1] - 2.0f * dev_sharedMem[threadIdx.y][threadIdx.x]) +
			cy * (dev_sharedMem[threadIdx.y - 1][threadIdx.x] + dev_sharedMem[threadIdx.y + 1][threadIdx.x] - 2.0f * dev_sharedMem[threadIdx.y][threadIdx.x]) +
			dev_sharedMem[threadIdx.y][threadIdx.x];
	}
	else if (ix > 0 && ix < X_SIZE - 1 && iy > 0 && iy < Y_SIZE - 1) {
		next[me] =
			cx * (current[east] + current[west] - 2.0f * current[me]) +
			cy * (current[south] + current[north] - 2.0f * current[me]) +
			current[me];
	}
}

__global__ void kernelCalculateNewGeneration(float* current, float* next, int ny, int nx) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const float cx = 0.1;
	const float cy = 0.1;

	int me = ix + iy * nx,
		east = ix + 1 + iy * nx,
		west = ix - 1 + iy * nx,
		north = ix + (iy - 1) * nx,
		south = ix + (iy + 1) * nx;

	if (ix > 0 && ix < X_SIZE-1 && iy > 0 && iy < Y_SIZE-1) {
		next[me] =
			cx * (current[east] + current[west] - 2.0f * current[me]) +
			cy * (current[south] + current[north] - 2.0f * current[me]) +
			current[me];
	}
}

#define CEILDIV(a,b) (((a)+(b)-1)/(b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int main() {
	float *dev_heatmap, *heatmap;
	float *dev_current_map, *dev_next_map;
	int iz;

	float duration = 0;
	cudaEvent_t startEvent, endEvent;

	gpuErrchk(cudaEventCreate(&startEvent));
	gpuErrchk(cudaEventCreate(&endEvent));

	heatmap = (float*)malloc(ARRAY_SIZE*sizeof(float));

	printf("Grid is %dx%d and block is %dx%d\n", CEILDIV(X_SIZE, BLOCK_SIZE_X), CEILDIV(Y_SIZE, BLOCK_SIZE_Y), BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// KERNEL CALL PARAMETRES INIT
	dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 gridDim(CEILDIV(X_SIZE, BLOCK_SIZE_X), CEILDIV(Y_SIZE, BLOCK_SIZE_Y));

	// CPU ARRAY INITIALIZATION
	inidat(X_SIZE, Y_SIZE, heatmap);
	prtdat(X_SIZE, Y_SIZE, heatmap, input_file_name);

	// GPU INIT
	gpuErrchk(cudaSetDevice(0));
	cudaDeviceProp prop;
	gpuErrchk(cudaGetDeviceProperties(&prop, 0));

	// Init timer to count the GPU processing time 
	// GPU processing time = Moving data from host to device + main loop (processing elements) + moving data from device to host
	cudaEventRecord(startEvent);
	// GPU MEMORY INIT
	gpuErrchk(cudaMalloc(&dev_heatmap, 2 * sizeof(float)*ARRAY_SIZE))
	gpuErrchk(cudaMemcpy(dev_heatmap, heatmap, sizeof(float)*ARRAY_SIZE, cudaMemcpyHostToDevice));
	memset(heatmap, '\0', sizeof(float)*ARRAY_SIZE);

	// PRE LOOP INITIALIZATIONS
	iz = 0;
	dev_current_map = dev_heatmap;
	dev_next_map = dev_heatmap + ARRAY_SIZE;

	// MAIN LOOP
	for (int t = 0 ; t < TIMESTEPS ; t++) {
		dev_current_map = dev_heatmap + ARRAY_SIZE * iz;
		dev_next_map = dev_heatmap + ARRAY_SIZE * (1 - iz);

		// KERNEL CALL
		//kernelCalculateNewGeneration<<<blockDim,gridDim>>>(dev_current_map,dev_next_map,Y_SIZE,X_SIZE);
		kernelCalculateNewGenerationWithSharedMemory<<<blockDim,gridDim >>>(dev_current_map, dev_next_map, Y_SIZE, X_SIZE);
		iz = 1 - iz;
	}

	gpuErrchk(cudaMemcpy(heatmap, dev_next_map, sizeof(float)*ARRAY_SIZE, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaEventRecord(endEvent));
	cudaDeviceSynchronize();

	prtdat(X_SIZE, Y_SIZE, heatmap, output_file_name);
	gpuErrchk(cudaEventElapsedTime(&duration, startEvent, endEvent));
	printf("GPU elapsed time: %f\n", duration);

	return 0;
}

void inidat(int nx, int ny, float *u) {
	int ix, iy;

	for (ix = 0; ix <= nx - 1; ix++)
		for (iy = 0; iy <= ny - 1; iy++)
			*(u + ix + nx * iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

void prtdat(int nx, int ny, float *current, const char *fnam) {
	int ix, iy;
	FILE *fp;

	fp = fopen(fnam, "w");
	for (iy = 0; iy < Y_SIZE; iy++) {
		for (ix = 0; ix < nx; ix++) {
			fprintf(fp, "%6.1f", *(current + ix + nx*iy));
			if (ix != nx - 1)
				fprintf(fp, " ");
			else
				fprintf(fp, "\n");
		}
	}
	fclose(fp);
}


/*for (int t = 0; t < TIMESTEPS; t++) {
		cudaError_t cudaStatus;
		dev_current_heatmap = dev_heatmap + iz * heatmap_size;
		dev_next_heatmap	= dev_heatmap + (1-iz) * heatmap_size;
		kernelCalculateNextIteration<<<dim3BlockSizes,dim3GridSizes>>>(dev_current_heatmap, dev_next_heatmap, Y_SIZE, X_SIZE, dev_someint);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		printf("Iteration %d\n", t);
		iz = 1 - iz;
	}*/
	//cudaMemcpy(&someint, dev_someint, heatmap_size* sizeof(int), cudaMemcpyDeviceToHost);