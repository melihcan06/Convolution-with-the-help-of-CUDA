#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h> 
#include<stdlib.h>
#include<string.h>
#include <iostream>

#include "cudaHelperHeader.cuh"

__global__ void carpCuda(unsigned char* resimBolgesi, float* filtre, unsigned char* donecekBolge, int boy, int en) {
	int b = threadIdx.x;
	int e = threadIdx.y;
	int x = (int)*(resimBolgesi + b * en + e);
	float y = *(filtre + b * en + e);
	float z = (float)x * y;
	*(donecekBolge + b * en + e) = z;
}

int carpTopla(unsigned char* resimBolgesi, float* filtre, int boy, int en) {

	unsigned char* donecekBolge = (unsigned char*)malloc(sizeof(unsigned char) * boy * en);

	unsigned char* gpu_rb = (unsigned char*)malloc(sizeof(unsigned char) * boy * en);
	float* gpu_f = (float*)malloc(sizeof(float) * boy * en);
	unsigned char* gpu_db = (unsigned char*)malloc(sizeof(unsigned char) * boy * en);

	cudaMalloc(&gpu_rb, sizeof(unsigned char) * boy * en);
	cudaMalloc(&gpu_f, sizeof(float) * boy * en);
	cudaMalloc(&gpu_db, sizeof(unsigned char) * boy * en);

	cudaMemcpy(gpu_rb, resimBolgesi, boy * en * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_f, filtre, boy * en * sizeof(float), cudaMemcpyHostToDevice);

	int numBlocks = 1;
	dim3 threadsPerBlock(boy, en);
	carpCuda << < numBlocks, threadsPerBlock >> > (gpu_rb, gpu_f, gpu_db, boy, en);

	cudaMemcpy(donecekBolge, gpu_db, boy * en * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(gpu_rb);
	cudaFree(gpu_f);
	cudaFree(gpu_db);

	int toplam = 0;
	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			toplam += (int)*(donecekBolge + i * en + j);
		}
	}

	return toplam;
}