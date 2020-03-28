#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h> 
#include<stdlib.h>
#include<string.h>
#include <iostream>

__global__ void carpCuda(unsigned char* resimBolgesi, float* filtre, unsigned char* donecekBolge, int boy, int en);
int carpTopla(unsigned char* resimBolgesi, float* filtre, int boy, int en);