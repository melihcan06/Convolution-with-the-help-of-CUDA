#include <stdio.h>
#include <time.h> 
#include<stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<string.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaHelperHeader.cuh"

using namespace std;
using namespace cv;

void matBas(Mat m[], int boy);
void matBas2(Mat m[], int boy, char* yazi);
unsigned char* mat2MatrisRenkli(Mat resim);
Mat matris2MatRenkli(unsigned char* resim, int boy, int en);
unsigned char* mat2MatrisGri(Mat resim);
Mat matris2MatGri(unsigned char* resim, int boy, int en);
void griBas(unsigned char* resim2, int boy, int en);
void renkliBas(unsigned char* r, int boy, int en);
unsigned char* resmeZeroPadding(unsigned char* resim, int* resimBoyutu, int* padding);
int* esitlikIcýnPaddingHesabi(int* resimBoyutu, int* filtreBoyutu, int* kaydirma);
int* yeniBoyutHesabi(int* resimBoyutu, int* filtreBoyutu, int* kaydirma, int* padding);
int carpTopla2(unsigned char* resimBolgesi, float* filtre, int boy, int en);
unsigned char* konvolusyonGri(unsigned char* resim, float* filtre, int* resimBoyutu, int* filtreBoyutu, int* kaydirma, bool boyutAyniKalsin);
unsigned char* konvolusyonRenkli(unsigned char* resim, float* filtre2, int* resimBoyutu, int* filtreBoyutu, int* kaydirma, bool boyutAyniKalsin);