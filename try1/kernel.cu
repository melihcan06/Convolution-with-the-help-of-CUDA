#include <stdio.h>
#include <time.h> 
#include<stdlib.h>
#include<string.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include"convolutionLibHeader.h"
//#include "cudaHelperHeader.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void renklidenemepaddingyokk() {
    float f[] = { 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 };
    Mat resim = imread("C:\\Users\\user\\source\\repos\\opencv_deneme1\\1812.jpg");//deneme1.bmp	
    unsigned char* r = mat2MatrisRenkli(resim);
    int kanal = 3;
    int filtreBoyutu[] = { 3, 3 };
    int resimBoyutu[] = { resim.rows, resim.cols, kanal };
    int kaydirma[] = { 1, 1 };
    unsigned char* r2 = konvolusyonRenkli(r, f, resimBoyutu, filtreBoyutu, kaydirma, false);//true	
    Mat resim3 = matris2MatRenkli(r2, resim.rows - 2, resim.cols - 2);
    matBas(new Mat[2]{ resim, resim3 }, 2);
}

void renklidenemepaddingvarr() {
    float f[] = { 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 };
    Mat resim = imread("C:\\Users\\user\\source\\repos\\opencv_deneme1\\1812.jpg");//deneme1.bmp	
    unsigned char* r = mat2MatrisRenkli(resim);
    int kanal = 3;
    int filtreBoyutu[] = { 3, 3 };
    int resimBoyutu[] = { resim.rows, resim.cols, kanal };
    int kaydirma[] = { 1, 1 };
    unsigned char* r2 = konvolusyonRenkli(r, f, resimBoyutu, filtreBoyutu, kaydirma, true);//true	
    Mat resim3 = matris2MatRenkli(r2, resim.rows, resim.cols);
    matBas(new Mat[2]{ resim, resim3 }, 2);
}

int main() {
    renklidenemepaddingvarr();
    return 0;
}
