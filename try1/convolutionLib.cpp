#include <stdio.h>
#include <time.h> 
#include<stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<string.h>
#include <iostream>

#include"convolutionLibHeader.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

void matBas(Mat m[], int boy) {
	for (int i = 0; i < boy; i++) {
		imshow(to_string(i + 1), m[i]);
	}
	waitKey(0);
}

void matBas2(Mat m[], int boy, char* yazi) {
	for (int i = 0; i < boy; i++) {
		imshow(yazi + to_string(i + 1), m[i]);
	}
	waitKey(0);
}

unsigned char* mat2MatrisRenkli(Mat resim) {
	int boy = resim.rows;
	int en = resim.cols;
	int kanal = resim.channels();
	unsigned char* resim2 = (unsigned char*)malloc(sizeof(unsigned char) * en * boy * kanal);

	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			for (int k = 0; k < kanal; k++) {
				*(resim2 + (i * en * kanal + j * kanal + k)) = resim.at<Vec3b>(Point(j, i))[k];
			}
		}
	}

	return resim2;
}

Mat matris2MatRenkli(unsigned char* resim, int boy, int en) {
	Mat resim2 = Mat(boy, en, CV_8UC3);
	int kanal = 3;

	Vec3b renk;
	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			for (int k = 0; k < kanal; k++)
				renk[k] = *(resim + (i * en * kanal + j * kanal + k));
			resim2.at<Vec3b>(Point(j, i)) = renk;
		}
	}
	return resim2;
}
unsigned char* mat2MatrisGri(Mat resim) {
	int boy = resim.rows;
	int en = resim.cols;
	unsigned char* resim2 = (unsigned char*)malloc(sizeof(unsigned char) * en * boy);

	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			*(resim2 + i * en + j) = resim.at<uchar>(Point(j, i));
		}
	}

	return resim2;
}

Mat matris2MatGri(unsigned char* resim, int boy, int en) {
	Mat resim2 = Mat(boy, en, CV_8UC1);

	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			resim2.at<uchar>(Point(j, i)) = *(resim + i * en + j);
		}
	}
	return resim2;
}

void griBas(unsigned char* resim2, int boy, int en) {
	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			printf("%d ", *(resim2 + i * en + j));
		}
		printf("\n");
	}
}

void renkliBas(unsigned char* r, int boy, int en) {
	int kanal = 3;
	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			for (int k = 0; k < kanal; k++)
				printf("%d ", *(r + (i * en * kanal + j * kanal + k)));
			printf("   ");
		}
		printf("\n");
	}
}

unsigned char* resmeZeroPadding(unsigned char* resim, int* resimBoyutu, int* padding) {
	int yeniEn = resimBoyutu[1] + 2 * padding[1];
	int yeniBoy = resimBoyutu[0] + 2 * padding[0];
	int paddingBoy = padding[0];
	int paddingEn = padding[1];
	unsigned char* yeni = (unsigned char*)malloc(sizeof(unsigned char) * yeniEn * yeniBoy);

	int resimBoyIndex = 0;
	int resimEnIndex = 0;
	for (int i = paddingBoy; i < yeniBoy - paddingBoy; i++) {
		resimEnIndex = 0;
		for (int j = paddingEn; j < yeniEn - paddingEn; j++) {
			*(yeni + i * yeniEn + j) = *(resim + resimBoyIndex * resimBoyutu[1] + resimEnIndex);
			resimEnIndex++;
		}
		resimBoyIndex++;
	}

	for (int i = 0; i < yeniBoy; i++) {
		for (int j = 0; j < paddingEn; j++) {
			*(yeni + i * yeniEn + j) = 0;
		}
	}
	for (int i = 0; i < yeniBoy; i++) {
		for (int j = yeniEn - paddingEn; j < yeniEn; j++) {
			*(yeni + i * yeniEn + j) = 0;
		}
	}
	for (int i = 0; i < paddingBoy; i++) {
		for (int j = 0; j < yeniEn; j++) {
			*(yeni + i * yeniEn + j) = 0;
		}
	}
	for (int i = yeniBoy - paddingBoy; i < yeniBoy; i++) {
		for (int j = 0; j < yeniEn; j++) {
			*(yeni + i * yeniEn + j) = 0;
		}
	}
	return yeni;
}

int* esitlikIcýnPaddingHesabi(int* resimBoyutu, int* filtreBoyutu, int* kaydirma) {
	int* yeni = (int*)malloc(sizeof(int) * 2);//yeni[0]=boy,yeni[1]=en	
	yeni[0] = (((resimBoyutu[0] - 1) * kaydirma[0]) + filtreBoyutu[0] - resimBoyutu[0]) / 2;
	yeni[1] = (((resimBoyutu[1] - 1) * kaydirma[1]) + filtreBoyutu[1] - resimBoyutu[1]) / 2;
	return yeni;
}

int* yeniBoyutHesabi(int* resimBoyutu, int* filtreBoyutu, int* kaydirma, int* padding) {
	int* yeni = (int*)malloc(sizeof(int) * 2);//yeni[0]=boy,yeni[1]=en	
	yeni[0] = ((resimBoyutu[0] - filtreBoyutu[0] + 2 * padding[0]) / kaydirma[0]) + 1;
	yeni[1] = ((resimBoyutu[1] - filtreBoyutu[1] + 2 * padding[1]) / kaydirma[1]) + 1;
	return yeni;
}

int carpTopla2(unsigned char* resimBolgesi, float* filtre, int boy, int en) {// cuda kernel kullan	
	//griBas(resimBolgesi,3,3);
	float yeni = 0;
	int x;
	float y;
	for (int b = 0; b < boy; b++) {
		for (int e = 0; e < en; e++) {
			x = (int)*(resimBolgesi + b * en + e);
			y = (float)*(filtre + b * en + e);
			yeni += (float)x * y;
		}
	}
	int yeni2 = (int)yeni;
	return yeni2;
}

unsigned char* konvolusyonGri(unsigned char* resim, float* filtre, int* resimBoyutu, int* filtreBoyutu, int* kaydirma, bool boyutAyniKalsin) {
	//kaydirmayi kullan!!!
	//matBas2(new Mat[1]{ matris2MatGri(resim,resimBoyutu[0],resimBoyutu[1]) }, 1,new char[4] {'g','e','l','\0'});
	int* padding;
	int* yeniBoyut = (int*)malloc(sizeof(int) * 2);
	unsigned char* yeni;
	int paddingBoy;
	int paddingEn;
	int yeniEn;
	int yeniBoy;
	unsigned char* carpilacakMatris;
	int kanal = 1;//resimBoyutu[2];	

	if (boyutAyniKalsin) {
		padding = esitlikIcýnPaddingHesabi(resimBoyutu, filtreBoyutu, kaydirma);
		yeniBoyut[0] = resimBoyutu[0];//boy
		yeniBoyut[1] = resimBoyutu[1];
		yeniEn = yeniBoyut[1];
		yeniBoy = yeniBoyut[0];
		yeni = (unsigned char*)malloc(sizeof(unsigned char) * yeniEn * yeniBoy);
		resim = resmeZeroPadding(resim, yeniBoyut, padding);
		resimBoyutu[0] = yeniBoyut[0] + padding[0] * 2;
		resimBoyutu[1] = yeniBoyut[1] + padding[1] * 2;
	}
	else {
		padding = (int*)malloc(sizeof(int) * 2);
		padding[0] = 0;
		padding[1] = 0;
		yeniBoyut = yeniBoyutHesabi(resimBoyutu, filtreBoyutu, kaydirma, padding);
		yeniEn = yeniBoyut[1];
		yeniBoy = yeniBoyut[0];
		yeni = (unsigned char*)malloc(sizeof(unsigned char) * yeniEn * yeniBoy);
	}

	int resimBoy = resimBoyutu[0];
	int resimEn = resimBoyutu[1];
	int baslangicBoy = filtreBoyutu[0] / 2;
	int baslangicEn = filtreBoyutu[1] / 2;
	int bitisBoy = resimBoy - baslangicBoy;
	int bitisEn = resimEn - baslangicEn;
	int filtreBoy = filtreBoyutu[0];
	int filtreEn = filtreBoyutu[1];
	carpilacakMatris = (unsigned char*)malloc(sizeof(unsigned char) * filtreBoy * filtreEn);

	int yeniE = 0;
	int yeniB = 0;
	int carpilacakE = 0;
	int carpilacakB = 0;

	for (int b = baslangicBoy; b < bitisBoy; b++) {
		yeniE = 0;
		for (int e = baslangicEn; e < bitisEn; e++) {
			carpilacakB = 0;
			//carpilip toplanacak matrisi resimden aliyor
			for (int i = b - baslangicBoy; i < b + baslangicBoy + 1; i++) {//ic donguyu cuda ile paralel olarak al 
				carpilacakE = 0;
				for (int j = e - baslangicEn; j < e + baslangicEn + 1; j++) {
					*(carpilacakMatris + carpilacakB * filtreEn + carpilacakE) = *(resim + i * resimEn + j);
					carpilacakE++;
				}
				carpilacakB++;
			}
			*(yeni + yeniB * yeniEn + yeniE) = carpTopla(carpilacakMatris, filtre, filtreBoy, filtreEn);
			yeniE++;
		}
		yeniB++;
	}
	//matBas2(new Mat[1]{ matris2MatGri(yeni,yeniBoyut[0],yeniBoyut[1]) }, 1, new char[4]{ 'g','i','t' ,'\0' });
	return yeni;
}

unsigned char* konvolusyonRenkli(unsigned char* resim, float* filtre2, int* resimBoyutu, int* filtreBoyutu, int* kaydirma, bool boyutAyniKalsin) {
	int boy = resimBoyutu[0];
	int en = resimBoyutu[1];
	int rsmByt[3] = { boy,en,1 };
	float filtre[9] = { 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 };


	unsigned char* b = (unsigned char*)malloc(sizeof(unsigned char) * boy * en);
	unsigned char* g = (unsigned char*)malloc(sizeof(unsigned char) * boy * en);
	unsigned char* r = (unsigned char*)malloc(sizeof(unsigned char) * boy * en);
	unsigned char* yeni = (unsigned char*)malloc(sizeof(unsigned char) * boy * en * 3);

	//3 kanala ayristirma
	for (int i = 0; i < boy; i++) {
		for (int j = 0; j < en; j++) {
			*(b + i * en + j) = *(resim + i * en * 3 + j * 3 + 0);
			*(g + i * en + j) = *(resim + i * en * 3 + j * 3 + 1);
			*(r + i * en + j) = *(resim + i * en * 3 + j * 3 + 2);
		}
	}

	b = konvolusyonGri(b, filtre, rsmByt, filtreBoyutu, kaydirma, boyutAyniKalsin);
	g = konvolusyonGri(g, filtre, rsmByt, filtreBoyutu, kaydirma, boyutAyniKalsin);
	r = konvolusyonGri(r, filtre, rsmByt, filtreBoyutu, kaydirma, boyutAyniKalsin);
	//matBas(new Mat[3]{matris2MatGri(b,boy,en), matris2MatGri(g,boy,en), matris2MatGri(r,boy,en) },3);

	int yeniBoy = 0;
	int yeniEn = 0;

	if (boyutAyniKalsin) {
		yeniBoy = boy;
		yeniEn = en;
	}
	else {
		int p[2] = { 0,0 };
		int* yeniBoyut = yeniBoyutHesabi(resimBoyutu, filtreBoyutu, kaydirma, p);
		yeniBoy = yeniBoyut[0];
		yeniEn = yeniBoyut[1];
	}

	//3 kanali birlestirme
	for (int i = 0; i < yeniBoy; i++) {
		for (int j = 0; j < yeniEn; j++) {
			*(yeni + i * yeniEn * 3 + j * 3 + 0) = *(b + i * yeniEn + j);
			*(yeni + i * yeniEn * 3 + j * 3 + 1) = *(g + i * yeniEn + j);
			*(yeni + i * yeniEn * 3 + j * 3 + 2) = *(r + i * yeniEn + j);
		}
	}

	return yeni;
}