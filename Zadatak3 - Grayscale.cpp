// Zadatak 3 - Grayscale.cpp
//
//Potrebno je implementirati algoritam za transformisanje slike cije su boje definisane u RGBA formatu u Grayscale format boja.
//Progam treba da procita zadatu sliku (na osnovu putanje) i da pored nje na fajl sistemu napravi transformisanu sliku.
//Smatrati da slika bmp formata
// Transformacija: 
// R <- R*0.299
// G <- G*0.587
// B <- B*0.114


#include <iostream>
#include "headers/_Timer.h"

#include <emmintrin.h>
#include <immintrin.h>


#include "headers/BMP.h"

using namespace std;


// https://software.intel.com/sites/landingpage/IntrinsicsGuide/

int main3(int argc, const char* argv[])
{
	string bmpImgURL = ".\\img\\slika.bmp";

	string bmpImgURLNOSIMD = bmpImgURL.substr(0, bmpImgURL.find_last_of(".")) + "_GRAYSCALE_NOSIMD.bmp";
	string bmpImgURLSIMD = bmpImgURL.substr(0, bmpImgURL.find_last_of(".")) + "_GRAYSCALE_SIMD.bmp";

	BMP bmp(bmpImgURL.c_str());
	const uint8_t* pixels = bmp.data.data();

	BMP bmp1(bmp.bmp_info_header.width, bmp.bmp_info_header.height);
	vector<uint8_t> newPixel(bmp.data.size());


	StartTimer(No SIMD)
		for (int i = 0; i < bmp.data.size(); i += 4)
		{

			float B = pixels[i + 0] * 0.114f;
			float G = pixels[i + 1] * 0.587f;
			float R = pixels[i + 2] * 0.299f;
			uint8_t A = pixels[i + 3];

			uint8_t I = (R + G + B);

			newPixel[i + 0] = I;
			newPixel[i + 1] = I;
			newPixel[i + 2] = I;
			newPixel[i + 3] = A;

		}
	bmp1.data = newPixel;

	bmp1.write(bmpImgURLNOSIMD.c_str());


	EndTimer

		// Brisanje (resetovanje) je suvisno
		std::fill(newPixel.begin(), newPixel.end(), 0);

	StartTimer(SIMD)

		uint8_t R[8], G[8], B[8], A[8];

	__m256 vCoefR = _mm256_set1_ps(0.299);
	__m256 vCoefG = _mm256_set1_ps(0.587);
	__m256 vCoefB = _mm256_set1_ps(0.114);
	for (int i = 0; i < newPixel.size();)
	{
		int k = i;

		for (int j = 0; j < 8; j++) {
			B[j] = pixels[i++];
			G[j] = pixels[i++];
			R[j] = pixels[i++];
			A[j] = pixels[i++];
		}

		__m256 vR = _mm256_cvtepi32_ps(_mm256_setr_epi32(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7]));
		//__m256 vR = _mm256_setr_ps(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7]);
		__m256 sum = _mm256_mul_ps(vR, vCoefR);

		__m256 vG = _mm256_cvtepi32_ps(_mm256_setr_epi32(G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]));
		//__m256 vG = _mm256_setr_ps(G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]);
		sum = _mm256_fmadd_ps(vG, vCoefG, sum);

		__m256 vB = _mm256_cvtepi32_ps(_mm256_setr_epi32(B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]));
		//__m256 vB = _mm256_setr_ps(B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]);
		sum = _mm256_fmadd_ps(vB, vCoefB, sum);

		__m256i sumI = _mm256_cvtps_epi32(sum);

		for (int j = 0; j < 8; j++) {
			uint16_t _gray = sumI.m256i_u8[j * 4];
			newPixel[k++] = _gray;
			newPixel[k++] = _gray;
			newPixel[k++] = _gray;
			newPixel[k++] = A[j];
		}

	}

	bmp1.data = newPixel;

	bmp1.write(bmpImgURLSIMD.c_str());

	EndTimer


		return 0;
}

