// Zadatak 1 - Matrix MULL.cpp
//
//Potrebno je optimizovati dati program koriscenjem SIMD instrukcija.
//Progam sabira mnozi dve kvadratne matrice velicine ARRAY_SIZE x ARRAY_SIZE i rezultat smesta u trecu matricu.
//Elementi matrice su tipa float (32 bit)
//Proveriti ispravnost optimizovanog programa.
//Originalni program rezultat smesta u matricu c[i][j], 
//a optimizan program treba rezultat da smesti u matricu d[i][j]

#include <iostream>
#include "headers/_Timer.h"

#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

#define ARRAY_SIZE 800

float a[ARRAY_SIZE][ARRAY_SIZE];
float b[ARRAY_SIZE][ARRAY_SIZE];
float c[ARRAY_SIZE][ARRAY_SIZE];
float d[ARRAY_SIZE][ARRAY_SIZE];

// https://software.intel.com/sites/landingpage/IntrinsicsGuide/


int main2(int argc, const char* argv[])
{

	short x = 0;
	for (int i = 0; i < ARRAY_SIZE; i++)
		for (int j = 0; j < ARRAY_SIZE; j++) {
			a[i][j] = x % 10;
			b[i][j] = x++ % 10;
			// % 10 -> zbog prekoracenja prilikom mnozenja i sabiranjas
		}


	StartTimer(No SIMD)
		for (int w = 0; w < 1000; w++)	// Za vise iteracija
			for (int i = 0; i < ARRAY_SIZE; i++) {
				for (int k = 0; k < ARRAY_SIZE; k++) {
					int sum = c[i][k];
					for (int j = 0; j < ARRAY_SIZE; j++) {
						sum = sum + a[i][j] * b[j][k];
					}
					c[i][k] = sum;
				}
			}
	EndTimer

		StartTimer(SIMD)
		for (int w = 0; w < 1000; w++)	// Za vise iteracija
		for (int i = 0; i < ARRAY_SIZE; i++) {
			for (int k = 0; k < ARRAY_SIZE; k++) {

				__m256 vd = _mm256_set1_ps(0);
				for (int j = 0; j < ARRAY_SIZE; j += 8) {

					__m256 va = _mm256_loadu_ps((const float*)(a[i] + j));
					__m256 vb = _mm256_set_ps(b[j + 7][k], b[j + 6][k], b[j + 5][k], b[j + 4][k], b[j + 3][k], b[j + 2][k], b[j + 1][k], b[j + 0][k]);
					vd = _mm256_fmadd_ps(va, vb, vd);

				}
				float sum = d[i][k];
				for (int z = 0; z < 8; z++)
					sum += vd.m256_f32[z];

				d[i][k] = sum;
			}
		}
	EndTimer

		int same = 1;
	for (int i = 0; i < ARRAY_SIZE && same == 1; i++)
		for (int j = 0; j < ARRAY_SIZE && same == 1; j++) {
			if (c[i][j] != d[i][j]) {
				same = 0;
				goto endcheck;
			}
		}

endcheck:
	if (same == 1)
		cout << endl << "OK" << endl;
	else
		cout << endl << "NOT OK" << endl;


	return 0;
}