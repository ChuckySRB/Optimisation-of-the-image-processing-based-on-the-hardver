// Zadatak 1 - Array ADD.cpp
//
//Potrebno je optimizovati dati program koriscenjem SIMD instrukcija.
//Progam sabira dva niza velicine ARRAY_SIZE i rezultat smesta u treci niz.
//Proveriti ispravnost optimizovanog programa.
//Originalni program rezultat smesta u niz c[i], 
//a optimizovan program treba rezultat da smesti u niz d[i]

#include <iostream>
#include "headers/_Timer.h"

#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

#define ARRAY_SIZE 10240


int a[ARRAY_SIZE];
int b[ARRAY_SIZE];
int c[ARRAY_SIZE];
int d[ARRAY_SIZE];

// https://software.intel.com/sites/landingpage/IntrinsicsGuide/

int main1(int argc, const char* argv[])
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		a[i] = rand();
		b[i] = rand();
	}

	StartTimer(No SIMD)

		for (int i = 0; i < ARRAY_SIZE; i++) {
			c[i] = a[i] + b[i];
		}

	EndTimer


		StartTimer(SIMD)

		int i = 0;
	for (; i <= ARRAY_SIZE - 8; i += 8) {

			__m256i va = _mm256_load_si256((__m256i*)(a + i));
			__m256i vb = _mm256_load_si256((__m256i*)(b + i));

			__m256i vd = _mm256_add_epi32(va, vb);

			_mm256_store_si256((__m256i*)(d + i), vd);

		}

	for (; i < ARRAY_SIZE; i++) {
			d[i] = a[i] + b[i];
		}



	EndTimer

		for (int i = 0; i < ARRAY_SIZE; i++) {
			if (c[i] != d[i]) {
				cout << std::endl << "NOT SAME ON INDEX:" << i << endl;
				break;
			}
		}

	return 0;
}