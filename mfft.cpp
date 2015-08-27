#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE 2 
#include <pmmintrin.h> // SSE 3
#include <tmmintrin.h> // SSSE 3
#include <smmintrin.h> // SSE 4 for media


#include "common_structure.h"
#include "common_function.h"

extern __m128i g_fft2048_r2048twiddle[512][2], g_fft2048_r32twiddle[32][2], g_fft2048_r64twiddle[64][2];
extern __m128i g_FFTr2048twiddle[512], g_FFTr32twiddle[32], g_FFTr64twiddle[64];
extern __m128i g_r1024twiddle[256][2], g_r32twiddle[32][2];
extern __m128i g_FFT1024r1024twiddle[256][2], g_FFT1024r32twiddle[32], g_FFT1024r32twiddle_core[32];

// IFFT twiddle 
extern __m128i g_ifft2048_r2048twiddle[512][2], g_ifft2048_r32twiddle[32][2], g_ifft2048_r64twiddle[64][2];
extern __m128i g_ifft1024_r1024twiddle[256][2], g_ifft1024_r32twiddle[32][2];


extern "C" void fft2048_c( void *in, void *out )
{
	fft2048_core( 
		  (__m128i *) in
		, (__m128i *) out
		, g_FFTr32twiddle
		, g_FFTr64twiddle
		, g_FFTr2048twiddle
		);
}

extern "C" void ifft2048( void *in, void *out )
{
	ifft2048(
		  (__m128i *)in
		, (__m128i *)out
		, &g_ifft2048_r32twiddle[0][0]
		, &g_ifft2048_r64twiddle[0][0]
		, &g_ifft2048_r2048twiddle[0][0]
		);
}

extern "C" void fft1024_c( void *in, void *out )
{
	fft1024_core(
		  (__m128i *) in
		, (__m128i *) out
		, g_FFT1024r32twiddle
		, g_FFT1024r32twiddle_core
		, &g_FFT1024r1024twiddle[0][0]
		);
}

extern "C" void fft2048_n( void *in, void *out )
{
	fft2048( 
		  (__m128i *) in
		, (__m128i *) out
		, &g_fft2048_r32twiddle[0][0]
		, &g_fft2048_r64twiddle[0][0]
		, &g_fft2048_r2048twiddle[0][0]
		);
}
	
