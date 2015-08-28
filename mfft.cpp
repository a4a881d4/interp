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

const static __m128i  IQ_switch = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
const static __m128i  Neg_I = _mm_setr_epi8(0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF);
const static __m128i  Neg_R = _mm_setr_epi8(0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1);

#define radix4_0_mul(in_addr, in_span, out_addr, out_span, twiddle_addr) \
{ \
    __m128i * in_addr_temp; \
    __m128i * out_addr_temp; \
    __m128i * twiddle_addr_temp; \
    in_addr_temp = in_addr; \
    m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
    m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
    m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
    m128_t3 = _mm_load_si128(in_addr_temp); \
    m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); /*AA*/ \
    m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); /*CC*/ \
    m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); /*BB*/ \
    m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); /*DD*/ \
    m128_t7 = _mm_shuffle_epi8(m128_t7, IQ_switch); \
    m128_t7 = _mm_sign_epi16(m128_t7, Neg_R);/*j*D*/ \
     \
    m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
    m128_t9 = _mm_subs_epi16(m128_t6,m128_t7); \
    m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
    m128_t11 = _mm_adds_epi16(m128_t6,m128_t7); \
    twiddle_addr_temp = twiddle_addr + 1; \
    m128_t2 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
    m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
    m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
    m128_t3 = _mm_shuffle_epi8(m128_t2, IQ_switch); \
    m128_t3 = _mm_sign_epi16(m128_t3, Neg_I); \
    m128_t5 = _mm_shuffle_epi8(m128_t4, IQ_switch); \
    m128_t5 = _mm_sign_epi16(m128_t5, Neg_I); \
    m128_t7 = _mm_shuffle_epi8(m128_t6, IQ_switch); \
    m128_t7 = _mm_sign_epi16(m128_t7, Neg_I); \
     \
    out_addr_temp = out_addr; \
    _mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t8,1)); out_addr_temp = out_addr_temp + out_span; \
     \
    m128_t12 = _mm_madd_epi16(m128_t9, m128_t2); \
    m128_t8 = _mm_madd_epi16(m128_t9, m128_t3); \
    m128_t8 = _mm_srli_si128(m128_t8,2); \
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp + out_span; \
    m128_t12 = _mm_madd_epi16(m128_t10, m128_t4); \
    m128_t8 = _mm_madd_epi16(m128_t10, m128_t5); \
    m128_t8 = _mm_srli_si128(m128_t8,2); \
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp + out_span; \
    m128_t12 = _mm_madd_epi16(m128_t11, m128_t6); \
    m128_t8 = _mm_madd_epi16(m128_t11, m128_t7); \
    m128_t8 = _mm_srli_si128(m128_t8,2); \
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp, m128_t12); \
}

extern unsigned short g_twiddle_8192_2048_4X4[8192*2];

extern "C" void fft8192( void *in, void *out )
{
	__m128i *pin, *pout, *pt;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	int span;
	int i_loop;
	pin = (__m128i *) in;
	pout = (__m128i *) out;
	pt = (__m128i *) g_twiddle_8192_2048_4X4;
	span = 512;
	for (i_loop=0;i_loop<2048/4;i_loop++)
	{
		radix4_0_mul( pin + i_loop, span, pin + i_loop, span, pt + i_loop*4 );
	}
	
	fft2048_n( pin+0, pout+0 );
	fft2048_n( pin+2048/4, pout+2048/4 );
	fft2048_n( pin+2*2048/4, pout+2*2048/4 );
	fft2048_n( pin+3*2048/4, pout+2*2048/4 );
/*
	fft2048_n( pout+0, pout+0 );
	fft2048_n( pout+2048/4, pout+2048/4 );
	fft2048_n( pout+2*2048/4, pout+2*2048/4 );
	fft2048_n( pout+3*2048/4, pout+2*2048/4 );
*/
}
	
