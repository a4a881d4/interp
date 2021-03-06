/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/

/**********************************************************************
* @file   phy_fft1024_core_dw.cpp
* @brief  Implementation of 1024-point_core fft function
* @author Chen,Jiansheng (jianshengx.chen@intel.com)
**********************************************************************/

/**********************************************************************
* Include public/global header files
**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE 2 
#include <pmmintrin.h> // SSE 3
#include <tmmintrin.h> // SSSE 3
#include <smmintrin.h> // SSE 4 for media
#include "common_typedef.h"

#ifndef PI
#define PI (3.14159265358979323846)
#endif

extern __m128i g_FFT1024r1024twiddle[256][2], g_FFT1024r32twiddle[32], g_FFT1024r32twiddle_core[32];

const static __m128i  FFT1024_IQ_switch = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
const static __m128i  FFT1024_Neg_I = _mm_setr_epi8(0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF );
const static __m128i  FFT1024_Neg_R = _mm_setr_epi8(0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1);
const static __m128i  FFT1024_Const_0707 = _mm_setr_epi8(0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A,0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A);
const static __m128i  FFT1024_Const_0707_Minus = _mm_setr_epi8(0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5);

static WORD16 double2short(DOUBLE64 d)
{
	d = floor(0.5 + d);
	if (d > 32767.0) return 32767;
	if (d < -32767.0) return -32767;
	return (WORD16)d;
}

void init_fft1024_wHscShift_twiddle_factor(WORD16 * r1024twiddle, WORD16 * r32twiddle, WORD16 * r32twiddle_core)
{
    WORD32 i, j;
    WORD32 twiddle_temp[1024][2];
    DOUBLE64 factor = 32768;
    WORD32 sign = 1;

	// r1024twiddle
	for (i=0; i<32; i++)
	{
		for (j=0; j<32; j++)
		{
			*(WORD16 *)(r1024twiddle + i*2*2*32 +2*2*j +0) = double2short( sign * factor * cos(-2.0 * PI / 1024 * (i+0.5) *j) / sqrt(2.0) ); /* c */
			*(WORD16 *)(r1024twiddle + i*2*2*32 +2*2*j +1) = double2short( sign * factor * cos(-2.0 * PI / 1024 * (i+0.5) *j) / sqrt(2.0) ); /* c */

			*(WORD16 *)(r1024twiddle + i*2*2*32 +2*2*j +2) = double2short( sign * factor * sin(-2.0 * PI / 1024 * (i+0.5) *j) / sqrt(2.0) ); /* d */
			*(WORD16 *)(r1024twiddle + i*2*2*32 +2*2*j +3) = double2short( -1 * sign * factor * sin(-2.0 * PI / 1024 * (i+0.5) *j) / sqrt(2.0) ); /* -d */
			sign *= -1;
		}
	}

	for (i=0;i<1024;i++)
	{
		twiddle_temp[i][0] = *( (WORD32 *)(r1024twiddle) + 2*i);
		twiddle_temp[i][1] = *( (WORD32 *)(r1024twiddle) + 2*i + 1);
	}

	for (i=0;i<1024/4;i++)
	{
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 0) =  twiddle_temp[i*4][0];
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 4) =  twiddle_temp[i*4][1];
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 1) =  twiddle_temp[i*4 + 1][0];
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 5) =  twiddle_temp[i*4 + 1][1];
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 2) =  twiddle_temp[i*4 + 2][0];
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 6) =  twiddle_temp[i*4 + 2][1];
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 3) =  twiddle_temp[i*4 + 3][0];
		*( (WORD32 *)(r1024twiddle) + 2*4*i + 7) =  twiddle_temp[i*4 + 3][1];
	}

	// r32twiddle
	for (i=0; i<8; i++)
	{
		for (j=0; j<4; j++)
		{
			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 0) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 1) = double2short( factor * cos(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 2) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 3) = double2short( factor * cos(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 4) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 5) = double2short( factor * cos(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 6) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*2*4*4 + 2*4*j + 7) = double2short( factor * cos(-2.0 * PI / 32 * i *j));
		}
	}
	
	// r32twiddle_core
	for (i=0; i<8; i++)
	{
		for (j=0; j<4; j++)
		{
			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 0) = double2short( factor * sin(-2.0 * PI / 32 * i * (j+0.5)));
			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 1) = double2short( factor * cos(-2.0 * PI / 32 * i * (j+0.5)));

			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 2) = double2short( factor * sin(-2.0 * PI / 32 * i * (j+0.5)));
			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 3) = double2short( factor * cos(-2.0 * PI / 32 * i * (j+0.5)));

			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 4) = double2short( factor * sin(-2.0 * PI / 32 * i * (j+0.5)));
			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 5) = double2short( factor * cos(-2.0 * PI / 32 * i * (j+0.5)));

			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 6) = double2short( factor * sin(-2.0 * PI / 32 * i * (j+0.5)));
			*(WORD16 *)(r32twiddle_core + i*2*4*4 + 2*4*j + 7) = double2short( factor * cos(-2.0 * PI / 32 * i * (j+0.5)));
		}
	}
}

#define radix4_register(in0, in1, in2, in3, out_addr, out_span) \
{ \
	__m128i * in_addr_temp; \
	m128_t2 = _mm_adds_epi16(in0, in2); \
	m128_t6 = _mm_adds_epi16(in1, in3); \
	m128_t7=  _mm_adds_epi16(m128_t2, m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2, m128_t6); \
	 \
	m128_t6 =  m128_t7; \
	 \
	m128_t3 = _mm_subs_epi16(in0, in2); \
	m128_t12 = _mm_subs_epi16(in1, in3); \
	m128_t12 = _mm_shuffle_epi8(m128_t12, FFT1024_IQ_switch); \
	m128_t12 = _mm_sign_epi16(m128_t12, FFT1024_Neg_R); \
	 \
	m128_t7 = _mm_subs_epi16(m128_t3, m128_t12); \
	m128_t3 = _mm_adds_epi16(m128_t3, m128_t12); \
	 \
	_mm_store_si128((__m128i *)out_addr, m128_t6); \
	out_addr = out_addr + out_span; \
	 \
	_mm_store_si128((__m128i *)out_addr, m128_t7); \
	out_addr = out_addr + out_span; \
	 \
	_mm_store_si128((__m128i *)out_addr, m128_t2); \
	out_addr = out_addr + out_span; \
	 \
	_mm_store_si128((__m128i *)out_addr, m128_t3); \
}

#define radix8_0(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i * out_addr_temp1, * out_addr_temp2; \
	__m128i * in_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t6 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t7 = _mm_load_si128((__m128i *)in_addr_temp); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t0, m128_t4); \
	m128_t9 = _mm_subs_epi16(m128_t0, m128_t4); \
	m128_t10 = _mm_adds_epi16(m128_t1, m128_t5); \
	m128_t11 = _mm_subs_epi16(m128_t1, m128_t5); \
	m128_t0 = _mm_adds_epi16(m128_t2, m128_t6); \
	m128_t1 = _mm_subs_epi16(m128_t2, m128_t6); \
	m128_t4 = _mm_adds_epi16(m128_t3, m128_t7); \
	m128_t5 = _mm_subs_epi16(m128_t3, m128_t7); \
	 \
	m128_t1 = _mm_shuffle_epi8(m128_t1, FFT1024_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, FFT1024_Neg_I); \
	 \
	m128_t2 = _mm_mulhrs_epi16(m128_t11, FFT1024_Const_0707); \
	m128_t3 = _mm_mulhrs_epi16(m128_t5, FFT1024_Const_0707_Minus); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t7 = _mm_adds_epi16(m128_t2, m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2, m128_t6); \
	m128_t11 = _mm_blend_epi16(m128_t2, m128_t7, 0x55); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t3, FFT1024_IQ_switch); \
    m128_t7 = _mm_adds_epi16(m128_t3, m128_t6); \
	m128_t3 = _mm_subs_epi16(m128_t3, m128_t6); \
	m128_t5 = _mm_blend_epi16(m128_t3, m128_t7, 0xAA); \
	 \
	 \
	out_addr_temp1 = out_addr; \
	out_addr_temp2 = out_addr + out_span; \
	radix4_register(m128_t8, m128_t10, m128_t0, m128_t4, out_addr_temp1, 2*out_span); \
	radix4_register(m128_t9, m128_t11, m128_t1, m128_t5, out_addr_temp2, 2*out_span); \
}

#define radix4_0_zeromul_core_cpy(in_addr, in_span, out_addr, in_addr_cpy) \
{ \
	__m128i * in_addr_temp; \
	__m128i * in_addr_cpy_temp; \
	__m128i * out_addr_temp; \
	__m128i * twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	 \
	in_addr_cpy_temp = in_addr_cpy; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t0);  in_addr_cpy_temp = in_addr_cpy_temp + in_span; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t1);  in_addr_cpy_temp = in_addr_cpy_temp + in_span; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t2);  in_addr_cpy_temp = in_addr_cpy_temp + in_span; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t3);  \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, FFT1024_Neg_I); \
	 \
	m128_t12 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t12, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_subs_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t11, 0x55); \
	 \
	m128_t13 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t13, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_subs_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_blend_epi16(m128_t13, m128_t11, 0xAA); \
	 \
	m128_t14 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t14, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_subs_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_blend_epi16(m128_t14, m128_t11, 0x55); \
	 \
	m128_t15 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t15, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_subs_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_blend_epi16(m128_t15, m128_t11, 0xAA); \
	 \
	 \
	m128_t4 = _mm_adds_epi16(m128_t0, m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t12, m128_t15); \
	m128_t6 = _mm_subs_epi16(m128_t0, m128_t2); \
	m128_t7 = _mm_adds_epi16(m128_t13, m128_t14); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4, m128_t5); \
	m128_t8 = _mm_srai_epi16(m128_t8, 1); \
	m128_t9 = _mm_adds_epi16(m128_t6, m128_t7); \
	m128_t9 = _mm_srai_epi16(m128_t9, 1); \
	m128_t10 = _mm_subs_epi16(m128_t4, m128_t5); \
	m128_t10 = _mm_srai_epi16(m128_t10, 1); \
	m128_t11 = _mm_subs_epi16(m128_t6, m128_t7); \
	m128_t11 = _mm_srai_epi16(m128_t11, 1); \
	 \
	out_addr_temp = out_addr; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t8);  out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t9);  out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t10); out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t11); \
}

#define radix4_0_mul_core_cpy(in_addr, in_span, out_addr, twiddle_addr, in_addr_cpy) \
{ \
	__m128i * in_addr_temp; \
	__m128i * in_addr_cpy_temp; \
	__m128i * out_addr_temp; \
	__m128i * twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	 \
	in_addr_cpy_temp = in_addr_cpy; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t0);  in_addr_cpy_temp = in_addr_cpy_temp + in_span; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t1);  in_addr_cpy_temp = in_addr_cpy_temp + in_span; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t2);  in_addr_cpy_temp = in_addr_cpy_temp + in_span; \
	_mm_store_si128((__m128i *)in_addr_cpy_temp, m128_t3);  \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, FFT1024_Neg_I); \
	 \
	m128_t12 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t12, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_subs_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t11, 0x55); \
	 \
	m128_t13 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t13, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_subs_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_blend_epi16(m128_t13, m128_t11, 0xAA); \
	 \
	m128_t14 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t14, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_subs_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_blend_epi16(m128_t14, m128_t11, 0x55); \
	 \
	m128_t15 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t15, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_subs_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_blend_epi16(m128_t15, m128_t11, 0xAA); \
	 \
	 \
	m128_t4 = _mm_adds_epi16(m128_t0, m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t12, m128_t15); \
	m128_t6 = _mm_subs_epi16(m128_t0, m128_t2); \
	m128_t7 = _mm_adds_epi16(m128_t13, m128_t14); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4, m128_t5); \
	m128_t9 = _mm_adds_epi16(m128_t6, m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4, m128_t5); \
	m128_t11 = _mm_subs_epi16(m128_t6, m128_t7); \
	 \
	twiddle_addr_temp = twiddle_addr; \
	m128_t14 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t2 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t3 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t3 = _mm_sign_epi16(m128_t3, FFT1024_Neg_I); \
	m128_t5 = _mm_shuffle_epi8(m128_t4, FFT1024_IQ_switch); \
	m128_t5 = _mm_sign_epi16(m128_t5, FFT1024_Neg_I); \
	m128_t7 = _mm_shuffle_epi8(m128_t6, FFT1024_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, FFT1024_Neg_I); \
	m128_t15 = _mm_shuffle_epi8(m128_t14, FFT1024_IQ_switch); \
	m128_t15 = _mm_sign_epi16(m128_t15, FFT1024_Neg_I); \
	 \
	out_addr_temp = out_addr; \
	m128_t12 = _mm_madd_epi16(m128_t8, m128_t14); \
	m128_t8 = _mm_madd_epi16(m128_t8, m128_t15); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12); out_addr_temp = out_addr_temp + 1; \
	 \
	m128_t12 = _mm_madd_epi16(m128_t9, m128_t2); \
	m128_t8 = _mm_madd_epi16(m128_t9, m128_t3); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp + 1; \
	m128_t12 = _mm_madd_epi16(m128_t10, m128_t4); \
	m128_t8 = _mm_madd_epi16(m128_t10, m128_t5); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp + 1; \
	m128_t12 = _mm_madd_epi16(m128_t11, m128_t6); \
	m128_t8 = _mm_madd_epi16(m128_t11, m128_t7); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12); \
}

#define radix4_0_zeromul_core(in_addr, in_span, out_addr) \
{ \
	__m128i * in_addr_temp; \
	__m128i * out_addr_temp; \
	__m128i * twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, FFT1024_Neg_I); \
	 \
	m128_t12 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t12, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_subs_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t11, 0x55); \
	 \
	m128_t13 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t13, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_subs_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_blend_epi16(m128_t13, m128_t11, 0xAA); \
	 \
	m128_t14 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t14, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_subs_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_blend_epi16(m128_t14, m128_t11, 0x55); \
	 \
	m128_t15 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t15, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_subs_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_blend_epi16(m128_t15, m128_t11, 0xAA); \
	 \
	 \
	m128_t4 = _mm_adds_epi16(m128_t0, m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t12, m128_t15); \
	m128_t6 = _mm_subs_epi16(m128_t0, m128_t2); \
	m128_t7 = _mm_adds_epi16(m128_t13, m128_t14); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4, m128_t5); \
	m128_t8 = _mm_srai_epi16(m128_t8, 1); \
	m128_t9 = _mm_adds_epi16(m128_t6, m128_t7); \
	m128_t9 = _mm_srai_epi16(m128_t9, 1); \
	m128_t10 = _mm_subs_epi16(m128_t4, m128_t5); \
	m128_t10 = _mm_srai_epi16(m128_t10, 1); \
	m128_t11 = _mm_subs_epi16(m128_t6, m128_t7); \
	m128_t11 = _mm_srai_epi16(m128_t11, 1); \
	 \
	out_addr_temp = out_addr; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t8);  out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t9);  out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t10); out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t11); \
}

#define radix4_0_mul_core(in_addr, in_span, out_addr, twiddle_addr) \
{ \
	__m128i * in_addr_temp; \
	__m128i * out_addr_temp; \
	__m128i * twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, FFT1024_Neg_I); \
	 \
	m128_t12 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t12, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_subs_epi16(m128_t12, m128_t10); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t11, 0x55); \
	 \
	m128_t13 = _mm_mulhrs_epi16(m128_t1, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t13, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_subs_epi16(m128_t13, m128_t10); \
	m128_t13 = _mm_blend_epi16(m128_t13, m128_t11, 0xAA); \
	 \
	m128_t14 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707); \
	m128_t10 = _mm_shuffle_epi8(m128_t14, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_subs_epi16(m128_t14, m128_t10); \
	m128_t14 = _mm_blend_epi16(m128_t14, m128_t11, 0x55); \
	 \
	m128_t15 = _mm_mulhrs_epi16(m128_t3, FFT1024_Const_0707_Minus); \
	m128_t10 = _mm_shuffle_epi8(m128_t15, FFT1024_IQ_switch); \
	m128_t11 = _mm_adds_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_subs_epi16(m128_t15, m128_t10); \
	m128_t15 = _mm_blend_epi16(m128_t15, m128_t11, 0xAA); \
	 \
	 \
	m128_t4 = _mm_adds_epi16(m128_t0, m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t12, m128_t15); \
	m128_t6 = _mm_subs_epi16(m128_t0, m128_t2); \
	m128_t7 = _mm_adds_epi16(m128_t13, m128_t14); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4, m128_t5); \
	m128_t9 = _mm_adds_epi16(m128_t6, m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4, m128_t5); \
	m128_t11 = _mm_subs_epi16(m128_t6, m128_t7); \
	 \
	twiddle_addr_temp = twiddle_addr; \
	m128_t14 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t2 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t3 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t3 = _mm_sign_epi16(m128_t3, FFT1024_Neg_I); \
	m128_t5 = _mm_shuffle_epi8(m128_t4, FFT1024_IQ_switch); \
	m128_t5 = _mm_sign_epi16(m128_t5, FFT1024_Neg_I); \
	m128_t7 = _mm_shuffle_epi8(m128_t6, FFT1024_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, FFT1024_Neg_I); \
	m128_t15 = _mm_shuffle_epi8(m128_t14, FFT1024_IQ_switch); \
	m128_t15 = _mm_sign_epi16(m128_t15, FFT1024_Neg_I); \
	 \
	out_addr_temp = out_addr; \
	m128_t12 = _mm_madd_epi16(m128_t8, m128_t14); \
	m128_t8 = _mm_madd_epi16(m128_t8, m128_t15); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12); out_addr_temp = out_addr_temp + 1; \
	 \
	m128_t12 = _mm_madd_epi16(m128_t9, m128_t2); \
	m128_t8 = _mm_madd_epi16(m128_t9, m128_t3); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12, m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp + 1; \
	m128_t12 = _mm_madd_epi16(m128_t10, m128_t4); \
	m128_t8 = _mm_madd_epi16(m128_t10, m128_t5); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp + 1; \
	m128_t12 = _mm_madd_epi16(m128_t11, m128_t6); \
	m128_t8 = _mm_madd_epi16(m128_t11, m128_t7); \
	m128_t8 = _mm_srli_si128(m128_t8, 2); \
	m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12); \
}

#define radix32_core_cpy(InBuf, in_span, OutBuf, out_span, r32twiddle_core, InBufCpy) \
{ \
	radix4_0_zeromul_core_cpy(InBuf + 0*in_span, 8*in_span, Temp32_32_Buf +0*4, InBufCpy + 0*in_span); \
	radix4_0_mul_core_cpy(InBuf + 1*in_span, 8*in_span, Temp32_32_Buf + 1*4, r32twiddle_core + 4*1, InBufCpy + 1*in_span); \
	radix4_0_mul_core_cpy(InBuf + 2*in_span, 8*in_span, Temp32_32_Buf + 2*4, r32twiddle_core + 4*2, InBufCpy + 2*in_span); \
	radix4_0_mul_core_cpy(InBuf + 3*in_span, 8*in_span, Temp32_32_Buf + 3*4, r32twiddle_core + 4*3, InBufCpy + 3*in_span); \
	radix4_0_mul_core_cpy(InBuf + 4*in_span, 8*in_span, Temp32_32_Buf + 4*4, r32twiddle_core + 4*4, InBufCpy + 4*in_span); \
	radix4_0_mul_core_cpy(InBuf + 5*in_span, 8*in_span, Temp32_32_Buf + 5*4, r32twiddle_core + 4*5, InBufCpy + 5*in_span); \
	radix4_0_mul_core_cpy(InBuf + 6*in_span, 8*in_span, Temp32_32_Buf + 6*4, r32twiddle_core + 4*6, InBufCpy + 6*in_span); \
	radix4_0_mul_core_cpy(InBuf + 7*in_span, 8*in_span, Temp32_32_Buf + 7*4, r32twiddle_core + 4*7, InBufCpy + 7*in_span); \
	 \
	radix8_0(Temp32_32_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span); \
}

#define radix32_core(InBuf, in_span, OutBuf, out_span, r32twiddle_core) \
{ \
	radix4_0_zeromul_core(InBuf + 0*in_span, 8*in_span, Temp32_32_Buf +0*4); \
	radix4_0_mul_core(InBuf + 1*in_span, 8*in_span, Temp32_32_Buf + 1*4, r32twiddle_core + 4*1); \
	radix4_0_mul_core(InBuf + 2*in_span, 8*in_span, Temp32_32_Buf + 2*4, r32twiddle_core + 4*2); \
	radix4_0_mul_core(InBuf + 3*in_span, 8*in_span, Temp32_32_Buf + 3*4, r32twiddle_core + 4*3); \
	radix4_0_mul_core(InBuf + 4*in_span, 8*in_span, Temp32_32_Buf + 4*4, r32twiddle_core + 4*4); \
	radix4_0_mul_core(InBuf + 5*in_span, 8*in_span, Temp32_32_Buf + 5*4, r32twiddle_core + 4*5); \
	radix4_0_mul_core(InBuf + 6*in_span, 8*in_span, Temp32_32_Buf + 6*4, r32twiddle_core + 4*6); \
	radix4_0_mul_core(InBuf + 7*in_span, 8*in_span, Temp32_32_Buf + 7*4, r32twiddle_core + 4*7); \
	 \
	radix8_0(Temp32_32_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span); \
}

#define radix4_0_zeromul(in_addr, in_span, out_addr) \
{ \
	__m128i * in_addr_temp; \
	__m128i * out_addr_temp; \
	__m128i * twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, FFT1024_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, FFT1024_Neg_R); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t8 = _mm_srai_epi16(m128_t8,1); \
	m128_t9 = _mm_subs_epi16(m128_t6,m128_t7); \
	m128_t9 = _mm_srai_epi16(m128_t9,1); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t10 = _mm_srai_epi16(m128_t10,1); \
	m128_t11 = _mm_adds_epi16(m128_t6,m128_t7); \
	m128_t11 = _mm_srai_epi16(m128_t11,1); \
	 \
	out_addr_temp = out_addr; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t8);  out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t9);  out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t10); out_addr_temp = out_addr_temp +1; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t11); \
}

#define radix4_0_mul(in_addr, in_span, out_addr, twiddle_addr) \
{ \
	__m128i * in_addr_temp; \
	__m128i * out_addr_temp; \
	__m128i * twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, FFT1024_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, FFT1024_Neg_R); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t9 = _mm_subs_epi16(m128_t6,m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t11 = _mm_adds_epi16(m128_t6,m128_t7); \
	twiddle_addr_temp = twiddle_addr + 1; \
	m128_t2 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t3 = _mm_shuffle_epi8(m128_t2, FFT1024_IQ_switch); \
	m128_t3 = _mm_sign_epi16(m128_t3, FFT1024_Neg_I); \
	m128_t5 = _mm_shuffle_epi8(m128_t4, FFT1024_IQ_switch); \
	m128_t5 = _mm_sign_epi16(m128_t5, FFT1024_Neg_I); \
	m128_t7 = _mm_shuffle_epi8(m128_t6, FFT1024_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, FFT1024_Neg_I); \
	 \
	out_addr_temp = out_addr; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t8,1)); out_addr_temp = out_addr_temp +1; \
	 \
	m128_t12 = _mm_madd_epi16(m128_t9, m128_t2); \
	m128_t8 = _mm_madd_epi16(m128_t9, m128_t3); \
	m128_t8 = _mm_srli_si128(m128_t8,2); \
	m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp +1; \
	m128_t12 = _mm_madd_epi16(m128_t10, m128_t4); \
	m128_t8 = _mm_madd_epi16(m128_t10, m128_t5); \
	m128_t8 = _mm_srli_si128(m128_t8,2); \
	m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp +1; \
	m128_t12 = _mm_madd_epi16(m128_t11, m128_t6); \
	m128_t8 = _mm_madd_epi16(m128_t11, m128_t7); \
	m128_t8 = _mm_srli_si128(m128_t8,2); \
	m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12); \
}

#define radix32(InBuf, in_span, OutBuf, out_span, r32twiddle) \
{ \
	radix4_0_zeromul(InBuf + 0*in_span, 8*in_span, Temp32_32_Buf + 0*4); \
	radix4_0_mul(InBuf + 1*in_span, 8*in_span, Temp32_32_Buf + 1*4, r32twiddle + 4*1); \
	radix4_0_mul(InBuf + 2*in_span, 8*in_span, Temp32_32_Buf + 2*4, r32twiddle + 4*2); \
	radix4_0_mul(InBuf + 3*in_span, 8*in_span, Temp32_32_Buf + 3*4, r32twiddle + 4*3); \
	radix4_0_mul(InBuf + 4*in_span, 8*in_span, Temp32_32_Buf + 4*4, r32twiddle + 4*4); \
	radix4_0_mul(InBuf + 5*in_span, 8*in_span, Temp32_32_Buf + 5*4, r32twiddle + 4*5); \
	radix4_0_mul(InBuf + 6*in_span, 8*in_span, Temp32_32_Buf + 6*4, r32twiddle + 4*6); \
	radix4_0_mul(InBuf + 7*in_span, 8*in_span, Temp32_32_Buf + 7*4, r32twiddle + 4*7); \
	 \
	radix8_0(Temp32_32_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span); \
	radix8_0(Temp32_32_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span); \
}

#define _MM_TRANSPOSE4_EPI32(in0, in1, in2, in3) \
{ \
	__m128i tmp0, tmp1, tmp2, tmp3; \
	tmp0 =  _mm_unpacklo_epi32(in0, in1); \
	tmp1 =  _mm_unpackhi_epi32(in0, in1); \
	tmp2 =  _mm_unpacklo_epi32(in2, in3); \
	tmp3 =  _mm_unpackhi_epi32(in2, in3); \
	 \
	in0 =  _mm_unpacklo_epi64(tmp0, tmp2); \
	in1 =  _mm_unpackhi_epi64(tmp0, tmp2); \
	in2 =  _mm_unpacklo_epi64(tmp1, tmp3); \
	in3 =  _mm_unpackhi_epi64(tmp1, tmp3); \
}

#define transpose32x4(InBuf, in_span, OutBuf, twiddle_addr) \
{ \
	__m128i *twiddle_addr_temp; \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	 \
	for (WORD32 ii=0;ii<32/4;ii++) \
	{ \
		twiddle_addr_temp = twiddle_addr + ii * 2; \
		m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t1 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 15; \
		m128_t2 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 15; \
		m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 15; \
		m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t7 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
		 \
		 \
		in_addr_temp = InBuf + ii*in_span; \
		m128_t8 = _mm_load_si128((__m128i *)in_addr_temp); in_addr_temp = in_addr_temp + 8*in_span; \
		m128_t9 = _mm_load_si128((__m128i *)in_addr_temp); in_addr_temp = in_addr_temp + 8*in_span; \
		m128_t10 = _mm_load_si128((__m128i *)in_addr_temp); in_addr_temp = in_addr_temp + 8*in_span; \
		m128_t11 = _mm_load_si128((__m128i *)in_addr_temp); \
		 \
		m128_t0 = _mm_mulhrs_epi16(m128_t8, m128_t0); \
		m128_t1 = _mm_mulhrs_epi16(m128_t8, m128_t1); \
		m128_t1 = _mm_shuffle_epi8(m128_t1,FFT1024_IQ_switch); \
		m128_t1 = _mm_adds_epi16(m128_t0,m128_t1); \
		 \
		 \
		m128_t2 = _mm_mulhrs_epi16(m128_t9, m128_t2); \
		m128_t3 = _mm_mulhrs_epi16(m128_t9, m128_t3); \
		m128_t3 = _mm_shuffle_epi8(m128_t3,FFT1024_IQ_switch); \
		m128_t3 = _mm_adds_epi16(m128_t2,m128_t3); \
		 \
		 \
		m128_t4 = _mm_mulhrs_epi16(m128_t10, m128_t4); \
		m128_t5 = _mm_mulhrs_epi16(m128_t10, m128_t5); \
		m128_t5 = _mm_shuffle_epi8(m128_t5,FFT1024_IQ_switch); \
		m128_t5 = _mm_adds_epi16(m128_t4,m128_t5); \
		 \
		 \
		m128_t6 = _mm_mulhrs_epi16(m128_t11, m128_t6); \
		m128_t7 = _mm_mulhrs_epi16(m128_t11, m128_t7); \
		m128_t7 = _mm_shuffle_epi8(m128_t7,FFT1024_IQ_switch); \
		m128_t7 = _mm_adds_epi16(m128_t6,m128_t7); \
		 \
		_MM_TRANSPOSE4_EPI32(m128_t1, m128_t3, m128_t5, m128_t7); \
		 \
		out_addr_temp = OutBuf + ii*4; \
		_mm_store_si128((__m128i *)out_addr_temp, m128_t1);  out_addr_temp = out_addr_temp + 1; \
		_mm_store_si128((__m128i *)out_addr_temp, m128_t3);  out_addr_temp = out_addr_temp + 1; \
		_mm_store_si128((__m128i *)out_addr_temp, m128_t5);  out_addr_temp = out_addr_temp + 1; \
		_mm_store_si128((__m128i *)out_addr_temp, m128_t7); \
	} \
}

/***************************************************************************//**
 * @fn void fft1024_core(__m128i *InBuf, __m128i *OutBuf,  __m128i *r32twiddle, __m128i *r32twiddle_2, __m128i *r1024twiddle)
 * @brief Perform the FFT1024
 * @param [in] InBuf Pointer of input buffer
 * @param [out] OutBuf Pointer of output buffer
 * @param [in] r32twiddle Pointer of twiddle factor for radix32
 * @param [in] r32twiddle_core Pointer of twiddle factor for radix32_core
 * @param [in] r1024twiddle Pointer of twiddle factor for radix1024
 *******************************************************************************/
void fft1024_core(__m128i *InBuf, __m128i *OutBuf, __m128i *r32twiddle, __m128i *r32twiddle_core, __m128i *r1024twiddle)
{
    WORD32 i_loop=0;
    WORD32 in_span, out_span;
	__m128i Temp32_32_Buf[32];
	__m128i TransposeBuf[32];
	__m128i OutTmp[256];
	__m128i m128_t0, m128_t1, m128_t2, m128_t3, m128_t4, m128_t5, m128_t6, m128_t7, \
		m128_t8, m128_t9, m128_t10, m128_t11, m128_t12, m128_t13, m128_t14, m128_t15;

	in_span = 8;
	out_span = 8;
	for (i_loop=0;i_loop<8;i_loop++)
	{
		radix32_core((__m128i *)(InBuf) + i_loop, in_span, (__m128i *)(OutTmp) + i_loop, out_span, (__m128i *)r32twiddle_core);
	}

	in_span = 1;
	out_span = 8;
	for (i_loop=0;i_loop<8;i_loop++)
	{
		transpose32x4((__m128i *)(OutTmp) + i_loop*32, in_span, (__m128i *)TransposeBuf, (__m128i *)r1024twiddle + 2*i_loop*32);
		radix32((__m128i *)TransposeBuf, in_span,(__m128i *)OutBuf + i_loop, out_span, (__m128i *)r32twiddle);
	}
}


void fft1024_core_cpy(__m128i *InBuf, __m128i *OutBuf, __m128i *r32twiddle, __m128i *r32twiddle_core, __m128i *r1024twiddle,  __m128i *InBufCpy)
{
    WORD32 i_loop=0;
    WORD32 in_span, out_span;
	__m128i Temp32_32_Buf[32];
	__m128i TransposeBuf[32];
	__m128i OutTmp[256];
	__m128i m128_t0, m128_t1, m128_t2, m128_t3, m128_t4, m128_t5, m128_t6, m128_t7, \
		m128_t8, m128_t9, m128_t10, m128_t11, m128_t12, m128_t13, m128_t14, m128_t15;

	in_span = 8;
	out_span = 8;
	for (i_loop=0;i_loop<8;i_loop++)
	{
		radix32_core_cpy((__m128i *)(InBuf) + i_loop, in_span, (__m128i *)(OutTmp) + i_loop, out_span, (__m128i *)r32twiddle_core, (__m128i *)(InBufCpy) + i_loop);
	}

	in_span = 1;
	out_span = 8;
	for (i_loop=0;i_loop<8;i_loop++)
	{
		transpose32x4((__m128i *)(OutTmp) + i_loop*32, in_span, (__m128i *)TransposeBuf, (__m128i *)r1024twiddle + 2*i_loop*32);
		radix32((__m128i *)TransposeBuf, in_span,(__m128i *)OutBuf + i_loop, out_span, (__m128i *)r32twiddle);
	}
}
