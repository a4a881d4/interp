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


#ifndef PI
#define PI (3.14159265358979323846)
#endif

const static __m128i  IQ_switch =  _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
const static __m128i  Neg_I = _mm_setr_epi8(0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF );
const static __m128i  Neg_R = _mm_setr_epi8(0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1);
const static __m128i  Const_0707 = _mm_setr_epi8(0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A,0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A);
const static __m128i  Const_0707_Minus = _mm_setr_epi8(0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5); 

static WORD16 double2short(DOUBLE64 d)
{
	d = floor(0.5 + d);
	if (d >= 32767.0) return 32767;
	if (d <-32768.0) return -32768;
	return (WORD16)d;
}

/** \fn void init_ifft2048_twiddle_factor(WORD16 * ifft2048_r2048twiddle, WORD16 * ifft2048_r32twiddle, WORD16* ifft2048_r64twiddle)
*  \brief generate twiddle factor table for 2048-point IFFT
*  \param ifft2048_r2048twiddle WORD16*  [o] constant table buffer. Its size is 512*2*16 bytes and the buffer is 16-byte aligned.
*  \param ifft2048_r32twiddle WORD16*  [o] constant table buffer. Its size is 32**2*16 bytes and the buffer is 16-byte aligned.
*  \param ifft2048_r64twiddle WORD16*  [o] constant table buffer. Its size is 64*2*16 bytes and the buffer is 16-byte aligned.
*/ 

/*  twiddle1:d +cj  twiddle2 : c -dj;    */
void init_ifft2048_twiddle_factor(WORD16 * ifft2048_r2048twiddle, WORD16 * ifft2048_r32twiddle, WORD16* ifft2048_r64twiddle)
{
	// ifft2048_r2048twiddle
    WORD32 i,j;
    WORD32 twiddle_temp[2048][2];
    DOUBLE64 factor= 32768; /*convert DOUBLE64 to short*/


	for (i=0;i<32;i++)
		for (j=0;j<64;j++)
		{
			/* d +cj  */
			*(WORD16 *)(ifft2048_r2048twiddle + i*2*2*64 +2*2*j +0) = double2short( factor * sin(2.0 * PI / 2048 * i *j)); /* d */
			*(WORD16 *)(ifft2048_r2048twiddle + i*2*2*64 +2*2*j +1) = double2short( factor * cos(2.0 * PI / 2048 * i *j)); /* c */
			/* c -dj; */
			*(WORD16 *)(ifft2048_r2048twiddle + i*2*2*64 +2*2*j +2) = double2short( factor * cos(2.0 * PI / 2048 * i *j)); /* c */
			*(WORD16 *)(ifft2048_r2048twiddle + i*2*2*64 +2*2*j +3) = double2short( -1*factor * sin(2.0 * PI / 2048 * i *j)); /* -d */
			/* */
		}

        memcpy((WORD8*)(&twiddle_temp[0][0]), (WORD8*)ifft2048_r2048twiddle, 2048*2*sizeof(WORD32));


		for (i=0;i<2048/4;i++)
		{
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 0) =  twiddle_temp[i*4][0];
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 4) =  twiddle_temp[i*4][1];
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 1) =  twiddle_temp[i*4 + 1][0];
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 5) =  twiddle_temp[i*4 + 1][1];
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 2) =  twiddle_temp[i*4 + 2][0];
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 6) =  twiddle_temp[i*4 + 2][1];
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 3) =  twiddle_temp[i*4 + 3][0];
			*( (WORD32 *)(ifft2048_r2048twiddle) + 2*4*i + 7) =  twiddle_temp[i*4 + 3][1];
		}

		for (i=0;i<8;i++)
			for (j=0;j<8;j++)
			{
				/* d +cj  */
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +0) = double2short( factor * sin(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +1) = double2short( factor * cos(2.0 * PI / 64 * i *j));

				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +2) = double2short( factor * sin(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +3) = double2short( factor * cos(2.0 * PI / 64 * i *j));

				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +4) = double2short( factor * sin(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +5) = double2short( factor * cos(2.0 * PI / 64 * i *j));

				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +6) = double2short( factor * sin(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +7) = double2short( factor * cos(2.0 * PI / 64 * i *j));

				/* c -dj; */
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +8) = double2short( factor * cos(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +9) = double2short( -1*factor * sin(2.0 * PI / 64 * i *j));

				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +10) = double2short( factor * cos(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +11) = double2short( -1*factor * sin(2.0 * PI / 64 * i *j));

				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +12) = double2short( factor * cos(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +13) = double2short( -1*factor * sin(2.0 * PI / 64 * i *j));

				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +14) = double2short( factor * cos(2.0 * PI / 64 * i *j));
				*(WORD16 *)(ifft2048_r64twiddle + i*2*2*4*8 +2*2*4*j +15) = double2short( -1*factor * sin(2.0 * PI / 64 * i *j));

			}


			for (i=0;i<8;i++)
				for (j=0;j<4;j++)
				{
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +0) = double2short( factor * sin(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +1) = double2short( factor * cos(2.0 * PI / 32 * i *j));

					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +2) = double2short( factor * sin(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +3) = double2short( factor * cos(2.0 * PI / 32 * i *j));

					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +4) = double2short( factor * sin(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +5) = double2short( factor * cos(2.0 * PI / 32 * i *j));

					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +6) = double2short( factor * sin(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +7) = double2short( factor * cos(2.0 * PI / 32 * i *j));

					/* */

					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +8) = double2short( factor * cos(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +9) = double2short( -1*factor * sin(2.0 * PI / 32 * i *j));

					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +10) = double2short( factor * cos(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +11) = double2short( -1*factor * sin(2.0 * PI / 32 * i *j));

					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +12) = double2short( factor * cos(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +13) = double2short( -1*factor * sin(2.0 * PI / 32 * i *j));

					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +14) = double2short( factor * cos(2.0 * PI / 32 * i *j));
					*(WORD16 *)(ifft2048_r32twiddle + i*2*2*4*4 +2*2*4*j +15) = double2short( -1*factor * sin(2.0 * PI / 32 * i *j));

				}

}

#define MUL_CROSS(in_addr, twiddle_addr) \
{\
	__m128i * twiddleaddr_temp;\
	__m128i * in_addr_temp;\
	in_addr_temp = in_addr;\
	m128_t0 = _mm_load_si128((__m128i *)(in_addr_temp));\
	twiddleaddr_temp = twiddle_addr;\
	m128_t3 = _mm_load_si128((__m128i *)twiddleaddr_temp); \
	twiddleaddr_temp = twiddleaddr_temp + 1;\
	m128_t4 = _mm_load_si128((__m128i *)twiddleaddr_temp); \
	m128_t1 = _mm_madd_epi16(m128_t0, m128_t3);  \
	m128_t0 = _mm_madd_epi16(m128_t0, m128_t4); \
	m128_t0 = _mm_srli_si128(m128_t0,2);\
	m128_t1 = _mm_blend_epi16(m128_t1,m128_t0, 0x55); \
	_mm_store_si128((__m128i *)in_addr_temp, m128_t1);  \
}

#define DIV_4_2(in_addr) \
{\
	__m128i * in_addr_temp;\
	in_addr_temp = in_addr;\
	m128_t0 = _mm_load_si128((__m128i *)(in_addr_temp)); \
	_mm_store_si128((__m128i *)in_addr_temp, _mm_srai_epi16(m128_t0,1));  \
	in_addr_temp = in_addr_temp + 1;\
	m128_t0 = _mm_load_si128((__m128i *)(in_addr_temp)); \
	_mm_store_si128((__m128i *)in_addr_temp, _mm_srai_epi16(m128_t0,1));  \
	in_addr_temp = in_addr_temp + 1;\
	m128_t0 = _mm_load_si128((__m128i *)(in_addr_temp)); \
	_mm_store_si128((__m128i *)in_addr_temp, _mm_srai_epi16(m128_t0,1));  \
	in_addr_temp = in_addr_temp + 1;\
	m128_t0 = _mm_load_si128((__m128i *)(in_addr_temp)); \
	_mm_store_si128((__m128i *)in_addr_temp, _mm_srai_epi16(m128_t0,1));  \
	in_addr_temp = in_addr_temp + 1;\
}

#define radix4_register(in0,in1,in2,in3,out_addr,out_span) \
{\
	__m128i * in_addr_temp;\
	m128_t2 = _mm_adds_epi16(in0,in2); /*AA*/\
	m128_t6 = _mm_adds_epi16(in1,in3); /*CC*/\
	m128_t7=  _mm_adds_epi16(m128_t2,m128_t6);  /*out 1*/ \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6);  /*out 3*/ \
	m128_t6 =  m128_t7; /*out 1*/\
	\
	m128_t3 = _mm_subs_epi16(in0,in2); /*BB*/\
	m128_t12 = _mm_subs_epi16(in1,in3); /*DD*/\
	m128_t12 = _mm_shuffle_epi8(m128_t12, IQ_switch);\
	m128_t12 = _mm_sign_epi16(m128_t12, Neg_R);/*j*D*/\
	\
	m128_t7 = _mm_adds_epi16(m128_t3,m128_t12); /*out 2 */\
	m128_t3 = _mm_subs_epi16(m128_t3,m128_t12); /*out 4 */\
	/*m128_t6  m128_t7 m128_t2  m128_t3*/ \
	\
	_mm_store_si128((__m128i *)out_addr, m128_t6);  \
	out_addr = out_addr + out_span;\
	\
	_mm_store_si128((__m128i *)out_addr, m128_t7);  \
	out_addr = out_addr + out_span;\
	\
	_mm_store_si128((__m128i *)out_addr, m128_t2);  \
	out_addr = out_addr + out_span;\
	\
	_mm_store_si128((__m128i *)out_addr, m128_t3);  \
	\
}

#define radix8_0(in_addr,in_span,out_addr,out_span)\
{\
	__m128i * out_addr_temp1,*out_addr_temp2;\
	__m128i * in_addr_temp;\
	in_addr_temp = in_addr;\
	m128_t0 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span;\
	m128_t1 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span;\
	m128_t2 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span;\
	m128_t3 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span;\
	m128_t4 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span;\
	m128_t5 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span;\
	m128_t6 = _mm_load_si128((__m128i *)in_addr_temp);  in_addr_temp = in_addr_temp + in_span;\
	m128_t7 = _mm_load_si128((__m128i *)in_addr_temp);\
	\
	m128_t8 = _mm_adds_epi16(m128_t0, m128_t4); /* Temp8_0 */ \
	m128_t9 = _mm_subs_epi16(m128_t0, m128_t4); /* Temp8_1 */ \
	m128_t10 = _mm_adds_epi16(m128_t1, m128_t5);/* Temp8_2 */ \
	m128_t11 = _mm_subs_epi16(m128_t1, m128_t5);/* Temp8_3 */ \
	m128_t0 = _mm_adds_epi16(m128_t2, m128_t6); /* Temp8_4 */ \
	m128_t1 = _mm_subs_epi16(m128_t2, m128_t6);/* Temp8_5 */ \
	m128_t4 = _mm_adds_epi16(m128_t3, m128_t7);/* Temp8_6 */ \
	m128_t5 = _mm_subs_epi16(m128_t3, m128_t7);/* Temp8_7 */ \
	\
	/*x1(1+5)  = x1(1 +5) * (0 - i) = imag - real*j */\
	m128_t1 = _mm_shuffle_epi8(m128_t1, IQ_switch);\
	m128_t1 = _mm_sign_epi16(m128_t1, Neg_R); /* Temp8_5 */ \
	\
	m128_t2 = _mm_mulhrs_epi16(m128_t11, Const_0707); /*Temp8_3*/ \
	m128_t3 = _mm_mulhrs_epi16(m128_t5, Const_0707_Minus);  /*Temp8_7*/ \
	\
	m128_t6 = _mm_shuffle_epi8(m128_t2, IQ_switch); /*Temp8_3*/ \
	m128_t7 = _mm_adds_epi16(m128_t2,m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6);\
	m128_t11 = _mm_blend_epi16(m128_t2,m128_t7, 0xAA);\
	\
	m128_t6 = _mm_shuffle_epi8(m128_t3, IQ_switch); /*Temp8_7*/ \
	m128_t7 = _mm_adds_epi16(m128_t3,m128_t6); \
	m128_t3 = _mm_subs_epi16(m128_t3,m128_t6);\
	m128_t5 = _mm_blend_epi16(m128_t3,m128_t7, 0x55);\
	\
	\
	out_addr_temp1 = out_addr;\
	out_addr_temp2 = out_addr + out_span;\
	radix4_register(m128_t8,m128_t10,m128_t0,m128_t4, out_addr_temp1, 2*out_span);\
	radix4_register(m128_t9,m128_t11,m128_t1,m128_t5, out_addr_temp2, 2*out_span);\
	\
}

#define radix4_register_mul_zero(in0, in1, in2, in3, out_addr, out_span, twiddle_addr) \
{ \
	__m128i *in_addr_temp; \
	__m128i *twiddle_addr_temp; \
	m128_t2 = _mm_adds_epi16(in0,in2); /*AA*/ \
	m128_t6 = _mm_adds_epi16(in1,in3); /*CC*/ \
	m128_t7=  _mm_adds_epi16(m128_t2,m128_t6);  /*out 1*/ \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6);  /*out 3*/ \
	m128_t6 =  m128_t7; /*out 1*/ \
	 \
	m128_t3 = _mm_subs_epi16(in0,in2); /*BB*/ \
	m128_t12 = _mm_subs_epi16(in1,in3); /*DD*/ \
	m128_t12 = _mm_shuffle_epi8(m128_t12, IQ_switch); \
	m128_t12 = _mm_sign_epi16(m128_t12, Neg_R);/*j*D*/ \
	 \
	m128_t7 = _mm_adds_epi16(m128_t3,m128_t12); /*out 2 */ \
	m128_t3 = _mm_subs_epi16(m128_t3,m128_t12); /*out 4 */ \
	/*m128_t6  m128_t7 m128_t2  m128_t3*/ \
	 \
	 \
	_mm_store_si128((__m128i *)out_addr,  _mm_srai_epi16(m128_t6,1));   out_addr = out_addr + out_span; \
	 \
	twiddle_addr_temp = twiddle_addr + 2 * 2; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 3; \
	m128_t0 = _mm_madd_epi16(m128_t7, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t7, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr, m128_t0);   out_addr = out_addr + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 3; \
	m128_t0 = _mm_madd_epi16(m128_t2, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t2, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr, m128_t0);   out_addr = out_addr + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t3, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t3, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr, m128_t0); \
}

#define radix4_register_mul(in0, in1, in2, in3, out_addr, out_span, twiddle_addr) \
{ \
	__m128i *in_addr_temp; \
	__m128i *twiddle_addr_temp; \
	m128_t2 = _mm_adds_epi16(in0,in2); /*AA*/ \
	m128_t6 = _mm_adds_epi16(in1,in3); /*CC*/ \
	m128_t7=  _mm_adds_epi16(m128_t2,m128_t6);  /*out 1*/ \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6);  /*out 3*/ \
	m128_t6 =  m128_t7; /*out 1*/ \
	 \
	m128_t3 = _mm_subs_epi16(in0,in2); /*BB*/ \
	m128_t12 = _mm_subs_epi16(in1,in3); /*DD*/ \
	m128_t12 = _mm_shuffle_epi8(m128_t12, IQ_switch); \
	m128_t12 = _mm_sign_epi16(m128_t12, Neg_R);/*j*D*/ \
	 \
	m128_t7 = _mm_adds_epi16(m128_t3,m128_t12); /*out 2 */ \
	m128_t3 = _mm_subs_epi16(m128_t3,m128_t12); /*out 4 */ \
	/*m128_t6  m128_t7 m128_t2  m128_t3*/ \
	 \
	twiddle_addr_temp = twiddle_addr; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 3; \
	m128_t0 = _mm_madd_epi16(m128_t6, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t6, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr, m128_t0);   out_addr = out_addr + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 3; \
	m128_t0 = _mm_madd_epi16(m128_t7, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t7, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr, m128_t0);   out_addr = out_addr + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 3; \
	m128_t0 = _mm_madd_epi16(m128_t2, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t2, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr, m128_t0);   out_addr = out_addr + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t3, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t3, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr, m128_t0); \
}

#define radix8_0_mul(in_addr, in_span, out_addr, out_span, twiddle_addr) \
{ \
	__m128i * out_addr_temp1,*out_addr_temp2; \
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
	m128_t8 = _mm_adds_epi16(m128_t0, m128_t4); /* Temp8_0 */ \
	m128_t9 = _mm_subs_epi16(m128_t0, m128_t4); /* Temp8_1 */ \
	m128_t10 = _mm_adds_epi16(m128_t1, m128_t5);/* Temp8_2 */ \
	m128_t11 = _mm_subs_epi16(m128_t1, m128_t5);/* Temp8_3 */ \
	m128_t0 = _mm_adds_epi16(m128_t2, m128_t6); /* Temp8_4 */ \
	m128_t1 = _mm_subs_epi16(m128_t2, m128_t6);/* Temp8_5 */ \
	m128_t4 = _mm_adds_epi16(m128_t3, m128_t7);/* Temp8_6 */ \
	m128_t5 = _mm_subs_epi16(m128_t3, m128_t7);/* Temp8_7 */ \
	 \
	/*x1(1+5)  = x1(1 +5) * (0 - i) = imag - real*j */ \
	m128_t1 = _mm_shuffle_epi8(m128_t1, IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, Neg_R); /* Temp8_5 */ \
	 \
	m128_t2 = _mm_mulhrs_epi16(m128_t11, Const_0707); /*Temp8_3*/ \
	m128_t3 = _mm_mulhrs_epi16(m128_t5, Const_0707_Minus);  /*Temp8_7*/ \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t2, IQ_switch); /*Temp8_3*/ \
	m128_t7 = _mm_adds_epi16(m128_t2,m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6); \
	m128_t11 = _mm_blend_epi16(m128_t2,m128_t7, 0xAA); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t3, IQ_switch); /*Temp8_7*/ \
	m128_t7 = _mm_adds_epi16(m128_t3,m128_t6); \
	m128_t3 = _mm_subs_epi16(m128_t3,m128_t6); \
	m128_t5 = _mm_blend_epi16(m128_t3,m128_t7, 0x55); \
	 \
	 \
	out_addr_temp1 = out_addr; \
	out_addr_temp2 = out_addr + out_span; \
	radix4_register_mul_zero(m128_t8,m128_t10,m128_t0,m128_t4, out_addr_temp1, 2*out_span,twiddle_addr); \
	radix4_register_mul(m128_t9,m128_t11,m128_t1,m128_t5, out_addr_temp2, 2*out_span,twiddle_addr + 2); \
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
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); /*AA*/ \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); /*CC*/ \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); /*BB*/ \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); /*DD*/ \
	m128_t7 = _mm_shuffle_epi8(m128_t7, IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, Neg_R);/*j*D*/ \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t8 = _mm_srai_epi16(m128_t8,1); \
	m128_t9 = _mm_adds_epi16(m128_t6,m128_t7); \
	m128_t9 = _mm_srai_epi16(m128_t9,1); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t10 = _mm_srai_epi16(m128_t10,1); \
	m128_t11 = _mm_subs_epi16(m128_t6,m128_t7); \
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
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); /*AA*/ \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); /*CC*/ \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); /*BB*/ \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); /*DD*/ \
	m128_t7 = _mm_shuffle_epi8(m128_t7, IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, Neg_R);/*j*D*/ \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t9 = _mm_adds_epi16(m128_t6,m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t11 = _mm_subs_epi16(m128_t6,m128_t7); \
	twiddle_addr_temp = twiddle_addr + 2; \
	m128_t2 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t7 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
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

#define transpose64x4_mulzero(InBuf, in_span, OutBuf)\
{\
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
    WORD32 in_span_temp = 16 * in_span; \
	 \
	for (WORD32 ii=0;ii<64/4;ii++) \
	{ \
		in_addr_temp = InBuf + ii*in_span; \
		m128_t1 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span_temp; \
		m128_t3 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span_temp; \
		m128_t5 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span_temp; \
		m128_t7 = _mm_load_si128(in_addr_temp); \
		 \
		_MM_TRANSPOSE4_EPI32(m128_t1, m128_t3, m128_t5, m128_t7); \
		 \
		out_addr_temp = OutBuf + ii*4; \
		_mm_store_si128(out_addr_temp, m128_t1);  out_addr_temp = out_addr_temp + 1; \
		_mm_store_si128(out_addr_temp, m128_t3);  out_addr_temp = out_addr_temp + 1; \
		_mm_store_si128(out_addr_temp, m128_t5);  out_addr_temp = out_addr_temp + 1; \
		_mm_store_si128(out_addr_temp, m128_t7); \
	} \
}

#define transpose64x4(InBuf, in_span, OutBuf, twiddle_addr) \
{ \
	__m128i *twiddle_addr_temp; \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	 \
	for (WORD32 ii=0;ii<64/4;ii++) \
	{ \
		twiddle_addr_temp = twiddle_addr + ii * 2; \
		m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t1 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 31; \
		m128_t2 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 31; \
		m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 31; \
		m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
		m128_t7 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
		 \
		 \
		in_addr_temp = InBuf + ii*in_span; \
		m128_t8 = _mm_load_si128((__m128i *)in_addr_temp); in_addr_temp = in_addr_temp + 16*in_span; \
		m128_t9 = _mm_load_si128((__m128i *)in_addr_temp); in_addr_temp = in_addr_temp + 16*in_span; \
		m128_t10 = _mm_load_si128((__m128i *)in_addr_temp); in_addr_temp = in_addr_temp + 16*in_span; \
		m128_t11 = _mm_load_si128((__m128i *)in_addr_temp); \
		 \
		m128_t0 = _mm_madd_epi16(m128_t8, m128_t0); \
		m128_t1 = _mm_madd_epi16(m128_t8, m128_t1); \
		m128_t1 = _mm_srli_si128(m128_t1,2); \
		m128_t1 = _mm_blend_epi16(m128_t0,m128_t1, 0x55); \
		 \
		 \
		m128_t2 = _mm_madd_epi16(m128_t9, m128_t2); \
		m128_t3 = _mm_madd_epi16(m128_t9, m128_t3); \
		m128_t3 = _mm_srli_si128(m128_t3,2); \
		m128_t3 = _mm_blend_epi16(m128_t2,m128_t3, 0x55); \
		 \
		 \
		m128_t4 = _mm_madd_epi16(m128_t10, m128_t4); \
		m128_t5 = _mm_madd_epi16(m128_t10, m128_t5); \
		m128_t5 = _mm_srli_si128(m128_t5,2); \
		m128_t5 = _mm_blend_epi16(m128_t4,m128_t5, 0x55); \
		 \
		 \
		m128_t6 = _mm_madd_epi16(m128_t11, m128_t6); \
		m128_t7 = _mm_madd_epi16(m128_t11, m128_t7); \
		m128_t7 = _mm_srli_si128(m128_t7,2); \
		m128_t7 = _mm_blend_epi16(m128_t6,m128_t7, 0x55); \
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

#define radix64(InBuf ,in_span, OutBuf , out_span, ifft2048_r64twiddle)\
{\
	radix8_0(InBuf + 0*in_span, 8*in_span, ifft2048_Temp64_32_Buf +0*8,1);\
	radix8_0_mul(InBuf + 1*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 1*8, 1, ifft2048_r64twiddle + 2*4*1*2); \
	radix8_0_mul(InBuf + 2*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 2*8, 1, ifft2048_r64twiddle + 2*4*2*2); \
	radix8_0_mul(InBuf + 3*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 3*8, 1, ifft2048_r64twiddle + 2*4*3*2); \
	radix8_0_mul(InBuf + 4*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 4*8, 1, ifft2048_r64twiddle + 2*4*4*2); \
	radix8_0_mul(InBuf + 5*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 5*8, 1, ifft2048_r64twiddle + 2*4*5*2); \
	radix8_0_mul(InBuf + 6*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 6*8, 1, ifft2048_r64twiddle + 2*4*6*2); \
	radix8_0_mul(InBuf + 7*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 7*8, 1, ifft2048_r64twiddle + 2*4*7*2); \
	\
	DIV_4_2((__m128i *)(ifft2048_Temp64_32_Buf) + 0*4);\
	DIV_4_2((__m128i *)(ifft2048_Temp64_32_Buf) + 1*4);\
	\
	radix8_0(ifft2048_Temp64_32_Buf + 0, 8, OutBuf + 0*out_span, 8*out_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 1, 8, OutBuf + 1*out_span, 8*out_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 2, 8, OutBuf + 2*out_span, 8*out_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 3, 8, OutBuf + 3*out_span, 8*out_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 4, 8, OutBuf + 4*out_span, 8*out_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 5, 8, OutBuf + 5*out_span, 8*out_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 6, 8, OutBuf + 6*out_span, 8*out_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 7, 8, OutBuf + 7*out_span, 8*out_span);\
	\
}


#define radix32(InBuf,  in_span,  ifft2048_Temp64_32_Buf,  ifft2048_r32twiddle, OutBuf)\
{\
	radix4_0_zeromul(InBuf + 0*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 0*4); \
	radix4_0_mul(InBuf + 1*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 1*4, ifft2048_r32twiddle + 2*4*1); \
	radix4_0_mul(InBuf + 2*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 2*4, ifft2048_r32twiddle + 2*4*2); \
	radix4_0_mul(InBuf + 3*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 3*4, ifft2048_r32twiddle + 2*4*3); \
	radix4_0_mul(InBuf + 4*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 4*4, ifft2048_r32twiddle + 2*4*4); \
	radix4_0_mul(InBuf + 5*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 5*4, ifft2048_r32twiddle + 2*4*5); \
	radix4_0_mul(InBuf + 6*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 6*4, ifft2048_r32twiddle + 2*4*6); \
	radix4_0_mul(InBuf + 7*in_span, 8*in_span, ifft2048_Temp64_32_Buf + 7*4, ifft2048_r32twiddle + 2*4*7); \
	\
	radix8_0(ifft2048_Temp64_32_Buf + 0, 4, OutBuf + 0*in_span, 4*in_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 1, 4, OutBuf + 1*in_span, 4*in_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 2, 4, OutBuf + 2*in_span, 4*in_span);\
	radix8_0(ifft2048_Temp64_32_Buf + 3, 4, OutBuf + 3*in_span, 4*in_span);\
}



/** \fn void ifft2048(__m128i *InBuf, __m128i *OutBuf, __m128i *ifft2048_Temp64_32_Buf,  __m128i *ifft2048_r32twiddle, __m128i *ifft2048_r64twiddle, __m128i *ifft2048_r2048twiddle)
*  \brief 2048-point IFFT
*  \param InBuf [i] __m128i*  store one symbol's frequency domain data at 2048 frequency bins. Its size is 2048*2*2 bytes and the buffer is 16-byte aligned.
*  \param OutBuf [o] __m128i*  store one symbol's time domain data at 2048 time sample. Its size is 2048*2*2 bytes and the buffer is 16-byte aligned.
*  \param ifft2048_r32twiddle [i] __m128* constant table buffer. Its size is 32*2*16 bytes and the buffer is 16-byte aligned.
*  \param ifft2048_r64twiddle [i] __m128* constant table buffer. Its size is 64**2*16 bytes and the buffer is 16-byte aligned.
*  \param ifft2048_r2048twiddle [i] __m128* constant table buffer. Its size is 2048*2*16 bytes and the buffer is 16-byte aligned.
*/ 


void ifft2048(__m128i *InBuf, __m128i *OutBuf, __m128i *ifft2048_r32twiddle, __m128i *ifft2048_r64twiddle, __m128i *ifft2048_r2048twiddle)
{
    WORD32 i;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	__m128i OutTmp[512];
	__m128i ifft2048_Temp64_32_Buf[64];
	__m128i TransposeBuf[64];
    WORD32 in_span, out_span;

	in_span = 16;
	for (i=0;i<16;i++)
	{
		radix32((__m128i *)(InBuf) + i, in_span, (__m128i *)ifft2048_Temp64_32_Buf, (__m128i *)ifft2048_r32twiddle, (__m128i *)(OutTmp) + i);
	}

	DIV_4_2((__m128i *)(OutTmp) + 0);
	DIV_4_2((__m128i *)(OutTmp) + 4);
	DIV_4_2((__m128i *)(OutTmp) + 8);
	DIV_4_2((__m128i *)(OutTmp) + 12);
	for (i=1;i<4;i++)
	{
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 0, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 0) ); 
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 1, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 1) ); 
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 2, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 2) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 3, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 3) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 4, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 4) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 5, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 5) ); 
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 6, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 6) ); 
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 7, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 7) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 8, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 8) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 9, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 9) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 10,(__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 10) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 11, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 11) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 12, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 12) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 13, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 13) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 14,(__m128i *) ifft2048_r2048twiddle + 2*(i*16 + 14) );
		MUL_CROSS((__m128i *)(OutTmp) + i*16 + 15, (__m128i *)ifft2048_r2048twiddle + 2*(i*16 + 15) );
	}

	in_span = 1;
	out_span = 8;
	transpose64x4_mulzero((__m128i *)(OutTmp) + 0*64, in_span, (__m128i *)TransposeBuf);
	radix64((__m128i *)TransposeBuf, in_span,(__m128i *)OutBuf + 0, out_span, (__m128i *)ifft2048_r64twiddle);
	for (i=1;i<8;i++)
	{
		transpose64x4((__m128i *)(OutTmp) + i*64, in_span, (__m128i *)TransposeBuf, (__m128i *)ifft2048_r2048twiddle + 2*i*64);
		radix64((__m128i *)TransposeBuf, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)ifft2048_r64twiddle);
	}


}


void swap(Complex *a, Complex *b)
{
#ifdef DEBUG
	__m128i *tempa =(__m128i *)(a);
	__m128i *tempb =(__m128i *)(b);

	__m128i temp1,temp2;

	temp1 = _mm_load_si128(tempa);
	temp2 = _mm_load_si128(tempb);
	_mm_store_si128(tempa,temp2);
	_mm_store_si128(tempb,temp1);
	++tempa;
	++tempb;

	temp1 = _mm_load_si128(tempa);
	temp2 = _mm_load_si128(tempb);
	_mm_store_si128(tempa,temp2);
	_mm_store_si128(tempb,temp1);
	++tempa;
	++tempb;

	temp1 = _mm_load_si128(tempa);
	temp2 = _mm_load_si128(tempb);
	_mm_store_si128(tempa,temp2);
	_mm_store_si128(tempb,temp1);
	++tempa;
	++tempb;

	temp1 = _mm_load_si128(tempa);
	temp2 = _mm_load_si128(tempb);
	_mm_store_si128(tempa,temp2);
	_mm_store_si128(tempb,temp1);

#else
	__m128i tempa ;
	tempa = *((__m128i*)a);
	*((__m128i*)a) = *((__m128i*)b);
	*((__m128i*)b) = tempa;

	tempa = *((__m128i*)(a + 4));
	*((__m128i*)(a + 4)) = *((__m128i*)(b + 4));	
	*((__m128i*)(b + 4)) = tempa;

	tempa = *((__m128i*)(a + 8));
	*((__m128i*)(a + 8)) = *((__m128i*)(b + 8));	
	*((__m128i*)(b + 8)) = tempa;

	tempa = *((__m128i*)(a + 12));
	*((__m128i*)(a + 12)) = *((__m128i*)(b + 12));	
	*((__m128i*)(b + 12)) = tempa;

#endif
}

void fftshift2048(Complex *pSymbol)
{
	for(WORD32 i = 0; i < 1024; i=i+16)
	{
		swap(pSymbol + i, pSymbol + i + 1024);
	}
}
void fftshift1024(Complex *pSymbol)
{
	for(WORD32 i = 0; i < 512; i = i + 16)
	{
		swap(pSymbol + i, pSymbol + i + 512);
	}
}
