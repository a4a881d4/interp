#include <stdio.h>
#include <math.h>

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE 2 
#include <pmmintrin.h> // SSE 3
#include <tmmintrin.h> // SSSE 3
#include <smmintrin.h> // SSE 4 for media

#include "common_structure.h"

#ifndef PI
#define PI (3.14159265358979323846)
#endif

const static __m128i DFT_IQ_switch =  _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
const static __m128i DFT_Neg_I = _mm_setr_epi8(0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF );
const static __m128i DFT_Neg_R = _mm_setr_epi8(0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1);
const static __m128i DFT_Const_0707 = _mm_setr_epi8(0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A,0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A, 0x7F, 0x5A);
const static __m128i DFT_Const_0707_Minus = _mm_setr_epi8(0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5,0x81, 0xA5); 
const static __m128i DFT_ConstMinus08660 = _mm_setr_epi8(0x27, 0x91, 0x27, 0x91, 0x27, 0x91, 0x27, 0x91, 0x27, 0x91, 0x27, 0x91, 0x27, 0x91, 0x27, 0x91);
const static __m128i DFT_Const0500_0866_0 = _mm_setr_epi8(0x00, 0x40, 0x00, 0x40, 0x00, 0x40, 0x00, 0x40, 0x00, 0x40, 0x00, 0x40, 0x00, 0x40, 0x00, 0x40); /* c c 0.5*/
const static __m128i DFT_Const0500_0866_1 = _mm_setr_epi8(0xD9, 0x6E, 0x27, 0x91,0xD9, 0x6E, 0x27, 0x91,0xD9, 0x6E, 0x27, 0x91,0xD9, 0x6E, 0x27, 0x91); /* -d  d  -0.86 0.86*/ /*diff*/
const static __m128i DFT_Const0500_0866_2 = _mm_setr_epi8(0x00, 0xC0,0x00, 0xC0, 0x00, 0xC0,0x00, 0xC0, 0x00, 0xC0,0x00, 0xC0 ,0x00, 0xC0,0x00, 0xC0) /*c -0.5*/;
const static __m128i DFT_Const03090 = _mm_setr_epi8(0x8D, 0x27, 0x8D, 0x27, 0x8D, 0x27, 0x8D, 0x27, 0x8D, 0x27, 0x8D, 0x27, 0x8D, 0x27, 0x8D, 0x27);
const static __m128i DFT_Const08090 = _mm_setr_epi8(0x8D, 0x67, 0x8D, 0x67, 0x8D, 0x67, 0x8D, 0x67, 0x8D, 0x67, 0x8D, 0x67, 0x8D, 0x67, 0x8D, 0x67);
const static __m128i DFT_ConstMinus09511 = _mm_setr_epi8(0x43, 0x86, 0x43, 0x86, 0x43, 0x86, 0x43, 0x86, 0x43, 0x86, 0x43, 0x86, 0x43, 0x86, 0x43, 0x86);
const static __m128i DFT_ConstMinus05878 = _mm_setr_epi8(0xC4, 0xB4, 0xC4, 0xB4, 0xC4, 0xB4, 0xC4, 0xB4, 0xC4, 0xB4, 0xC4, 0xB4, 0xC4, 0xB4, 0xC4, 0xB4);
const static __m128i DFT_Const_07660 = _mm_setr_epi8(0x0C, 0x62, 0x0C, 0x62, 0x0C, 0x62, 0x0C, 0x62, 0x0C, 0x62, 0x0C, 0x62, 0x0C, 0x62, 0x0C, 0x62);/*c, c*/
const static __m128i DFT_Const_Minus06428 = _mm_setr_epi8(0xB9, 0xAD, 0x47, 0x52, 0xB9, 0xAD, 0x47, 0x52, 0xB9, 0xAD, 0x47, 0x52, 0xB9, 0xAD, 0x47, 0x52);/*d, -d*/
const static __m128i DFT_Const_01736 = _mm_setr_epi8(0x39, 0x16, 0x39, 0x16, 0x39, 0x16, 0x39, 0x16, 0x39, 0x16, 0x39, 0x16, 0x39, 0x16, 0x39, 0x16);/*c, c*/
const static __m128i DFT_Const_Minus09848 = _mm_setr_epi8(0xF2, 0x81, 0x0E, 0x7E, 0xF2, 0x81, 0x0E, 0x7E, 0xF2, 0x81, 0x0E, 0x7E, 0xF2, 0x81, 0x0E, 0x7E);/*d, -d*/
const static __m128i DFT_Const_Minus09397 = _mm_setr_epi8(0xB8, 0x87, 0xB8, 0x87, 0xB8, 0x87, 0xB8, 0x87, 0xB8, 0x87, 0xB8, 0x87, 0xB8, 0x87, 0xB8, 0x87);/*c, c*/
const static __m128i DFT_Const_Minus03420 = _mm_setr_epi8(0x39, 0xD4, 0xC7, 0x2B, 0x39, 0xD4, 0xC7, 0x2B, 0x39, 0xD4, 0xC7, 0x2B, 0x39, 0xD4, 0xC7, 0x2B);/*d, -d*/

static WORD16 double2short(DOUBLE64 d)
{
    d = floor(0.5 + d);
    if (d >= 32767)    return 32767;
    if (d < -32768)     return -32768;
    return (WORD16)d;
}

void init_dft12_twiddle_factor(WORD16 *r12twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0; i<3; ++i)
		for (j=0; j<4; ++j)
        	{
                /* d +cj */
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +0) = double2short( factor * sin(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +1) = double2short( factor * cos(-2.0 * PI / 12 * i *j));

                *(WORD16 *)(r12twiddle + i*16*4 +16*j +2) = double2short( factor * sin(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +3) = double2short( factor * cos(-2.0 * PI / 12 * i *j));

                *(WORD16 *)(r12twiddle + i*16*4 +16*j +4) = double2short( factor * sin(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +5) = double2short( factor * cos(-2.0 * PI / 12 * i *j));

                *(WORD16 *)(r12twiddle + i*16*4 +16*j +6) = double2short( factor * sin(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +7) = double2short( factor * cos(-2.0 * PI / 12 * i *j));

                /* c -dj; */
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +8) = double2short( factor * cos(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 12 * i *j));

                *(WORD16 *)(r12twiddle + i*16*4 +16*j +10) = double2short( factor * cos(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 12 * i *j));

                *(WORD16 *)(r12twiddle + i*16*4 +16*j +12) = double2short( factor * cos(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 12 * i *j));

                *(WORD16 *)(r12twiddle + i*16*4 +16*j +14) = double2short( factor * cos(-2.0 * PI / 12 * i *j));
                *(WORD16 *)(r12twiddle + i*16*4 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 12 * i *j));
        	}
}

void init_dft15_twiddle_factor(WORD16 *r15twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

    	for (i=0; i<3; ++i)
        	for (j=0; j<5; ++j)
        	{
            	/* d +cj */
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +0) = double2short( factor * sin(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +1) = double2short( factor * cos(-2.0 * PI / 15 * i *j));

                *(WORD16 *)(r15twiddle + i*16*5 +16*j +2) = double2short( factor * sin(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +3) = double2short( factor * cos(-2.0 * PI / 15 * i *j));

                *(WORD16 *)(r15twiddle + i*16*5 +16*j +4) = double2short( factor * sin(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +5) = double2short( factor * cos(-2.0 * PI / 15 * i *j));

                *(WORD16 *)(r15twiddle + i*16*5 +16*j +6) = double2short( factor * sin(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +7) = double2short( factor * cos(-2.0 * PI / 15 * i *j));

                /* c -dj; */
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +8) = double2short( factor * cos(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 15 * i *j));

                *(WORD16 *)(r15twiddle + i*16*5 +16*j +10) = double2short( factor * cos(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 15 * i *j));

                *(WORD16 *)(r15twiddle + i*16*5 +16*j +12) = double2short( factor * cos(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 15 * i *j));

                *(WORD16 *)(r15twiddle + i*16*5 +16*j +14) = double2short( factor * cos(-2.0 * PI / 15 * i *j));
                *(WORD16 *)(r15twiddle + i*16*5 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 15 * i *j));
        	}
}

void init_dft16_twiddle_factor(WORD16 *r16twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

    	for (i=0; i<4; ++i)
        	for (j=0; j<4; ++j)
        	{
            	/* d +cj */
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +0) = double2short( factor * sin(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +1) = double2short( factor * cos(-2.0 * PI / 16 * i *j));

                *(WORD16 *)(r16twiddle + i*16*4 +16*j +2) = double2short( factor * sin(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +3) = double2short( factor * cos(-2.0 * PI / 16 * i *j));

                *(WORD16 *)(r16twiddle + i*16*4 +16*j +4) = double2short( factor * sin(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +5) = double2short( factor * cos(-2.0 * PI / 16 * i *j));

                *(WORD16 *)(r16twiddle + i*16*4 +16*j +6) = double2short( factor * sin(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +7) = double2short( factor * cos(-2.0 * PI / 16 * i *j));

                /* c -dj; */
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +8) = double2short( factor * cos(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 16 * i *j));

                *(WORD16 *)(r16twiddle + i*16*4 +16*j +10) = double2short( factor * cos(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 16 * i *j));

                *(WORD16 *)(r16twiddle + i*16*4 +16*j +12) = double2short( factor * cos(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 16 * i *j));

                *(WORD16 *)(r16twiddle + i*16*4 +16*j +14) = double2short( factor * cos(-2.0 * PI / 16 * i *j));
                *(WORD16 *)(r16twiddle + i*16*4 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 16 * i *j));
        	}
}

void init_dft18_twiddle_factor(WORD16 *r18twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

    	for (i=0; i<3; ++i)
        	for (j=0; j<6; ++j)
        	{
            	/* d +cj */
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +0) = double2short( factor * sin(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +1) = double2short( factor * cos(-2.0 * PI / 18 * i *j));

                *(WORD16 *)(r18twiddle + i*16*6 +16*j +2) = double2short( factor * sin(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +3) = double2short( factor * cos(-2.0 * PI / 18 * i *j));

                *(WORD16 *)(r18twiddle + i*16*6 +16*j +4) = double2short( factor * sin(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +5) = double2short( factor * cos(-2.0 * PI / 18 * i *j));

                *(WORD16 *)(r18twiddle + i*16*6 +16*j +6) = double2short( factor * sin(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +7) = double2short( factor * cos(-2.0 * PI / 18 * i *j));

                /* c -dj; */
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +8) = double2short( factor * cos(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 18 * i *j));

                *(WORD16 *)(r18twiddle + i*16*6 +16*j +10) = double2short( factor * cos(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 18 * i *j));

                *(WORD16 *)(r18twiddle + i*16*6 +16*j +12) = double2short( factor * cos(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 18 * i *j));

                *(WORD16 *)(r18twiddle + i*16*6 +16*j +14) = double2short( factor * cos(-2.0 * PI / 18 * i *j));
                *(WORD16 *)(r18twiddle + i*16*6 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 18 * i *j));
        	}
}

void init_dft20_twiddle_factor(WORD16 *r20twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

    	for (i=0; i<4; ++i)
        	for (j=0; j<5; ++j)
        	{
            	/* d +cj */
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +0) = double2short( factor * sin(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +1) = double2short( factor * cos(-2.0 * PI / 20 * i *j));

                *(WORD16 *)(r20twiddle + i*16*5 +16*j +2) = double2short( factor * sin(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +3) = double2short( factor * cos(-2.0 * PI / 20 * i *j));

                *(WORD16 *)(r20twiddle + i*16*5 +16*j +4) = double2short( factor * sin(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +5) = double2short( factor * cos(-2.0 * PI / 20 * i *j));

                *(WORD16 *)(r20twiddle + i*16*5 +16*j +6) = double2short( factor * sin(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +7) = double2short( factor * cos(-2.0 * PI / 20 * i *j));

                /* c -dj; */
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +8) = double2short( factor * cos(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 20 * i *j));

                *(WORD16 *)(r20twiddle + i*16*5 +16*j +10) = double2short( factor * cos(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 20 * i *j));

                *(WORD16 *)(r20twiddle + i*16*5 +16*j +12) = double2short( factor * cos(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 20 * i *j));

                *(WORD16 *)(r20twiddle + i*16*5 +16*j +14) = double2short( factor * cos(-2.0 * PI / 20 * i *j));
                *(WORD16 *)(r20twiddle + i*16*5 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 20 * i *j));
        	}
}

void init_dft24_twiddle_factor(WORD16 *r24twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

    	for (i=0; i<4; ++i)
        	for (j=0; j<6; ++j)
        	{
            	/* d +cj */
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +0) = double2short( factor * sin(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +1) = double2short( factor * cos(-2.0 * PI / 24 * i *j));

                *(WORD16 *)(r24twiddle + i*16*6 +16*j +2) = double2short( factor * sin(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +3) = double2short( factor * cos(-2.0 * PI / 24 * i *j));

                *(WORD16 *)(r24twiddle + i*16*6 +16*j +4) = double2short( factor * sin(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +5) = double2short( factor * cos(-2.0 * PI / 24 * i *j));

                *(WORD16 *)(r24twiddle + i*16*6 +16*j +6) = double2short( factor * sin(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +7) = double2short( factor * cos(-2.0 * PI / 24 * i *j));

                /* c -dj; */
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +8) = double2short( factor * cos(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 24 * i *j));

                *(WORD16 *)(r24twiddle + i*16*6 +16*j +10) = double2short( factor * cos(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 24 * i *j));

                *(WORD16 *)(r24twiddle + i*16*6 +16*j +12) = double2short( factor * cos(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 24 * i *j));

                *(WORD16 *)(r24twiddle + i*16*6 +16*j +14) = double2short( factor * cos(-2.0 * PI / 24 * i *j));
                *(WORD16 *)(r24twiddle + i*16*6 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 24 * i *j));
        	}
}

void init_dft25_twiddle_factor(WORD16 *r25twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

    	for (i=0; i<5; ++i)
        	for (j=0; j<5; ++j)
        	{
            	/* d +cj */
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +0) = double2short( factor * sin(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +1) = double2short( factor * cos(-2.0 * PI / 25 * i *j));

                *(WORD16 *)(r25twiddle + i*16*5 +16*j +2) = double2short( factor * sin(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +3) = double2short( factor * cos(-2.0 * PI / 25 * i *j));

                *(WORD16 *)(r25twiddle + i*16*5 +16*j +4) = double2short( factor * sin(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +5) = double2short( factor * cos(-2.0 * PI / 25 * i *j));

                *(WORD16 *)(r25twiddle + i*16*5 +16*j +6) = double2short( factor * sin(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +7) = double2short( factor * cos(-2.0 * PI / 25 * i *j));

                /* c -dj; */
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +8) = double2short( factor * cos(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 25 * i *j));

                *(WORD16 *)(r25twiddle + i*16*5 +16*j +10) = double2short( factor * cos(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 25 * i *j));

                *(WORD16 *)(r25twiddle + i*16*5 +16*j +12) = double2short( factor * cos(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 25 * i *j));

                *(WORD16 *)(r25twiddle + i*16*5 +16*j +14) = double2short( factor * cos(-2.0 * PI / 25 * i *j));
                *(WORD16 *)(r25twiddle + i*16*5 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 25 * i *j));
        	}
}

void init_dft27_twiddle_factor(WORD16 *r27twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

    	for (i=0; i<3; ++i)
        	for (j=0; j<9; ++j)
        	{
            	/* d +cj */
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +0) = double2short( factor * sin(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +1) = double2short( factor * cos(-2.0 * PI / 27 * i *j));

                *(WORD16 *)(r27twiddle + i*16*9 +16*j +2) = double2short( factor * sin(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +3) = double2short( factor * cos(-2.0 * PI / 27 * i *j));

                *(WORD16 *)(r27twiddle + i*16*9 +16*j +4) = double2short( factor * sin(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +5) = double2short( factor * cos(-2.0 * PI / 27 * i *j));

                *(WORD16 *)(r27twiddle + i*16*9 +16*j +6) = double2short( factor * sin(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +7) = double2short( factor * cos(-2.0 * PI / 27 * i *j));

                /* c -dj; */
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +8) = double2short( factor * cos(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 27 * i *j));

                *(WORD16 *)(r27twiddle + i*16*9 +16*j +10) = double2short( factor * cos(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 27 * i *j));

                *(WORD16 *)(r27twiddle + i*16*9 +16*j +12) = double2short( factor * cos(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 27 * i *j));

                *(WORD16 *)(r27twiddle + i*16*9 +16*j +14) = double2short( factor * cos(-2.0 * PI / 27 * i *j));
                *(WORD16 *)(r27twiddle + i*16*9 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 27 * i *j));
        	}
}

void init_dft30_twiddle_factor(WORD16 *r30twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<5;i++)
		for (j=0;j<6;j++)
		{
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +0) = double2short( factor * sin(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +1) = double2short( factor * cos(-2.0 * PI / 30 * i *j));

			*(WORD16 *)(r30twiddle + i*16*6 +16*j +2) = double2short( factor * sin(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +3) = double2short( factor * cos(-2.0 * PI / 30 * i *j));

			*(WORD16 *)(r30twiddle + i*16*6 +16*j +4) = double2short( factor * sin(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +5) = double2short( factor * cos(-2.0 * PI / 30 * i *j));

			*(WORD16 *)(r30twiddle + i*16*6 +16*j +6) = double2short( factor * sin(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +7) = double2short( factor * cos(-2.0 * PI / 30 * i *j));

			/* c -dj; */
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +8) = double2short( factor * cos(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 30 * i *j));

			*(WORD16 *)(r30twiddle + i*16*6 +16*j +10) = double2short( factor * cos(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 30 * i *j));

			*(WORD16 *)(r30twiddle + i*16*6 +16*j +12) = double2short( factor * cos(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 30 * i *j));

			*(WORD16 *)(r30twiddle + i*16*6 +16*j +14) = double2short( factor * cos(-2.0 * PI / 30 * i *j));
			*(WORD16 *)(r30twiddle + i*16*6 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 30 * i *j));

		}
}

void init_dft32_twiddle_factor(WORD16 *r32twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<4;i++)
		for (j=0;j<8;j++)
		{
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +0) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +1) = double2short( factor * cos(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*16*8 +16*j +2) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +3) = double2short( factor * cos(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*16*8 +16*j +4) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +5) = double2short( factor * cos(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*16*8 +16*j +6) = double2short( factor * sin(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +7) = double2short( factor * cos(-2.0 * PI / 32 * i *j));

			/* c -dj; */
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +8) = double2short( factor * cos(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*16*8 +16*j +10) = double2short( factor * cos(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*16*8 +16*j +12) = double2short( factor * cos(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 32 * i *j));

			*(WORD16 *)(r32twiddle + i*16*8 +16*j +14) = double2short( factor * cos(-2.0 * PI / 32 * i *j));
			*(WORD16 *)(r32twiddle + i*16*8 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 32 * i *j));

		}
}

void init_dft36_twiddle_factor(WORD16 *r36twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<6;i++)
		for (j=0;j<6;j++)
		{
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +0) = double2short( factor * sin(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +1) = double2short( factor * cos(-2.0 * PI / 36 * i *j));

			*(WORD16 *)(r36twiddle + i*16*6 +16*j +2) = double2short( factor * sin(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +3) = double2short( factor * cos(-2.0 * PI / 36 * i *j));

			*(WORD16 *)(r36twiddle + i*16*6 +16*j +4) = double2short( factor * sin(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +5) = double2short( factor * cos(-2.0 * PI / 36 * i *j));

			*(WORD16 *)(r36twiddle + i*16*6 +16*j +6) = double2short( factor * sin(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +7) = double2short( factor * cos(-2.0 * PI / 36 * i *j));

			/* c -dj; */
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +8) = double2short( factor * cos(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 36 * i *j));

			*(WORD16 *)(r36twiddle + i*16*6 +16*j +10) = double2short( factor * cos(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 36 * i *j));

			*(WORD16 *)(r36twiddle + i*16*6 +16*j +12) = double2short( factor * cos(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 36 * i *j));

			*(WORD16 *)(r36twiddle + i*16*6 +16*j +14) = double2short( factor * cos(-2.0 * PI / 36 * i *j));
			*(WORD16 *)(r36twiddle + i*16*6 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 36 * i *j));

		}
}

void init_dft40_twiddle_factor(WORD16 *r40twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<5;i++)
		for (j=0;j<8;j++)
		{
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +0) = double2short( factor * sin(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +1) = double2short( factor * cos(-2.0 * PI / 40 * i *j));

			*(WORD16 *)(r40twiddle + i*16*8 +16*j +2) = double2short( factor * sin(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +3) = double2short( factor * cos(-2.0 * PI / 40 * i *j));

			*(WORD16 *)(r40twiddle + i*16*8 +16*j +4) = double2short( factor * sin(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +5) = double2short( factor * cos(-2.0 * PI / 40 * i *j));

			*(WORD16 *)(r40twiddle + i*16*8 +16*j +6) = double2short( factor * sin(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +7) = double2short( factor * cos(-2.0 * PI / 40 * i *j));

			/* c -dj; */
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +8) = double2short( factor * cos(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 40 * i *j));

			*(WORD16 *)(r40twiddle + i*16*8 +16*j +10) = double2short( factor * cos(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 40 * i *j));

			*(WORD16 *)(r40twiddle + i*16*8 +16*j +12) = double2short( factor * cos(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 40 * i *j));

			*(WORD16 *)(r40twiddle + i*16*8 +16*j +14) = double2short( factor * cos(-2.0 * PI / 40 * i *j));
			*(WORD16 *)(r40twiddle + i*16*8 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 40 * i *j));

		}
}

void init_dft48_twiddle_factor(WORD16 *r48twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<6;i++)
		for (j=0;j<8;j++)
		{
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +0) = double2short( factor * sin(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +1) = double2short( factor * cos(-2.0 * PI / 48 * i *j));

			*(WORD16 *)(r48twiddle + i*16*8 +16*j +2) = double2short( factor * sin(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +3) = double2short( factor * cos(-2.0 * PI / 48 * i *j));

			*(WORD16 *)(r48twiddle + i*16*8 +16*j +4) = double2short( factor * sin(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +5) = double2short( factor * cos(-2.0 * PI / 48 * i *j));

			*(WORD16 *)(r48twiddle + i*16*8 +16*j +6) = double2short( factor * sin(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +7) = double2short( factor * cos(-2.0 * PI / 48 * i *j));

			/* c -dj; */
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +8) = double2short( factor * cos(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 48 * i *j));

			*(WORD16 *)(r48twiddle + i*16*8 +16*j +10) = double2short( factor * cos(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 48 * i *j));

			*(WORD16 *)(r48twiddle + i*16*8 +16*j +12) = double2short( factor * cos(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 48 * i *j));

			*(WORD16 *)(r48twiddle + i*16*8 +16*j +14) = double2short( factor * cos(-2.0 * PI / 48 * i *j));
			*(WORD16 *)(r48twiddle + i*16*8 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 48 * i *j));

		}
}

void init_dft648_twiddle_factor(WORD16 *r648twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<27;i++)
		for (j=0;j<24;j++)
		{
			/* d +cj */
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +0) = double2short( factor * sin(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +1) = double2short( factor * cos(-2.0 * PI / 648 * i *j));

			*(WORD16 *)(r648twiddle + i*16*24 +16*j +2) = double2short( factor * sin(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +3) = double2short( factor * cos(-2.0 * PI / 648 * i *j));

			*(WORD16 *)(r648twiddle + i*16*24 +16*j +4) = double2short( factor * sin(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +5) = double2short( factor * cos(-2.0 * PI / 648 * i *j));

			*(WORD16 *)(r648twiddle + i*16*24 +16*j +6) = double2short( factor * sin(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +7) = double2short( factor * cos(-2.0 * PI / 648 * i *j));

			/* c -dj; */
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +8) = double2short( factor * cos(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 648 * i *j));

			*(WORD16 *)(r648twiddle + i*16*24 +16*j +10) = double2short( factor * cos(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 648 * i *j));

			*(WORD16 *)(r648twiddle + i*16*24 +16*j +12) = double2short( factor * cos(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 648 * i *j));

			*(WORD16 *)(r648twiddle + i*16*24 +16*j +14) = double2short( factor * cos(-2.0 * PI / 648 * i *j));
			*(WORD16 *)(r648twiddle + i*16*24 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 648 * i *j));

		}
}

void init_dft720_twiddle_factor(WORD16 *r720twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<36;i++)
		for (j=0;j<20;j++)
		{
			/* d +cj */
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +0) = double2short( factor * sin(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +1) = double2short( factor * cos(-2.0 * PI / 720 * i *j));

			*(WORD16 *)(r720twiddle + i*16*20 +16*j +2) = double2short( factor * sin(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +3) = double2short( factor * cos(-2.0 * PI / 720 * i *j));

			*(WORD16 *)(r720twiddle + i*16*20 +16*j +4) = double2short( factor * sin(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +5) = double2short( factor * cos(-2.0 * PI / 720 * i *j));

			*(WORD16 *)(r720twiddle + i*16*20 +16*j +6) = double2short( factor * sin(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +7) = double2short( factor * cos(-2.0 * PI / 720 * i *j));

			/* c -dj; */
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +8) = double2short( factor * cos(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 720 * i *j));

			*(WORD16 *)(r720twiddle + i*16*20 +16*j +10) = double2short( factor * cos(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 720 * i *j));

			*(WORD16 *)(r720twiddle + i*16*20 +16*j +12) = double2short( factor * cos(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 720 * i *j));

			*(WORD16 *)(r720twiddle + i*16*20 +16*j +14) = double2short( factor * cos(-2.0 * PI / 720 * i *j));
			*(WORD16 *)(r720twiddle + i*16*20 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 720 * i *j));

		}
}

void init_dft768_twiddle_factor(WORD16 *r768twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<48;i++)
		for (j=0;j<16;j++)
		{
			/* d +cj */
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +0) = double2short( factor * sin(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +1) = double2short( factor * cos(-2.0 * PI / 768 * i *j));

			*(WORD16 *)(r768twiddle + i*16*16 +16*j +2) = double2short( factor * sin(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +3) = double2short( factor * cos(-2.0 * PI / 768 * i *j));

			*(WORD16 *)(r768twiddle + i*16*16 +16*j +4) = double2short( factor * sin(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +5) = double2short( factor * cos(-2.0 * PI / 768 * i *j));

			*(WORD16 *)(r768twiddle + i*16*16 +16*j +6) = double2short( factor * sin(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +7) = double2short( factor * cos(-2.0 * PI / 768 * i *j));

			/* c -dj; */
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +8) = double2short( factor * cos(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 768 * i *j));

			*(WORD16 *)(r768twiddle + i*16*16 +16*j +10) = double2short( factor * cos(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 768 * i *j));

			*(WORD16 *)(r768twiddle + i*16*16 +16*j +12) = double2short( factor * cos(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 768 * i *j));

			*(WORD16 *)(r768twiddle + i*16*16 +16*j +14) = double2short( factor * cos(-2.0 * PI / 768 * i *j));
			*(WORD16 *)(r768twiddle + i*16*16 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 768 * i *j));

		}
}

void init_dft864_twiddle_factor(WORD16 *r864twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<36;i++)
		for (j=0;j<24;j++)
		{
			/* d +cj */
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +0) = double2short( factor * sin(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +1) = double2short( factor * cos(-2.0 * PI / 864 * i *j));

			*(WORD16 *)(r864twiddle + i*16*24 +16*j +2) = double2short( factor * sin(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +3) = double2short( factor * cos(-2.0 * PI / 864 * i *j));

			*(WORD16 *)(r864twiddle + i*16*24 +16*j +4) = double2short( factor * sin(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +5) = double2short( factor * cos(-2.0 * PI / 864 * i *j));

			*(WORD16 *)(r864twiddle + i*16*24 +16*j +6) = double2short( factor * sin(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +7) = double2short( factor * cos(-2.0 * PI / 864 * i *j));

			/* c -dj; */
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +8) = double2short( factor * cos(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 864 * i *j));

			*(WORD16 *)(r864twiddle + i*16*24 +16*j +10) = double2short( factor * cos(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 864 * i *j));

			*(WORD16 *)(r864twiddle + i*16*24 +16*j +12) = double2short( factor * cos(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 864 * i *j));

			*(WORD16 *)(r864twiddle + i*16*24 +16*j +14) = double2short( factor * cos(-2.0 * PI / 864 * i *j));
			*(WORD16 *)(r864twiddle + i*16*24 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 864 * i *j));

		}
}

void init_dft900_twiddle_factor(WORD16 *r900twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<36;i++)
		for (j=0;j<25;j++)
		{
			/* d +cj */
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +0) = double2short( factor * sin(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +1) = double2short( factor * cos(-2.0 * PI / 900 * i *j));

			*(WORD16 *)(r900twiddle + i*16*25 +16*j +2) = double2short( factor * sin(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +3) = double2short( factor * cos(-2.0 * PI / 900 * i *j));

			*(WORD16 *)(r900twiddle + i*16*25 +16*j +4) = double2short( factor * sin(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +5) = double2short( factor * cos(-2.0 * PI / 900 * i *j));

			*(WORD16 *)(r900twiddle + i*16*25 +16*j +6) = double2short( factor * sin(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +7) = double2short( factor * cos(-2.0 * PI / 900 * i *j));

			/* c -dj; */
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +8) = double2short( factor * cos(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 900 * i *j));

			*(WORD16 *)(r900twiddle + i*16*25 +16*j +10) = double2short( factor * cos(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 900 * i *j));

			*(WORD16 *)(r900twiddle + i*16*25 +16*j +12) = double2short( factor * cos(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 900 * i *j));

			*(WORD16 *)(r900twiddle + i*16*25 +16*j +14) = double2short( factor * cos(-2.0 * PI / 900 * i *j));
			*(WORD16 *)(r900twiddle + i*16*25 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 900 * i *j));

		}
}

void init_dft960_twiddle_factor(WORD16 *r960twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<40;i++)
		for (j=0;j<24;j++)
		{
			/* d +cj */
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +0) = double2short( factor * sin(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +1) = double2short( factor * cos(-2.0 * PI / 960 * i *j));

			*(WORD16 *)(r960twiddle + i*16*24 +16*j +2) = double2short( factor * sin(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +3) = double2short( factor * cos(-2.0 * PI / 960 * i *j));

			*(WORD16 *)(r960twiddle + i*16*24 +16*j +4) = double2short( factor * sin(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +5) = double2short( factor * cos(-2.0 * PI / 960 * i *j));

			*(WORD16 *)(r960twiddle + i*16*24 +16*j +6) = double2short( factor * sin(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +7) = double2short( factor * cos(-2.0 * PI / 960 * i *j));

			/* c -dj; */
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +8) = double2short( factor * cos(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 960 * i *j));

			*(WORD16 *)(r960twiddle + i*16*24 +16*j +10) = double2short( factor * cos(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 960 * i *j));

			*(WORD16 *)(r960twiddle + i*16*24 +16*j +12) = double2short( factor * cos(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 960 * i *j));

			*(WORD16 *)(r960twiddle + i*16*24 +16*j +14) = double2short( factor * cos(-2.0 * PI / 960 * i *j));
			*(WORD16 *)(r960twiddle + i*16*24 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 960 * i *j));

		}
}

void init_dft972_twiddle_factor(WORD16 *r972twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<36;i++)
		for (j=0;j<27;j++)
		{
			/* d +cj */
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +0) = double2short( factor * sin(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +1) = double2short( factor * cos(-2.0 * PI / 972 * i *j));

			*(WORD16 *)(r972twiddle + i*16*27 +16*j +2) = double2short( factor * sin(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +3) = double2short( factor * cos(-2.0 * PI / 972 * i *j));

			*(WORD16 *)(r972twiddle + i*16*27 +16*j +4) = double2short( factor * sin(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +5) = double2short( factor * cos(-2.0 * PI / 972 * i *j));

			*(WORD16 *)(r972twiddle + i*16*27 +16*j +6) = double2short( factor * sin(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +7) = double2short( factor * cos(-2.0 * PI / 972 * i *j));

			/* c -dj; */
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +8) = double2short( factor * cos(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 972 * i *j));

			*(WORD16 *)(r972twiddle + i*16*27 +16*j +10) = double2short( factor * cos(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 972 * i *j));

			*(WORD16 *)(r972twiddle + i*16*27 +16*j +12) = double2short( factor * cos(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 972 * i *j));

			*(WORD16 *)(r972twiddle + i*16*27 +16*j +14) = double2short( factor * cos(-2.0 * PI / 972 * i *j));
			*(WORD16 *)(r972twiddle + i*16*27 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 972 * i *j));

		}
}

void init_dft1080_twiddle_factor(WORD16 *r1080twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<36;i++)
		for (j=0;j<30;j++)
		{
			/* d +cj */
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +0) = double2short( factor * sin(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +1) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));

			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +2) = double2short( factor * sin(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +3) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));

			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +4) = double2short( factor * sin(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +5) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));

			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +6) = double2short( factor * sin(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +7) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));

			/* c -dj; */
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +8) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 1080 * i *j));

			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +10) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 1080 * i *j));

			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +12) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 1080 * i *j));

			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +14) = double2short( factor * cos(-2.0 * PI / 1080 * i *j));
			*(WORD16 *)(r1080twiddle + i*16*30 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 1080 * i *j));

		}
}

void init_dft1152_twiddle_factor(WORD16 *r1152twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<48;i++)
		for (j=0;j<24;j++)
		{
			/* d +cj */
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +0) = double2short( factor * sin(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +1) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));

			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +2) = double2short( factor * sin(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +3) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));

			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +4) = double2short( factor * sin(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +5) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));

			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +6) = double2short( factor * sin(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +7) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));

			/* c -dj; */
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +8) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 1152 * i *j));

			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +10) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 1152 * i *j));

			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +12) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 1152 * i *j));

			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +14) = double2short( factor * cos(-2.0 * PI / 1152 * i *j));
			*(WORD16 *)(r1152twiddle + i*16*24 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 1152 * i *j));

		}
}

void init_dft1200_twiddle_factor(WORD16 *r1200twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<48;i++)
		for (j=0;j<25;j++)
		{
			/* d +cj */
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +0) = double2short( factor * sin(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +1) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));

			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +2) = double2short( factor * sin(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +3) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));

			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +4) = double2short( factor * sin(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +5) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));

			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +6) = double2short( factor * sin(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +7) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));

			/* c -dj; */
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +8) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 1200 * i *j));

			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +10) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 1200 * i *j));

			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +12) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 1200 * i *j));

			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +14) = double2short( factor * cos(-2.0 * PI / 1200 * i *j));
			*(WORD16 *)(r1200twiddle + i*16*25 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 1200 * i *j));

		}
}

void init_dft1536_twiddle_factor(WORD16 *r1536twiddle)
{
    WORD32 i, j;
    DOUBLE64 factor = 32768;

	for (i=0;i<48;i++)
		for (j=0;j<32;j++)
		{
			/* d +cj */
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +0) = double2short( factor * sin(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +1) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));

			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +2) = double2short( factor * sin(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +3) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));

			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +4) = double2short( factor * sin(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +5) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));

			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +6) = double2short( factor * sin(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +7) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));

			/* c -dj; */
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +8) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +9) = double2short( -1*factor * sin(-2.0 * PI / 1536 * i *j));

			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +10) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +11) = double2short( -1*factor * sin(-2.0 * PI / 1536 * i *j));

			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +12) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +13) = double2short( -1*factor * sin(-2.0 * PI / 1536 * i *j));

			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +14) = double2short( factor * cos(-2.0 * PI / 1536 * i *j));
			*(WORD16 *)(r1536twiddle + i*16*32 +16*j +15) = double2short( -1*factor * sin(-2.0 * PI / 1536 * i *j));

		}
}

#define radix2_register(in0, in1, out0, out1)\
{\
	out0 = _mm_adds_epi16(in0, in1);\
	out1 = _mm_subs_epi16(in0, in1);\
}

#define radix3_register(in0, in1, in2, out0, out1, out2)\
{\
	m128_t0 = _mm_adds_epi16(in1, in2);\
	m128_t1 = _mm_subs_epi16(in1, in2);\
	\
	out0 = _mm_adds_epi16(in0, m128_t0);\
	m128_t0 = _mm_srai_epi16(m128_t0, 1); \
	m128_t0 = _mm_subs_epi16(in0, m128_t0 );\
	m128_t1 = _mm_mulhrs_epi16(m128_t1, DFT_ConstMinus08660); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch);\
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_R); \
	\
	out1 = _mm_adds_epi16(m128_t0, m128_t1);\
	out2 = _mm_subs_epi16(m128_t0, m128_t1);\
}

#define radix3_register_mul_zero_span(in0, in1, in2, out0, out1, out2, twiddle_addr, twiddle_span)\
{\
	m128_t0 = _mm_adds_epi16(in1, in2);\
	m128_t1 = _mm_subs_epi16(in1, in2);\
	\
	out0 = _mm_adds_epi16(in0, m128_t0);\
	out0 = _mm_srai_epi16(out0, 1); \
	 \
	m128_t0 = _mm_srai_epi16(m128_t0, 1); \
	m128_t0 = _mm_subs_epi16(in0, m128_t0 );\
	m128_t1 = _mm_mulhrs_epi16(m128_t1, DFT_ConstMinus08660); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch);\
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_R); \
	\
	out1 = _mm_adds_epi16(m128_t0, m128_t1);\
	out2 = _mm_subs_epi16(m128_t0, m128_t1);\
	 \
	twiddle_addr = twiddle_addr + 2*twiddle_span; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 1; \
	m128_t1 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(out1, m128_t0); \
	m128_t1 = _mm_madd_epi16(out1, m128_t1); \
	m128_t1 = _mm_srli_si128(m128_t1, 2); \
	out1 = _mm_blend_epi16(m128_t0, m128_t1, 0x55); \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 1; \
	m128_t1 = _mm_load_si128((__m128i *)twiddle_addr); \
	m128_t0 = _mm_madd_epi16(out2, m128_t0); \
	m128_t1 = _mm_madd_epi16(out2, m128_t1); \
	m128_t1 = _mm_srli_si128(m128_t1, 2); \
	out2 = _mm_blend_epi16(m128_t0, m128_t1, 0x55); \
}

#define radix3_register_mul_span(in0, in1, in2, out0, out1, out2, twiddle_addr, twiddle_span)\
{\
	m128_t0 = _mm_adds_epi16(in1, in2);\
	m128_t1 = _mm_subs_epi16(in1, in2);\
	\
	out0 = _mm_adds_epi16(in0, m128_t0);\
	m128_t0 = _mm_srai_epi16(m128_t0, 1); \
	m128_t0 = _mm_subs_epi16(in0, m128_t0 );\
	m128_t1 = _mm_mulhrs_epi16(m128_t1, DFT_ConstMinus08660); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch);\
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_R); \
	\
	out1 = _mm_adds_epi16(m128_t0, m128_t1);\
	out2 = _mm_subs_epi16(m128_t0, m128_t1);\
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 1; \
	m128_t1 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(out0, m128_t0); \
	m128_t1 = _mm_madd_epi16(out0, m128_t1); \
	m128_t1 = _mm_srli_si128(m128_t1, 2); \
	out0 = _mm_blend_epi16(m128_t0, m128_t1, 0x55); \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 1; \
	m128_t1 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(out1, m128_t0); \
	m128_t1 = _mm_madd_epi16(out1, m128_t1); \
	m128_t1 = _mm_srli_si128(m128_t1, 2); \
	out1 = _mm_blend_epi16(m128_t0, m128_t1, 0x55); \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr); twiddle_addr = twiddle_addr + 1; \
	m128_t1 = _mm_load_si128((__m128i *)twiddle_addr); \
	m128_t0 = _mm_madd_epi16(out2, m128_t0); \
	m128_t1 = _mm_madd_epi16(out2, m128_t1); \
	m128_t1 = _mm_srli_si128(m128_t1, 2); \
	out2 = _mm_blend_epi16(m128_t0, m128_t1, 0x55); \
}

#define radix3_0(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp); \
     \
    m128_t3 = _mm_adds_epi16(m128_t1, m128_t2); \
    m128_t4 = _mm_subs_epi16(m128_t1, m128_t2); \
     \
    m128_t6 = _mm_adds_epi16(m128_t0, m128_t3); \
    m128_t3 = _mm_srai_epi16(m128_t3, 1); \
    m128_t4 = _mm_mulhrs_epi16(m128_t4, DFT_ConstMinus08660); \
    m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
    m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
     \
    m128_t0 = _mm_subs_epi16(m128_t0, m128_t3); \
    m128_t7 = _mm_adds_epi16(m128_t0, m128_t4); \
    m128_t8 = _mm_subs_epi16(m128_t0, m128_t4); \
     \
    out_addr_temp = out_addr; \
    _mm_store_si128(out_addr_temp, m128_t6);    out_addr_temp = out_addr_temp + out_span; \
    _mm_store_si128(out_addr_temp, m128_t7);    out_addr_temp = out_addr_temp + out_span; \
    _mm_store_si128(out_addr_temp, m128_t8); \
}

#define radix3_0_zeromul(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp); \
     \
    m128_t3 = _mm_adds_epi16(m128_t1, m128_t2); \
    m128_t4 = _mm_subs_epi16(m128_t1, m128_t2); \
     \
    m128_t6 = _mm_adds_epi16(m128_t0, m128_t3); \
    m128_t3 = _mm_srai_epi16(m128_t3, 1); \
    m128_t4 = _mm_mulhrs_epi16(m128_t4, DFT_ConstMinus08660); \
    m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
    m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
     \
    m128_t0 = _mm_subs_epi16(m128_t0, m128_t3); \
    m128_t7 = _mm_adds_epi16(m128_t0, m128_t4); \
    m128_t8 = _mm_subs_epi16(m128_t0, m128_t4); \
     \
    out_addr_temp = out_addr; \
    _mm_store_si128(out_addr_temp, _mm_srai_epi16(m128_t6, 1));    out_addr_temp = out_addr_temp + out_span; \
    _mm_store_si128(out_addr_temp, _mm_srai_epi16(m128_t7, 1));    out_addr_temp = out_addr_temp + out_span; \
    _mm_store_si128(out_addr_temp, _mm_srai_epi16(m128_t8, 1)); \
}

#define radix3_0_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
    __m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp); \
     \
    m128_t3 = _mm_adds_epi16(m128_t1, m128_t2); \
    m128_t4 = _mm_subs_epi16(m128_t1, m128_t2); \
     \
    m128_t6 = _mm_adds_epi16(m128_t0, m128_t3); \
    m128_t3 = _mm_srai_epi16(m128_t3, 1); \
    m128_t4 = _mm_mulhrs_epi16(m128_t4, DFT_ConstMinus08660); \
    m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
    m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
     \
    m128_t0 = _mm_subs_epi16(m128_t0, m128_t3); \
    m128_t7 = _mm_adds_epi16(m128_t0, m128_t4); \
    m128_t8 = _mm_subs_epi16(m128_t0, m128_t4); \
     \
    out_addr_temp = out_addr; \
    _mm_store_si128(out_addr_temp, _mm_srai_epi16(m128_t6, 1));    out_addr_temp = out_addr_temp + out_span; \
     \
    twiddle_addr_temp = twiddle_addr + 2*twiddle_span; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t7, m128_t0); \
	m128_t3 = _mm_madd_epi16(m128_t7, m128_t3); \
	m128_t3 = _mm_srli_si128(m128_t3, 2); \
	m128_t0 = _mm_blend_epi16(m128_t0, m128_t3, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp,  m128_t0);    out_addr_temp = out_addr_temp + out_span; \
     \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t8, m128_t0); \
	m128_t3 = _mm_madd_epi16(m128_t8, m128_t3); \
	m128_t3 = _mm_srli_si128(m128_t3, 2); \
	m128_t0 = _mm_blend_epi16(m128_t0, m128_t3, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp,  m128_t0); \
}

#define radix3_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
    __m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp); \
     \
    m128_t3 = _mm_adds_epi16(m128_t1, m128_t2); \
    m128_t4 = _mm_subs_epi16(m128_t1, m128_t2); \
     \
    m128_t6 = _mm_adds_epi16(m128_t0, m128_t3); \
    m128_t3 = _mm_srai_epi16(m128_t3, 1); \
    m128_t4 = _mm_mulhrs_epi16(m128_t4, DFT_ConstMinus08660); \
    m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
    m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
     \
    m128_t0 = _mm_subs_epi16(m128_t0, m128_t3); \
    m128_t7 = _mm_adds_epi16(m128_t0, m128_t4); \
    m128_t8 = _mm_subs_epi16(m128_t0, m128_t4); \
     \
    out_addr_temp = out_addr; \
    twiddle_addr_temp = twiddle_addr; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t6, m128_t0); \
	m128_t3 = _mm_madd_epi16(m128_t6, m128_t3); \
	m128_t3 = _mm_srli_si128(m128_t3, 2); \
	m128_t0 = _mm_blend_epi16(m128_t0, m128_t3, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp,  m128_t0);    out_addr_temp = out_addr_temp + out_span; \
     \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t7, m128_t0); \
	m128_t3 = _mm_madd_epi16(m128_t7, m128_t3); \
	m128_t3 = _mm_srli_si128(m128_t3, 2); \
	m128_t0 = _mm_blend_epi16(m128_t0, m128_t3, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp,  m128_t0);    out_addr_temp = out_addr_temp + out_span; \
     \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t3 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t8, m128_t0); \
	m128_t3 = _mm_madd_epi16(m128_t8, m128_t3); \
	m128_t3 = _mm_srli_si128(m128_t3, 2); \
	m128_t0 = _mm_blend_epi16(m128_t0, m128_t3, 0x55); \
    _mm_store_si128((__m128i *)out_addr_temp,  m128_t0); \
}

#define radix4_register(in0,in1,in2,in3,out_addr,out_span) \
{\
	m128_t2 = _mm_adds_epi16(in0,in2); \
	m128_t6 = _mm_adds_epi16(in1,in3); \
	m128_t7=  _mm_adds_epi16(m128_t2,m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6); \
	m128_t6 =  m128_t7; \
	 \
	m128_t3 = _mm_subs_epi16(in0,in2); \
	m128_t12 = _mm_subs_epi16(in1,in3); \
	m128_t12 = _mm_shuffle_epi8(m128_t12, DFT_IQ_switch);\
	m128_t12 = _mm_sign_epi16(m128_t12, DFT_Neg_R); \
	 \
	m128_t7 = _mm_subs_epi16(m128_t3,m128_t12); \
	m128_t3 = _mm_adds_epi16(m128_t3,m128_t12); \
	 \
	_mm_store_si128((__m128i *)out_addr, m128_t6); out_addr = out_addr + out_span; \
	_mm_store_si128((__m128i *)out_addr, m128_t7); out_addr = out_addr + out_span; \
	_mm_store_si128((__m128i *)out_addr, m128_t2); out_addr = out_addr + out_span; \
	_mm_store_si128((__m128i *)out_addr, m128_t3); \
}

#define radix4_register_zeromul(in0,in1,in2,in3,out_addr,out_span) \
{\
	m128_t2 = _mm_adds_epi16(in0,in2); \
	m128_t6 = _mm_adds_epi16(in1,in3); \
	m128_t7=  _mm_adds_epi16(m128_t2,m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6); \
	m128_t6 =  m128_t7; \
	 \
	m128_t3 = _mm_subs_epi16(in0,in2); \
	m128_t12 = _mm_subs_epi16(in1,in3); \
	m128_t12 = _mm_shuffle_epi8(m128_t12, DFT_IQ_switch);\
	m128_t12 = _mm_sign_epi16(m128_t12, DFT_Neg_R); \
	 \
	m128_t7 = _mm_subs_epi16(m128_t3,m128_t12); \
	m128_t3 = _mm_adds_epi16(m128_t3,m128_t12); \
	 \
	_mm_store_si128((__m128i *)out_addr, _mm_srai_epi16(m128_t6, 1)); out_addr = out_addr + out_span; \
	_mm_store_si128((__m128i *)out_addr, _mm_srai_epi16(m128_t7, 1)); out_addr = out_addr + out_span;\
	_mm_store_si128((__m128i *)out_addr, _mm_srai_epi16(m128_t2, 1)); out_addr = out_addr + out_span;\
	_mm_store_si128((__m128i *)out_addr, _mm_srai_epi16(m128_t3, 1)); \
}

#define radix4_register_mul_zero(in0, in1, in2, in3, out_addr, out_span, twiddle_addr) \
{ \
	__m128i *twiddle_addr_temp; \
	m128_t2 = _mm_adds_epi16(in0,in2); \
	m128_t6 = _mm_adds_epi16(in1,in3); \
	m128_t7=  _mm_adds_epi16(m128_t2,m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6); \
	m128_t6 =  m128_t7; \
	 \
	m128_t3 = _mm_subs_epi16(in0,in2); \
	m128_t12 = _mm_subs_epi16(in1,in3); \
	m128_t12 = _mm_shuffle_epi8(m128_t12, DFT_IQ_switch); \
	m128_t12 = _mm_sign_epi16(m128_t12, DFT_Neg_R); \
	 \
	m128_t7 = _mm_subs_epi16(m128_t3,m128_t12); \
	m128_t3 = _mm_adds_epi16(m128_t3,m128_t12); \
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
	__m128i *twiddle_addr_temp; \
	m128_t2 = _mm_adds_epi16(in0,in2); \
	m128_t6 = _mm_adds_epi16(in1,in3); \
	m128_t7=  _mm_adds_epi16(m128_t2,m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6); \
	m128_t6 =  m128_t7; \
	 \
	m128_t3 = _mm_subs_epi16(in0,in2); \
	m128_t12 = _mm_subs_epi16(in1,in3); \
	m128_t12 = _mm_shuffle_epi8(m128_t12, DFT_IQ_switch); \
	m128_t12 = _mm_sign_epi16(m128_t12, DFT_Neg_R); \
	 \
	m128_t7 = _mm_subs_epi16(m128_t3,m128_t12); \
	m128_t3 = _mm_adds_epi16(m128_t3,m128_t12); \
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

#define radix4_0(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, DFT_Neg_R); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t9 = _mm_subs_epi16(m128_t6,m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t11 = _mm_adds_epi16(m128_t6,m128_t7); \
	 \
	out_addr_temp = out_addr; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t8);  out_addr_temp = out_addr_temp +out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t9);  out_addr_temp = out_addr_temp +out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t10); out_addr_temp = out_addr_temp +out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t11); \
}

#define radix4_0_zeromul(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	in_addr_temp = in_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, DFT_Neg_R); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t9 = _mm_subs_epi16(m128_t6,m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t11 = _mm_adds_epi16(m128_t6,m128_t7); \
	 \
	out_addr_temp = out_addr; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t8, 1));  out_addr_temp = out_addr_temp +out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t9, 1));  out_addr_temp = out_addr_temp +out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t10, 1)); out_addr_temp = out_addr_temp +out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t11, 1)); \
}

#define radix4_0_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	__m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, DFT_Neg_R); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t9 = _mm_subs_epi16(m128_t6,m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t11 = _mm_adds_epi16(m128_t6,m128_t7); \
	 \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t8, 1));  out_addr_temp = out_addr_temp +out_span; \
	 \
	twiddle_addr_temp = twiddle_addr + 2*twiddle_span; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t9, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t9, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t10, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t10, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t11, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t11, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); \
}

#define radix4_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	__m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); \
	m128_t4 = _mm_adds_epi16(m128_t0,m128_t2); \
	m128_t5 = _mm_adds_epi16(m128_t1,m128_t3); \
	m128_t6 = _mm_subs_epi16(m128_t0,m128_t2); \
	m128_t7 = _mm_subs_epi16(m128_t1,m128_t3); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t7 = _mm_sign_epi16(m128_t7, DFT_Neg_R); \
	 \
	m128_t8 = _mm_adds_epi16(m128_t4,m128_t5); \
	m128_t9 = _mm_subs_epi16(m128_t6,m128_t7); \
	m128_t10 = _mm_subs_epi16(m128_t4,m128_t5); \
	m128_t11 = _mm_adds_epi16(m128_t6,m128_t7); \
	 \
    	twiddle_addr_temp = twiddle_addr; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t8, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t8, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t9, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t9, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t10, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t10, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t4 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t11, m128_t0); \
	m128_t4 = _mm_madd_epi16(m128_t11, m128_t4); \
	m128_t4 = _mm_srli_si128(m128_t4,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t4, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); \
}

#define radix5_0(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp); \
	 \
	m128_t5 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t6 = _mm_adds_epi16(m128_t2, m128_t3); \
	 \
	m128_t7 = _mm_adds_epi16(m128_t5, m128_t6); \
	m128_t7 = _mm_adds_epi16(m128_t7, m128_t0); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t7);  out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t7 = _mm_subs_epi16(m128_t1, m128_t4); \
	m128_t8 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t9 = _mm_mulhrs_epi16(m128_t5, DFT_Const03090); \
	m128_t10 = _mm_mulhrs_epi16(m128_t5, DFT_Const08090); \
	m128_t11 = _mm_mulhrs_epi16(m128_t6, DFT_Const03090); \
	m128_t12 = _mm_mulhrs_epi16(m128_t6, DFT_Const08090); \
	 \
	m128_t1 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus09511); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_R); \
	m128_t2 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus05878); \
	m128_t2 = _mm_shuffle_epi8(m128_t2, DFT_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, DFT_Neg_R); \
	m128_t3 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus09511); \
	m128_t3 = _mm_shuffle_epi8(m128_t3, DFT_IQ_switch); \
	m128_t3 = _mm_sign_epi16(m128_t3, DFT_Neg_R); \
	m128_t4 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus05878); \
	m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
	m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
	 \
	m128_t9 = _mm_subs_epi16(m128_t9, m128_t12); \
	m128_t9 = _mm_adds_epi16(m128_t9, m128_t0); \
	m128_t10 = _mm_subs_epi16(m128_t10, m128_t11); \
	m128_t10 = _mm_subs_epi16(m128_t0, m128_t10); \
	m128_t11 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t12 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t1 = _mm_adds_epi16(m128_t9, m128_t11); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t1);  out_addr_temp = out_addr_temp + out_span; \
	m128_t1 = _mm_adds_epi16(m128_t10, m128_t12); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t1);  out_addr_temp = out_addr_temp + out_span; \
	m128_t1 = _mm_sub_epi16(m128_t10, m128_t12); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t1);  out_addr_temp = out_addr_temp + out_span; \
	m128_t1 = _mm_sub_epi16(m128_t9, m128_t11); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t1); \
}

#define radix5_0_zeromul(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp); \
	 \
	m128_t5 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t6 = _mm_adds_epi16(m128_t2, m128_t3); \
	 \
	m128_t7 = _mm_adds_epi16(m128_t5, m128_t6); \
	m128_t7 = _mm_adds_epi16(m128_t7, m128_t0); \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t7, 1));  out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t7 = _mm_subs_epi16(m128_t1, m128_t4); \
	m128_t8 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t9 = _mm_mulhrs_epi16(m128_t5, DFT_Const03090); \
	m128_t10 = _mm_mulhrs_epi16(m128_t5, DFT_Const08090); \
	m128_t11 = _mm_mulhrs_epi16(m128_t6, DFT_Const03090); \
	m128_t12 = _mm_mulhrs_epi16(m128_t6, DFT_Const08090); \
	 \
	m128_t1 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus09511); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_R); \
	m128_t2 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus05878); \
	m128_t2 = _mm_shuffle_epi8(m128_t2, DFT_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, DFT_Neg_R); \
	m128_t3 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus09511); \
	m128_t3 = _mm_shuffle_epi8(m128_t3, DFT_IQ_switch); \
	m128_t3 = _mm_sign_epi16(m128_t3, DFT_Neg_R); \
	m128_t4 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus05878); \
	m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
	m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
	 \
	m128_t9 = _mm_subs_epi16(m128_t9, m128_t12); \
	m128_t9 = _mm_adds_epi16(m128_t9, m128_t0); \
	m128_t10 = _mm_subs_epi16(m128_t10, m128_t11); \
	m128_t10 = _mm_subs_epi16(m128_t0, m128_t10); \
	m128_t11 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t12 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t1 = _mm_adds_epi16(m128_t9, m128_t11); \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t1, 1));  out_addr_temp = out_addr_temp + out_span; \
	m128_t1 = _mm_adds_epi16(m128_t10, m128_t12); \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t1, 1));  out_addr_temp = out_addr_temp + out_span; \
	m128_t1 = _mm_sub_epi16(m128_t10, m128_t12); \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t1, 1));  out_addr_temp = out_addr_temp + out_span; \
	m128_t1 = _mm_sub_epi16(m128_t9, m128_t11); \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t1, 1)); \
}

#define radix5_0_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	__m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp); \
	 \
	m128_t5 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t6 = _mm_adds_epi16(m128_t2, m128_t3); \
	 \
	m128_t7 = _mm_adds_epi16(m128_t5, m128_t6); \
	m128_t7 = _mm_adds_epi16(m128_t7, m128_t0); \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t7, 1));  out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t7 = _mm_subs_epi16(m128_t1, m128_t4); \
	m128_t8 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t9 = _mm_mulhrs_epi16(m128_t5, DFT_Const03090); \
	m128_t10 = _mm_mulhrs_epi16(m128_t5, DFT_Const08090); \
	m128_t11 = _mm_mulhrs_epi16(m128_t6, DFT_Const03090); \
	m128_t12 = _mm_mulhrs_epi16(m128_t6, DFT_Const08090); \
	 \
	m128_t1 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus09511); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_R); \
	m128_t2 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus05878); \
	m128_t2 = _mm_shuffle_epi8(m128_t2, DFT_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, DFT_Neg_R); \
	m128_t3 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus09511); \
	m128_t3 = _mm_shuffle_epi8(m128_t3, DFT_IQ_switch); \
	m128_t3 = _mm_sign_epi16(m128_t3, DFT_Neg_R); \
	m128_t4 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus05878); \
	m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
	m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
	 \
	m128_t9 = _mm_subs_epi16(m128_t9, m128_t12); \
	m128_t9 = _mm_adds_epi16(m128_t9, m128_t0); \
	m128_t10 = _mm_subs_epi16(m128_t10, m128_t11); \
	m128_t10 = _mm_subs_epi16(m128_t0, m128_t10); \
	m128_t11 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t12 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t1 = _mm_adds_epi16(m128_t9, m128_t11); \
	m128_t2 = _mm_adds_epi16(m128_t10, m128_t12); \
	m128_t3 = _mm_sub_epi16(m128_t10, m128_t12); \
	m128_t4 = _mm_sub_epi16(m128_t9, m128_t11); \
	 \
	twiddle_addr_temp = twiddle_addr + 2*twiddle_span; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t1, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t1, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t2, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t2, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t3, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t3, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t4, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t4, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); \
}

#define radix5_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	__m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp); \
	 \
	m128_t5 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t6 = _mm_adds_epi16(m128_t2, m128_t3); \
	 \
	m128_t7 = _mm_adds_epi16(m128_t5, m128_t6); \
	m128_t7 = _mm_adds_epi16(m128_t7, m128_t0); \
     \
	twiddle_addr_temp = twiddle_addr; \
	m128_t8 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t9 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t8 = _mm_madd_epi16(m128_t7, m128_t8); \
	m128_t9 = _mm_madd_epi16(m128_t7, m128_t9); \
	m128_t9 = _mm_srli_si128(m128_t9,2); \
	m128_t8 = _mm_blend_epi16(m128_t8,m128_t9, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t8); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t7 = _mm_subs_epi16(m128_t1, m128_t4); \
	m128_t8 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t9 = _mm_mulhrs_epi16(m128_t5, DFT_Const03090); \
	m128_t10 = _mm_mulhrs_epi16(m128_t5, DFT_Const08090); \
	m128_t11 = _mm_mulhrs_epi16(m128_t6, DFT_Const03090); \
	m128_t12 = _mm_mulhrs_epi16(m128_t6, DFT_Const08090); \
	 \
	m128_t1 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus09511); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_R); \
	m128_t2 = _mm_mulhrs_epi16(m128_t7, DFT_ConstMinus05878); \
	m128_t2 = _mm_shuffle_epi8(m128_t2, DFT_IQ_switch); \
	m128_t2 = _mm_sign_epi16(m128_t2, DFT_Neg_R); \
	m128_t3 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus09511); \
	m128_t3 = _mm_shuffle_epi8(m128_t3, DFT_IQ_switch); \
	m128_t3 = _mm_sign_epi16(m128_t3, DFT_Neg_R); \
	m128_t4 = _mm_mulhrs_epi16(m128_t8, DFT_ConstMinus05878); \
	m128_t4 = _mm_shuffle_epi8(m128_t4, DFT_IQ_switch); \
	m128_t4 = _mm_sign_epi16(m128_t4, DFT_Neg_R); \
	 \
	m128_t9 = _mm_subs_epi16(m128_t9, m128_t12); \
	m128_t9 = _mm_adds_epi16(m128_t9, m128_t0); \
	m128_t10 = _mm_subs_epi16(m128_t10, m128_t11); \
	m128_t10 = _mm_subs_epi16(m128_t0, m128_t10); \
	m128_t11 = _mm_adds_epi16(m128_t1, m128_t4); \
	m128_t12 = _mm_subs_epi16(m128_t2, m128_t3); \
	 \
	m128_t1 = _mm_adds_epi16(m128_t9, m128_t11); \
	m128_t2 = _mm_adds_epi16(m128_t10, m128_t12); \
	m128_t3 = _mm_sub_epi16(m128_t10, m128_t12); \
	m128_t4 = _mm_sub_epi16(m128_t9, m128_t11); \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t1, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t1, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t2, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t2, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t3, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t3, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t5 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t4, m128_t0); \
	m128_t5 = _mm_madd_epi16(m128_t4, m128_t5); \
	m128_t5 = _mm_srli_si128(m128_t5,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t5, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); \
}

#define radix6_0(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128(in_addr_temp);  \
     \
	radix2_register(m128_t0,m128_t3,m128_t6,m128_t7); \
	radix2_register(m128_t1,m128_t4,m128_t8,m128_t9); \
	radix2_register(m128_t2,m128_t5,m128_t10,m128_t11); \
	m128_t2 = _mm_shuffle_epi8(m128_t9, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t9, DFT_Const0500_0866_0); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t9 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t11, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t11, DFT_Const0500_0866_2); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t11 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	radix3_register(m128_t6,m128_t8,m128_t10,m128_t2,m128_t3,m128_t4); \
	radix3_register(m128_t7,m128_t9,m128_t11,m128_t5,m128_t12,m128_t13); \
	 \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t2);  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t5);  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t3);  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12);  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t4);  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t13); \
}

#define radix6_0_zeromul(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128(in_addr_temp);  \
    	 \
	radix2_register(m128_t0,m128_t3,m128_t6,m128_t7); \
	radix2_register(m128_t1,m128_t4,m128_t8,m128_t9); \
	radix2_register(m128_t2,m128_t5,m128_t10,m128_t11); \
	m128_t2 = _mm_shuffle_epi8(m128_t9, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t9, DFT_Const0500_0866_0); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t9 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t11, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t11, DFT_Const0500_0866_2); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t11 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	radix3_register(m128_t6,m128_t8,m128_t10,m128_t2,m128_t3,m128_t4); \
	radix3_register(m128_t7,m128_t9,m128_t11,m128_t5,m128_t12,m128_t13); \
	 \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t2, 1));  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t5, 1));  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t3, 1));  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t12, 1));  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t4, 1));  out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t13, 1)); \
}

#define radix6_0_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	__m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128(in_addr_temp); \
    	 \
	radix2_register(m128_t0,m128_t3,m128_t6,m128_t7); \
	radix2_register(m128_t1,m128_t4,m128_t8,m128_t9); \
	radix2_register(m128_t2,m128_t5,m128_t10,m128_t11); \
	m128_t2 = _mm_shuffle_epi8(m128_t9, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t9, DFT_Const0500_0866_0); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t9 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t11, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t11, DFT_Const0500_0866_2); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t11 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	radix3_register(m128_t6,m128_t8,m128_t10,m128_t2,m128_t3,m128_t4); \
	radix3_register(m128_t7,m128_t9,m128_t11,m128_t5,m128_t12,m128_t13); \
	 \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t2, 1)); out_addr_temp = out_addr_temp + out_span; \
	 \
	twiddle_addr_temp = twiddle_addr + 2*twiddle_span; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t5, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t5, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t3, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t3, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t12, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t12, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t4, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t4, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t13, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t13, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); \
}

#define radix6_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_span) \
{ \
	__m128i *out_addr_temp; \
	__m128i *in_addr_temp; \
	__m128i *twiddle_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr; \
	m128_t0 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t1 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t2 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp);  in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128(in_addr_temp); \
    	 \
	radix2_register(m128_t0,m128_t3,m128_t6,m128_t7); \
	radix2_register(m128_t1,m128_t4,m128_t8,m128_t9); \
	radix2_register(m128_t2,m128_t5,m128_t10,m128_t11); \
	m128_t2 = _mm_shuffle_epi8(m128_t9, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t9, DFT_Const0500_0866_0); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t9 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	m128_t2 = _mm_shuffle_epi8(m128_t11, DFT_IQ_switch); \
	m128_t0 = _mm_mulhrs_epi16(m128_t11, DFT_Const0500_0866_2); \
	m128_t1 = _mm_mulhrs_epi16(m128_t2, DFT_Const0500_0866_1); \
	m128_t11 = _mm_adds_epi16(m128_t0, m128_t1); \
	 \
	radix3_register(m128_t6,m128_t8,m128_t10,m128_t2,m128_t3,m128_t4); \
	radix3_register(m128_t7,m128_t9,m128_t11,m128_t5,m128_t12,m128_t13); \
	 \
    twiddle_addr_temp = twiddle_addr; \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t2, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t2, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t5, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t5, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t3, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t3, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t12, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t12, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 2*twiddle_span - 1; \
	m128_t0 = _mm_madd_epi16(m128_t4, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t4, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); out_addr_temp = out_addr_temp + out_span; \
	 \
	m128_t0 = _mm_load_si128((__m128i *)twiddle_addr_temp); twiddle_addr_temp = twiddle_addr_temp + 1; \
	m128_t6 = _mm_load_si128((__m128i *)twiddle_addr_temp); \
	m128_t0 = _mm_madd_epi16(m128_t13, m128_t0); \
	m128_t6 = _mm_madd_epi16(m128_t13, m128_t6); \
	m128_t6 = _mm_srli_si128(m128_t6,2); \
	m128_t0 = _mm_blend_epi16(m128_t0,m128_t6, 0x55); \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t0); \
}

#define radix8_0(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp1, *out_addr_temp2; \
	__m128i *in_addr_temp; \
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
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_I); \
	 \
	m128_t2 = _mm_mulhrs_epi16(m128_t11, DFT_Const_0707); \
	m128_t3 = _mm_mulhrs_epi16(m128_t5, DFT_Const_0707_Minus); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t2, DFT_IQ_switch); \
	m128_t7 = _mm_adds_epi16(m128_t2, m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2, m128_t6); \
	m128_t11 = _mm_blend_epi16(m128_t2, m128_t7, 0x55); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t3, DFT_IQ_switch); 	m128_t7 = _mm_adds_epi16(m128_t3, m128_t6); \
	m128_t3 = _mm_subs_epi16(m128_t3, m128_t6); \
	m128_t5 = _mm_blend_epi16(m128_t3, m128_t7, 0xAA); \
	 \
	out_addr_temp1 = out_addr; \
	out_addr_temp2 = out_addr + out_span; \
	radix4_register(m128_t8, m128_t10, m128_t0, m128_t4, out_addr_temp1, 2*out_span); \
	radix4_register(m128_t9, m128_t11, m128_t1, m128_t5, out_addr_temp2, 2*out_span); \
}

#define radix8_0_zeromul(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *out_addr_temp1, *out_addr_temp2; \
	__m128i *in_addr_temp; \
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
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_I); \
	 \
	m128_t2 = _mm_mulhrs_epi16(m128_t11, DFT_Const_0707); \
	m128_t3 = _mm_mulhrs_epi16(m128_t5, DFT_Const_0707_Minus); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t2, DFT_IQ_switch); \
	m128_t7 = _mm_adds_epi16(m128_t2, m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2, m128_t6); \
	m128_t11 = _mm_blend_epi16(m128_t2, m128_t7, 0x55); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t3, DFT_IQ_switch); 	m128_t7 = _mm_adds_epi16(m128_t3, m128_t6); \
	m128_t3 = _mm_subs_epi16(m128_t3, m128_t6); \
	m128_t5 = _mm_blend_epi16(m128_t3, m128_t7, 0xAA); \
	 \
	out_addr_temp1 = out_addr; \
	out_addr_temp2 = out_addr + out_span; \
	radix4_register_zeromul(m128_t8, m128_t10, m128_t0, m128_t4, out_addr_temp1, 2*out_span); \
	radix4_register_zeromul(m128_t9, m128_t11, m128_t1, m128_t5, out_addr_temp2, 2*out_span); \
}

#define radix8_0_mul(in_addr, in_span, out_addr, out_span, twiddle_addr) \
{ \
	__m128i * out_addr_temp1, *out_addr_temp2; \
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
	m128_t0 = _mm_adds_epi16(m128_t2, m128_t6);  \
	m128_t1 = _mm_subs_epi16(m128_t2, m128_t6); \
	m128_t4 = _mm_adds_epi16(m128_t3, m128_t7); \
	m128_t5 = _mm_subs_epi16(m128_t3, m128_t7); \
	m128_t1 = _mm_shuffle_epi8(m128_t1, DFT_IQ_switch); \
	m128_t1 = _mm_sign_epi16(m128_t1, DFT_Neg_I); \
	 \
	m128_t2 = _mm_mulhrs_epi16(m128_t11, DFT_Const_0707); \
	m128_t3 = _mm_mulhrs_epi16(m128_t5, DFT_Const_0707_Minus); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t2, DFT_IQ_switch); \
	m128_t7 = _mm_adds_epi16(m128_t2,m128_t6); \
	m128_t2 = _mm_subs_epi16(m128_t2,m128_t6); \
	m128_t11 = _mm_blend_epi16(m128_t2,m128_t7, 0x55); \
	 \
	m128_t6 = _mm_shuffle_epi8(m128_t3, DFT_IQ_switch); \
	m128_t7 = _mm_adds_epi16(m128_t3,m128_t6); \
	m128_t3 = _mm_subs_epi16(m128_t3,m128_t6); \
	m128_t5 = _mm_blend_epi16(m128_t3,m128_t7, 0xAA); \
	 \
	out_addr_temp1 = out_addr; \
	out_addr_temp2 = out_addr + out_span; \
	radix4_register_mul_zero(m128_t8, m128_t10, m128_t0, m128_t4, out_addr_temp1, 2*out_span, twiddle_addr); \
	radix4_register_mul(m128_t9, m128_t11, m128_t1, m128_t5, out_addr_temp2, 2*out_span, twiddle_addr + 2); \
}

#define radix9_0(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr;\
	m128_t2 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t6 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t7 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t8 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t9 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t10 = _mm_load_si128(in_addr_temp); \
	 \
	radix3_register(m128_t2, m128_t5, m128_t8, m128_t11, m128_t12, m128_t13); \
	radix3_register(m128_t3, m128_t6, m128_t9, m128_t2, m128_t5, m128_t8); \
	radix3_register(m128_t4, m128_t7, m128_t10, m128_t3, m128_t6, m128_t9); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t5, DFT_Const_07660); \
	m128_t7 = _mm_mulhrs_epi16(m128_t5, DFT_Const_Minus06428); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t5 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t8, DFT_Const_01736); \
	m128_t7 = _mm_mulhrs_epi16(m128_t8, DFT_Const_Minus09848); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t8 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t6, DFT_Const_01736); \
	m128_t7 = _mm_mulhrs_epi16(m128_t6, DFT_Const_Minus09848); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t6 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t9, DFT_Const_Minus09397); \
	m128_t7 = _mm_mulhrs_epi16(m128_t9, DFT_Const_Minus03420); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t9 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	radix3_register(m128_t11, m128_t2, m128_t3, m128_t4, m128_t7, m128_t10); \
	radix3_register(m128_t12, m128_t5, m128_t6, m128_t11, m128_t2, m128_t3); \
	radix3_register(m128_t13, m128_t8, m128_t9, m128_t12, m128_t5, m128_t6); \
	 \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t4); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t11); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t7); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t2); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t5); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t10); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t3); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t6); \
}

#define radix9_0_zeromul(in_addr, in_span, out_addr, out_span) \
{ \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr;\
	m128_t2 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t6 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t7 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t8 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t9 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t10 = _mm_load_si128(in_addr_temp); \
	 \
	radix3_register(m128_t2, m128_t5, m128_t8, m128_t11, m128_t12, m128_t13); \
	radix3_register(m128_t3, m128_t6, m128_t9, m128_t2, m128_t5, m128_t8); \
	radix3_register(m128_t4, m128_t7, m128_t10, m128_t3, m128_t6, m128_t9); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t5, DFT_Const_07660); \
	m128_t7 = _mm_mulhrs_epi16(m128_t5, DFT_Const_Minus06428); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t5 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t8, DFT_Const_01736); \
	m128_t7 = _mm_mulhrs_epi16(m128_t8, DFT_Const_Minus09848); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t8 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t6, DFT_Const_01736); \
	m128_t7 = _mm_mulhrs_epi16(m128_t6, DFT_Const_Minus09848); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t6 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t9, DFT_Const_Minus09397); \
	m128_t7 = _mm_mulhrs_epi16(m128_t9, DFT_Const_Minus03420); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t9 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	radix3_register(m128_t11, m128_t2, m128_t3, m128_t4, m128_t7, m128_t10); \
	radix3_register(m128_t12, m128_t5, m128_t6, m128_t11, m128_t2, m128_t3); \
	radix3_register(m128_t13, m128_t8, m128_t9, m128_t12, m128_t5, m128_t6); \
	 \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t4, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t11, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t12, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t7, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t2, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t5, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t10, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t3, 1)); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, _mm_srai_epi16(m128_t6, 1)); \
}

#define radix9_0_mul_span(in_addr, in_span, out_addr, out_span, twiddle_addr, twiddle_scale) \
{ \
	__m128i *in_addr_temp; \
	__m128i *out_addr_temp; \
	__m128i *twiddle_addr_temp1, *twiddle_addr_temp2, *twiddle_addr_temp3; \
	in_addr_temp = in_addr; \
	out_addr_temp = out_addr;\
	m128_t2 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t3 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t4 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t5 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t6 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t7 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t8 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t9 = _mm_load_si128(in_addr_temp); in_addr_temp = in_addr_temp + in_span; \
	m128_t10 = _mm_load_si128(in_addr_temp); \
	 \
	radix3_register(m128_t2, m128_t5, m128_t8, m128_t11, m128_t12, m128_t13); \
	radix3_register(m128_t3, m128_t6, m128_t9, m128_t2, m128_t5, m128_t8); \
	radix3_register(m128_t4, m128_t7, m128_t10, m128_t3, m128_t6, m128_t9); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t5, DFT_Const_07660); \
	m128_t7 = _mm_mulhrs_epi16(m128_t5, DFT_Const_Minus06428); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t5 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t8, DFT_Const_01736); \
	m128_t7 = _mm_mulhrs_epi16(m128_t8, DFT_Const_Minus09848); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t8 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t6, DFT_Const_01736); \
	m128_t7 = _mm_mulhrs_epi16(m128_t6, DFT_Const_Minus09848); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t6 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	m128_t4 = _mm_mulhrs_epi16(m128_t9, DFT_Const_Minus09397); \
	m128_t7 = _mm_mulhrs_epi16(m128_t9, DFT_Const_Minus03420); \
	m128_t7 = _mm_shuffle_epi8(m128_t7, DFT_IQ_switch); \
	m128_t9 = _mm_adds_epi16(m128_t4, m128_t7); \
	 \
	twiddle_addr_temp1 = twiddle_addr + 2*out_span*twiddle_scale*0; \
    twiddle_addr_temp2 = twiddle_addr + 2*out_span*twiddle_scale*1; \
    twiddle_addr_temp3 = twiddle_addr + 2*out_span*twiddle_scale*2; \
	radix3_register_mul_zero_span(m128_t11, m128_t2, m128_t3, m128_t4, m128_t7, m128_t10, twiddle_addr_temp1, 3*out_span*twiddle_scale); \
	radix3_register_mul_span(m128_t12, m128_t5, m128_t6, m128_t11, m128_t2, m128_t3, twiddle_addr_temp2, 3*out_span*twiddle_scale);  \
	radix3_register_mul_span(m128_t13, m128_t8, m128_t9, m128_t12, m128_t5, m128_t6, twiddle_addr_temp3, 3*out_span*twiddle_scale); \
	 \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t4); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t11); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t12); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t7); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t2); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t5); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t10); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t3); out_addr_temp = out_addr_temp + out_span; \
	_mm_store_si128((__m128i *)out_addr_temp, m128_t6); \
}

#define radix12(InBuf, in_span, OutBuf, out_span, r12twiddle) \
{ \
	radix4_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*4, 1); \
	radix4_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*4, 1, r12twiddle + 2*4*1, 1); \
	radix4_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*4, 1, r12twiddle + 2*4*2, 1); \
     \
	radix3_0(dft_Temp_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span); \
	radix3_0(dft_Temp_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span); \
	radix3_0(dft_Temp_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span); \
	radix3_0(dft_Temp_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span); \
}

#define radix12_0_zeromul(InBuf, in_span, OutBuf, out_span, r12twiddle) \
{ \
	radix4_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*4, 1); \
	radix4_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*4, 1, r12twiddle + 2*4*1, 1); \
	radix4_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*4, 1, r12twiddle + 2*4*2, 1); \
     \
	radix3_0_zeromul(dft_Temp_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span); \
}

#define radix12_0_mul(InBuf, in_span, OutBuf, out_span, r12twiddle, twiddle, twiddle_scale) \
{ \
	radix4_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*4, 1); \
	radix4_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*4, 1, r12twiddle + 2*4*1, 1); \
	radix4_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*4, 1, r12twiddle + 2*4*2, 1); \
     \
	radix3_0_mul_span(dft_Temp_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*0, 4*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*1, 4*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*2, 4*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*3, 4*out_span*twiddle_scale); \
}

#define radix15(InBuf, in_span, OutBuf, out_span, r15twiddle) \
{ \
	radix5_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*5, 1); \
	radix5_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*5, 1, r15twiddle + 2*5*1, 1); \
	radix5_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*5, 1, r15twiddle + 2*5*2, 1); \
	 \
	radix3_0(dft_Temp_Buf + 0, 5, OutBuf + 0*out_span, 5*out_span); \
	radix3_0(dft_Temp_Buf + 1, 5, OutBuf + 1*out_span, 5*out_span); \
	radix3_0(dft_Temp_Buf + 2, 5, OutBuf + 2*out_span, 5*out_span); \
	radix3_0(dft_Temp_Buf + 3, 5, OutBuf + 3*out_span, 5*out_span); \
	radix3_0(dft_Temp_Buf + 4, 5, OutBuf + 4*out_span, 5*out_span); \
}

#define radix16(InBuf, in_span, OutBuf, out_span, r16twiddle) \
{ \
	radix4_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*4, 1); \
	radix4_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*4, 1, r16twiddle + 2*4*1, 1); \
	radix4_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*4, 1, r16twiddle + 2*4*2, 1); \
	radix4_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*4, 1, r16twiddle + 2*4*3, 1); \
	 \
	radix4_0(dft_Temp_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span); \
	radix4_0(dft_Temp_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span); \
	radix4_0(dft_Temp_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span); \
	radix4_0(dft_Temp_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span); \
}

#define radix16_0_zeromul(InBuf, in_span, OutBuf, out_span, r16twiddle) \
{ \
	radix4_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*4, 1); \
	radix4_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*4, 1, r16twiddle + 2*4*1, 1); \
	radix4_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*4, 1, r16twiddle + 2*4*2, 1); \
	radix4_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*4, 1, r16twiddle + 2*4*3, 1); \
	 \
	radix4_0_zeromul(dft_Temp_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span); \
}

#define radix16_0_mul(InBuf, in_span, OutBuf, out_span, r16twiddle, twiddle, twiddle_scale) \
{ \
	radix4_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*4, 1); \
	radix4_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*4, 1, r16twiddle + 2*4*1, 1); \
	radix4_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*4, 1, r16twiddle + 2*4*2, 1); \
	radix4_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*4, 1, r16twiddle + 2*4*3, 1); \
	 \
	radix4_0_mul_span(dft_Temp_Buf + 0, 4, OutBuf + 0*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*0, 4*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 1, 4, OutBuf + 1*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*1, 4*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 2, 4, OutBuf + 2*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*2, 4*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 3, 4, OutBuf + 3*out_span, 4*out_span, twiddle + 2*out_span*twiddle_scale*3, 4*out_span*twiddle_scale); \
}

#define radix18_0_zeromul(InBuf, in_span, OutBuf, out_span, r18twiddle) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*6, 1, r18twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*6, 1, r18twiddle + 2*6*2, 1); \
	 \
	radix3_0_zeromul(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span); \
}

#define radix18_0_mul(InBuf, in_span, OutBuf, out_span, r18twiddle, twiddle, twiddle_scale) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*6, 1, r18twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*6, 1, r18twiddle + 2*6*2, 1); \
	 \
	radix3_0_mul_span(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*0, 6*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*1, 6*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*2, 6*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*3, 6*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*4, 6*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*5, 6*out_span*twiddle_scale); \
}

#define radix20(InBuf, in_span, OutBuf, out_span, r20twiddle) \
{ \
	radix5_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*5, 1); \
	radix5_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*5, 1, r20twiddle + 2*5*1, 1); \
	radix5_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*5, 1, r20twiddle + 2*5*2, 1); \
	radix5_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*5, 1, r20twiddle + 2*5*3, 1); \
	 \
	radix4_0(dft_Temp_Buf + 0, 5, OutBuf + 0*out_span, 5*out_span); \
	radix4_0(dft_Temp_Buf + 1, 5, OutBuf + 1*out_span, 5*out_span); \
	radix4_0(dft_Temp_Buf + 2, 5, OutBuf + 2*out_span, 5*out_span); \
	radix4_0(dft_Temp_Buf + 3, 5, OutBuf + 3*out_span, 5*out_span); \
	radix4_0(dft_Temp_Buf + 4, 5, OutBuf + 4*out_span, 5*out_span); \
}

#define radix20_0_zeromul(InBuf, in_span, OutBuf, out_span, r20twiddle) \
{ \
	radix5_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*5, 1); \
	radix5_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*5, 1, r20twiddle + 2*5*1, 1); \
	radix5_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*5, 1, r20twiddle + 2*5*2, 1); \
	radix5_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*5, 1, r20twiddle + 2*5*3, 1); \
	 \
	radix4_0_zeromul(dft_Temp_Buf + 0, 5, OutBuf + 0*out_span, 5*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 1, 5, OutBuf + 1*out_span, 5*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 2, 5, OutBuf + 2*out_span, 5*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 3, 5, OutBuf + 3*out_span, 5*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 4, 5, OutBuf + 4*out_span, 5*out_span); \
}

#define radix20_0_mul(InBuf, in_span, OutBuf, out_span, r20twiddle, twiddle, twiddle_scale) \
{ \
	radix5_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*5, 1); \
	radix5_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*5, 1, r20twiddle + 2*5*1, 1); \
	radix5_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*5, 1, r20twiddle + 2*5*2, 1); \
	radix5_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*5, 1, r20twiddle + 2*5*3, 1); \
	 \
	radix4_0_mul_span(dft_Temp_Buf + 0, 5, OutBuf + 0*out_span, 5*out_span, twiddle + 2*out_span*twiddle_scale*0, 5*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 1, 5, OutBuf + 1*out_span, 5*out_span, twiddle + 2*out_span*twiddle_scale*1, 5*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 2, 5, OutBuf + 2*out_span, 5*out_span, twiddle + 2*out_span*twiddle_scale*2, 5*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 3, 5, OutBuf + 3*out_span, 5*out_span, twiddle + 2*out_span*twiddle_scale*3, 5*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 4, 5, OutBuf + 4*out_span, 5*out_span, twiddle + 2*out_span*twiddle_scale*4, 5*out_span*twiddle_scale); \
}

#define radix24(InBuf, in_span, OutBuf, out_span, r24twiddle) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*6, 1, r24twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*6, 1, r24twiddle + 2*6*2, 1); \
	radix6_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*6, 1, r24twiddle + 2*6*3, 1); \
	 \
	radix4_0(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span); \
	radix4_0(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span); \
	radix4_0(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span); \
	radix4_0(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span); \
	radix4_0(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span); \
	radix4_0(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span); \
}

#define radix24_0_zeromul(InBuf, in_span, OutBuf, out_span, r24twiddle) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*6, 1, r24twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*6, 1, r24twiddle + 2*6*2, 1); \
	radix6_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*6, 1, r24twiddle + 2*6*3, 1); \
	 \
	radix4_0_zeromul(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span); \
	radix4_0_zeromul(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span); \
}

#define radix24_0_mul(InBuf, in_span, OutBuf, out_span, r24twiddle, twiddle, twiddle_scale) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*6, 1, r24twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*6, 1, r24twiddle + 2*6*2, 1); \
	radix6_0_mul_span(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*6, 1, r24twiddle + 2*6*3, 1); \
	 \
	radix4_0_mul_span(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*0, 6*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*1, 6*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*2, 6*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*3, 6*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*4, 6*out_span*twiddle_scale); \
	radix4_mul_span(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*5, 6*out_span*twiddle_scale); \
}

#define radix25(InBuf, in_span, OutBuf, out_span, r25twiddle) \
{ \
	radix5_0_zeromul(InBuf + 0*in_span, 5*in_span, dft_Temp_Buf + 0*5, 1); \
	radix5_0_mul_span(InBuf + 1*in_span, 5*in_span, dft_Temp_Buf + 1*5, 1, r25twiddle + 2*5*1, 1); \
	radix5_0_mul_span(InBuf + 2*in_span, 5*in_span, dft_Temp_Buf + 2*5, 1, r25twiddle + 2*5*2, 1); \
	radix5_0_mul_span(InBuf + 3*in_span, 5*in_span, dft_Temp_Buf + 3*5, 1, r25twiddle + 2*5*3, 1); \
	radix5_0_mul_span(InBuf + 4*in_span, 5*in_span, dft_Temp_Buf + 4*5, 1, r25twiddle + 2*5*4, 1);  \
    	 \
	radix5_0(dft_Temp_Buf + 0, 5, OutBuf + 0*out_span, 5*out_span); \
	radix5_0(dft_Temp_Buf + 1, 5, OutBuf + 1*out_span, 5*out_span); \
	radix5_0(dft_Temp_Buf + 2, 5, OutBuf + 2*out_span, 5*out_span); \
	radix5_0(dft_Temp_Buf + 3, 5, OutBuf + 3*out_span, 5*out_span); \
	radix5_0(dft_Temp_Buf + 4, 5, OutBuf + 4*out_span, 5*out_span); \
}

#define radix27(InBuf, in_span, OutBuf, out_span, r27twiddle) \
{ \
	radix9_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*9, 1); \
	radix9_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*9, 1, r27twiddle + 2*9*1, 1); \
	radix9_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*9, 1, r27twiddle + 2*9*2, 1); \
	 \
	radix3_0(dft_Temp_Buf + 0, 9, OutBuf + 0*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 1, 9, OutBuf + 1*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 2, 9, OutBuf + 2*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 3, 9, OutBuf + 3*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 4, 9, OutBuf + 4*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 5, 9, OutBuf + 5*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 6, 9, OutBuf + 6*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 7, 9, OutBuf + 7*out_span, 9*out_span); \
	radix3_0(dft_Temp_Buf + 8, 9, OutBuf + 8*out_span, 9*out_span); \
}

#define radix27_0_zeromul(InBuf, in_span, OutBuf, out_span, r27twiddle) \
{ \
	radix9_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*9, 1); \
	radix9_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*9, 1, r27twiddle + 2*9*1, 1); \
	radix9_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*9, 1, r27twiddle + 2*9*2, 1); \
	 \
	radix3_0_zeromul(dft_Temp_Buf + 0, 9, OutBuf + 0*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 1, 9, OutBuf + 1*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 2, 9, OutBuf + 2*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 3, 9, OutBuf + 3*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 4, 9, OutBuf + 4*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 5, 9, OutBuf + 5*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 6, 9, OutBuf + 6*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 7, 9, OutBuf + 7*out_span, 9*out_span); \
	radix3_0_zeromul(dft_Temp_Buf + 8, 9, OutBuf + 8*out_span, 9*out_span); \
}

#define radix27_0_mul(InBuf, in_span, OutBuf, out_span, r27twiddle, twiddle, twiddle_scale) \
{ \
	radix9_0_zeromul(InBuf + 0*in_span, 3*in_span, dft_Temp_Buf + 0*9, 1); \
	radix9_0_mul_span(InBuf + 1*in_span, 3*in_span, dft_Temp_Buf + 1*9, 1, r27twiddle + 2*9*1, 1); \
	radix9_0_mul_span(InBuf + 2*in_span, 3*in_span, dft_Temp_Buf + 2*9, 1, r27twiddle + 2*9*2, 1); \
	 \
	radix3_0_mul_span(dft_Temp_Buf + 0, 9, OutBuf + 0*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*0, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 1, 9, OutBuf + 1*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*1, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 2, 9, OutBuf + 2*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*2, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 3, 9, OutBuf + 3*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*3, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 4, 9, OutBuf + 4*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*4, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 5, 9, OutBuf + 5*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*5, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 6, 9, OutBuf + 6*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*6, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 7, 9, OutBuf + 7*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*7, 9*out_span*twiddle_scale); \
	radix3_mul_span(dft_Temp_Buf + 8, 9, OutBuf + 8*out_span, 9*out_span, twiddle + 2*out_span*twiddle_scale*8, 9*out_span*twiddle_scale); \
}

#define radix30(InBuf, in_span, OutBuf, out_span, r30twiddle) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 5*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 5*in_span, dft_Temp_Buf + 1*6, 1, r30twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 5*in_span, dft_Temp_Buf + 2*6, 1, r30twiddle + 2*6*2, 1); \
	radix6_0_mul_span(InBuf + 3*in_span, 5*in_span, dft_Temp_Buf + 3*6, 1, r30twiddle + 2*6*3, 1); \
	radix6_0_mul_span(InBuf + 4*in_span, 5*in_span, dft_Temp_Buf + 4*6, 1, r30twiddle + 2*6*4, 1); \
	 \
	radix5_0(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span); \
	radix5_0(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span); \
	radix5_0(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span); \
	radix5_0(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span); \
	radix5_0(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span); \
	radix5_0(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span); \
}

#define radix32(InBuf, in_span, OutBuf, out_span, r32twiddle) \
{ \
	radix8_0_zeromul(InBuf + 0*in_span, 4*in_span, dft_Temp_Buf + 0*8, 1); \
	radix8_0_mul(InBuf + 1*in_span, 4*in_span, dft_Temp_Buf + 1*8, 1, r32twiddle + 2*8*1); \
	radix8_0_mul(InBuf + 2*in_span, 4*in_span, dft_Temp_Buf + 2*8, 1, r32twiddle + 2*8*2); \
	radix8_0_mul(InBuf + 3*in_span, 4*in_span, dft_Temp_Buf + 3*8, 1, r32twiddle + 2*8*3); \
	 \
	radix4_0(dft_Temp_Buf + 0, 8, OutBuf + 0*out_span, 8*out_span); \
	radix4_0(dft_Temp_Buf + 1, 8, OutBuf + 1*out_span, 8*out_span); \
	radix4_0(dft_Temp_Buf + 2, 8, OutBuf + 2*out_span, 8*out_span); \
	radix4_0(dft_Temp_Buf + 3, 8, OutBuf + 3*out_span, 8*out_span); \
	radix4_0(dft_Temp_Buf + 4, 8, OutBuf + 4*out_span, 8*out_span); \
	radix4_0(dft_Temp_Buf + 5, 8, OutBuf + 5*out_span, 8*out_span); \
	radix4_0(dft_Temp_Buf + 6, 8, OutBuf + 6*out_span, 8*out_span); \
	radix4_0(dft_Temp_Buf + 7, 8, OutBuf + 7*out_span, 8*out_span); \
}

#define radix36(InBuf, in_span, OutBuf, out_span, r36twiddle) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 6*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 6*in_span, dft_Temp_Buf + 1*6, 1, r36twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 6*in_span, dft_Temp_Buf + 2*6, 1, r36twiddle + 2*6*2, 1); \
	radix6_0_mul_span(InBuf + 3*in_span, 6*in_span, dft_Temp_Buf + 3*6, 1, r36twiddle + 2*6*3, 1); \
	radix6_0_mul_span(InBuf + 4*in_span, 6*in_span, dft_Temp_Buf + 4*6, 1, r36twiddle + 2*6*4, 1); \
	radix6_0_mul_span(InBuf + 5*in_span, 6*in_span, dft_Temp_Buf + 5*6, 1, r36twiddle + 2*6*5, 1); \
	 \
	radix6_0(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span); \
	radix6_0(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span); \
	radix6_0(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span); \
	radix6_0(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span); \
	radix6_0(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span); \
	radix6_0(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span); \
}

#define radix36_0_zeromul(InBuf, in_span, OutBuf, out_span, r36twiddle) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 6*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 6*in_span, dft_Temp_Buf + 1*6, 1, r36twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 6*in_span, dft_Temp_Buf + 2*6, 1, r36twiddle + 2*6*2, 1); \
	radix6_0_mul_span(InBuf + 3*in_span, 6*in_span, dft_Temp_Buf + 3*6, 1, r36twiddle + 2*6*3, 1); \
	radix6_0_mul_span(InBuf + 4*in_span, 6*in_span, dft_Temp_Buf + 4*6, 1, r36twiddle + 2*6*4, 1); \
	radix6_0_mul_span(InBuf + 5*in_span, 6*in_span, dft_Temp_Buf + 5*6, 1, r36twiddle + 2*6*5, 1); \
	 \
	radix6_0_zeromul(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span); \
}

#define radix36_0_mul(InBuf, in_span, OutBuf, out_span, r36twiddle, twiddle, twiddle_scale) \
{ \
	radix6_0_zeromul(InBuf + 0*in_span, 6*in_span, dft_Temp_Buf + 0*6, 1); \
	radix6_0_mul_span(InBuf + 1*in_span, 6*in_span, dft_Temp_Buf + 1*6, 1, r36twiddle + 2*6*1, 1); \
	radix6_0_mul_span(InBuf + 2*in_span, 6*in_span, dft_Temp_Buf + 2*6, 1, r36twiddle + 2*6*2, 1); \
	radix6_0_mul_span(InBuf + 3*in_span, 6*in_span, dft_Temp_Buf + 3*6, 1, r36twiddle + 2*6*3, 1); \
	radix6_0_mul_span(InBuf + 4*in_span, 6*in_span, dft_Temp_Buf + 4*6, 1, r36twiddle + 2*6*4, 1); \
	radix6_0_mul_span(InBuf + 5*in_span, 6*in_span, dft_Temp_Buf + 5*6, 1, r36twiddle + 2*6*5, 1); \
	 \
	radix6_0_mul_span(dft_Temp_Buf + 0, 6, OutBuf + 0*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*0, 6*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 1, 6, OutBuf + 1*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*1, 6*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 2, 6, OutBuf + 2*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*2, 6*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 3, 6, OutBuf + 3*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*3, 6*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 4, 6, OutBuf + 4*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*4, 6*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 5, 6, OutBuf + 5*out_span, 6*out_span, twiddle + 2*out_span*twiddle_scale*5, 6*out_span*twiddle_scale); \
}

#define radix40_0_zeromul(InBuf, in_span, OutBuf, out_span, r40twiddle) \
{ \
	radix8_0_zeromul(InBuf + 0*in_span, 5*in_span, dft_Temp_Buf + 0*8, 1); \
	radix8_0_mul(InBuf + 1*in_span, 5*in_span, dft_Temp_Buf + 1*8, 1, r40twiddle + 2*8*1); \
	radix8_0_mul(InBuf + 2*in_span, 5*in_span, dft_Temp_Buf + 2*8, 1, r40twiddle + 2*8*2); \
	radix8_0_mul(InBuf + 3*in_span, 5*in_span, dft_Temp_Buf + 3*8, 1, r40twiddle + 2*8*3); \
	radix8_0_mul(InBuf + 4*in_span, 5*in_span, dft_Temp_Buf + 4*8, 1, r40twiddle + 2*8*4); \
	 \
	radix5_0_zeromul(dft_Temp_Buf + 0, 8, OutBuf + 0*out_span, 8*out_span); \
	radix5_0_zeromul(dft_Temp_Buf + 1, 8, OutBuf + 1*out_span, 8*out_span); \
	radix5_0_zeromul(dft_Temp_Buf + 2, 8, OutBuf + 2*out_span, 8*out_span); \
	radix5_0_zeromul(dft_Temp_Buf + 3, 8, OutBuf + 3*out_span, 8*out_span); \
	radix5_0_zeromul(dft_Temp_Buf + 4, 8, OutBuf + 4*out_span, 8*out_span); \
	radix5_0_zeromul(dft_Temp_Buf + 5, 8, OutBuf + 5*out_span, 8*out_span); \
	radix5_0_zeromul(dft_Temp_Buf + 6, 8, OutBuf + 6*out_span, 8*out_span); \
	radix5_0_zeromul(dft_Temp_Buf + 7, 8, OutBuf + 7*out_span, 8*out_span); \
}

#define radix40_0_mul(InBuf, in_span, OutBuf, out_span, r40twiddle, twiddle, twiddle_scale) \
{ \
	radix8_0_zeromul(InBuf + 0*in_span, 5*in_span, dft_Temp_Buf + 0*8, 1); \
	radix8_0_mul(InBuf + 1*in_span, 5*in_span, dft_Temp_Buf + 1*8, 1, r40twiddle + 2*8*1); \
	radix8_0_mul(InBuf + 2*in_span, 5*in_span, dft_Temp_Buf + 2*8, 1, r40twiddle + 2*8*2); \
	radix8_0_mul(InBuf + 3*in_span, 5*in_span, dft_Temp_Buf + 3*8, 1, r40twiddle + 2*8*3); \
	radix8_0_mul(InBuf + 4*in_span, 5*in_span, dft_Temp_Buf + 4*8, 1, r40twiddle + 2*8*4); \
	 \
	radix5_0_mul_span(dft_Temp_Buf + 0, 8, OutBuf + 0*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*0, 8*out_span*twiddle_scale); \
	radix5_mul_span(dft_Temp_Buf + 1, 8, OutBuf + 1*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*1, 8*out_span*twiddle_scale); \
	radix5_mul_span(dft_Temp_Buf + 2, 8, OutBuf + 2*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*2, 8*out_span*twiddle_scale); \
	radix5_mul_span(dft_Temp_Buf + 3, 8, OutBuf + 3*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*3, 8*out_span*twiddle_scale); \
	radix5_mul_span(dft_Temp_Buf + 4, 8, OutBuf + 4*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*4, 8*out_span*twiddle_scale); \
	radix5_mul_span(dft_Temp_Buf + 5, 8, OutBuf + 5*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*5, 8*out_span*twiddle_scale); \
	radix5_mul_span(dft_Temp_Buf + 6, 8, OutBuf + 6*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*6, 8*out_span*twiddle_scale); \
	radix5_mul_span(dft_Temp_Buf + 7, 8, OutBuf + 7*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*7, 8*out_span*twiddle_scale); \
}

#define radix48(InBuf, in_span, OutBuf, out_span, r48twiddle) \
{ \
	radix8_0_zeromul(InBuf + 0*in_span, 6*in_span, dft_Temp_Buf + 0*8, 1); \
	radix8_0_mul(InBuf + 1*in_span, 6*in_span, dft_Temp_Buf + 1*8, 1, r48twiddle + 2*8*1); \
	radix8_0_mul(InBuf + 2*in_span, 6*in_span, dft_Temp_Buf + 2*8, 1, r48twiddle + 2*8*2); \
	radix8_0_mul(InBuf + 3*in_span, 6*in_span, dft_Temp_Buf + 3*8, 1, r48twiddle + 2*8*3); \
	radix8_0_mul(InBuf + 4*in_span, 6*in_span, dft_Temp_Buf + 4*8, 1, r48twiddle + 2*8*4); \
	radix8_0_mul(InBuf + 5*in_span, 6*in_span, dft_Temp_Buf + 5*8, 1, r48twiddle + 2*8*5); \
	 \
	radix6_0(dft_Temp_Buf + 0, 8, OutBuf + 0*out_span, 8*out_span); \
	radix6_0(dft_Temp_Buf + 1, 8, OutBuf + 1*out_span, 8*out_span); \
	radix6_0(dft_Temp_Buf + 2, 8, OutBuf + 2*out_span, 8*out_span); \
	radix6_0(dft_Temp_Buf + 3, 8, OutBuf + 3*out_span, 8*out_span); \
	radix6_0(dft_Temp_Buf + 4, 8, OutBuf + 4*out_span, 8*out_span); \
	radix6_0(dft_Temp_Buf + 5, 8, OutBuf + 5*out_span, 8*out_span); \
	radix6_0(dft_Temp_Buf + 6, 8, OutBuf + 6*out_span, 8*out_span); \
	radix6_0(dft_Temp_Buf + 7, 8, OutBuf + 7*out_span, 8*out_span); \
}

#define radix48_0_zeromul(InBuf, in_span, OutBuf, out_span, r48twiddle) \
{ \
	radix8_0_zeromul(InBuf + 0*in_span, 6*in_span, dft_Temp_Buf + 0*8, 1); \
	radix8_0_mul(InBuf + 1*in_span, 6*in_span, dft_Temp_Buf + 1*8, 1, r48twiddle + 2*8*1); \
	radix8_0_mul(InBuf + 2*in_span, 6*in_span, dft_Temp_Buf + 2*8, 1, r48twiddle + 2*8*2); \
	radix8_0_mul(InBuf + 3*in_span, 6*in_span, dft_Temp_Buf + 3*8, 1, r48twiddle + 2*8*3); \
	radix8_0_mul(InBuf + 4*in_span, 6*in_span, dft_Temp_Buf + 4*8, 1, r48twiddle + 2*8*4); \
	radix8_0_mul(InBuf + 5*in_span, 6*in_span, dft_Temp_Buf + 5*8, 1, r48twiddle + 2*8*5); \
	 \
	radix6_0_zeromul(dft_Temp_Buf + 0, 8, OutBuf + 0*out_span, 8*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 1, 8, OutBuf + 1*out_span, 8*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 2, 8, OutBuf + 2*out_span, 8*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 3, 8, OutBuf + 3*out_span, 8*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 4, 8, OutBuf + 4*out_span, 8*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 5, 8, OutBuf + 5*out_span, 8*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 6, 8, OutBuf + 6*out_span, 8*out_span); \
	radix6_0_zeromul(dft_Temp_Buf + 7, 8, OutBuf + 7*out_span, 8*out_span); \
}

#define radix48_0_mul(InBuf, in_span, OutBuf, out_span, r48twiddle, twiddle, twiddle_scale) \
{ \
	radix8_0_zeromul(InBuf + 0*in_span, 6*in_span, dft_Temp_Buf + 0*8, 1); \
	radix8_0_mul(InBuf + 1*in_span, 6*in_span, dft_Temp_Buf + 1*8, 1, r48twiddle + 2*8*1); \
	radix8_0_mul(InBuf + 2*in_span, 6*in_span, dft_Temp_Buf + 2*8, 1, r48twiddle + 2*8*2); \
	radix8_0_mul(InBuf + 3*in_span, 6*in_span, dft_Temp_Buf + 3*8, 1, r48twiddle + 2*8*3); \
	radix8_0_mul(InBuf + 4*in_span, 6*in_span, dft_Temp_Buf + 4*8, 1, r48twiddle + 2*8*4); \
	radix8_0_mul(InBuf + 5*in_span, 6*in_span, dft_Temp_Buf + 5*8, 1, r48twiddle + 2*8*5); \
	 \
	radix6_0_mul_span(dft_Temp_Buf + 0, 8, OutBuf + 0*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*0, 8*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 1, 8, OutBuf + 1*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*1, 8*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 2, 8, OutBuf + 2*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*2, 8*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 3, 8, OutBuf + 3*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*3, 8*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 4, 8, OutBuf + 4*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*4, 8*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 5, 8, OutBuf + 5*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*5, 8*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 6, 8, OutBuf + 6*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*6, 8*out_span*twiddle_scale); \
	radix6_mul_span(dft_Temp_Buf + 7, 8, OutBuf + 7*out_span, 8*out_span, twiddle + 2*out_span*twiddle_scale*7, 8*out_span*twiddle_scale); \
}

static void dft12(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle)
{
	__m128i dft_Temp_Buf[12];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11;
	radix12((__m128i *)InBuf, 1, (__m128i *)OutBuf, 1, (__m128i *)r12twiddle);
}

static void dft24(__m128i *InBuf, __m128i *OutBuf, __m128i *r24twiddle)
{
	__m128i dft_Temp_Buf[24];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;
	radix24((__m128i *)InBuf, 1, (__m128i *)OutBuf, 1, (__m128i *)r24twiddle);
}

static void dft36(__m128i *InBuf, __m128i *OutBuf, __m128i *r36twiddle)
{
	__m128i dft_Temp_Buf[36];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;
	radix36((__m128i *)InBuf, 1, (__m128i *)OutBuf, 1, (__m128i *)r36twiddle);
}

static void dft48(__m128i *InBuf, __m128i *OutBuf, __m128i *r48twiddle)
{
	__m128i dft_Temp_Buf[48];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;
	radix48((__m128i *)InBuf, 1, (__m128i *)OutBuf, 1, (__m128i *)r48twiddle);
}

static void dft60(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle, __m128i *r5twiddle, __m128i *r720twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[60];
	__m128i dft_Temp_Buf[12];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;

	in_span = 5;
	out_span = 5;
	radix12_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r12twiddle);
	for (i=1;i<5;i++)
	{
		radix12_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r12twiddle, (__m128i *)r720twiddle + 2*4*i, 12);
	}

	in_span = 1;
	out_span = 12;
	for (i=0;i<12;i++)
	{
		radix5_0((__m128i *)(OutTemp) + i*5, in_span,(__m128i *)OutBuf + i, out_span);
	}
}

static void dft72(__m128i *InBuf, __m128i *OutBuf, __m128i *r9twiddle, __m128i *r8twiddle, __m128i *r648twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[72];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 8;
	out_span = 8;
	radix9_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span);
	for (i=1;i<8;i++)
	{
		radix9_0_mul_span((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r648twiddle + 2*3*i, 9);
	}

	in_span = 1;
	out_span = 9;
	for (i=0;i<9;i++)
	{

		radix8_0((__m128i *)(OutTemp) + i*8, in_span,(__m128i *)OutBuf + i, out_span);
	}
}

static void dft96(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle, __m128i *r8twiddle, __m128i *r768twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[96];
	__m128i dft_Temp_Buf[12];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;

	in_span = 8;
	out_span = 8;
	radix12_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r12twiddle);
	for (i=1;i<8;i++)
	{
		radix12_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r12twiddle, (__m128i *)r768twiddle + 2*2*i, 8);
	}

	in_span = 1;
	out_span = 12;
	for (i=0;i<12;i++)
	{
		radix8_0((__m128i *)(OutTemp) + i*8, in_span,(__m128i *)OutBuf + i, out_span);
	}
}

static void dft108(__m128i *InBuf, __m128i *OutBuf, __m128i *r9twiddle, __m128i *r12twiddle, __m128i *r648twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[108];
	__m128i dft_Temp_Buf[12];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 12;
	out_span = 12;
	radix9_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span);
	for (i=1;i<12;i++)
	{
		radix9_0_mul_span((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r648twiddle + 2*2*i, 6);
	}

	in_span = 1;
	out_span = 9;
	for (i=0;i<9;i++)
	{
		radix12((__m128i *)(OutTemp) + i*12, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r12twiddle);
	}
}

static void dft120(__m128i *InBuf, __m128i *OutBuf, __m128i *r24twiddle, __m128i *r5twiddle, __m128i *r1200twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[120];
	__m128i dft_Temp_Buf[24];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 5;
	out_span = 5;
	radix24_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r24twiddle);
	for (i=1;i<5;i++)
	{
		radix24_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r24twiddle, (__m128i *)r1200twiddle + 2*5*i, 10);
	}

	in_span = 1;
	out_span = 24;
	for (i=0;i<24;i++)
	{
		radix5_0((__m128i *)(OutTemp) + i*5, in_span,(__m128i *)OutBuf + i, out_span);
	}
}

static void dft144(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle, __m128i *r864twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[144];
	__m128i dft_Temp_Buf[12];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11;

	in_span = 12;
	out_span = 12;
	radix12_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r12twiddle);
	for (i=1;i<12;i++)
	{
		radix12_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r12twiddle, (__m128i *)r864twiddle + 2*2*i, 6);
	}

	in_span = 1;
	out_span = 12;
	for (i=0;i<12;i++)
	{
		radix12((__m128i *)(OutTemp) + i*12, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r12twiddle);
	}
}

static void dft180(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle, __m128i *r15twiddle, __m128i *r1080twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[180];
	__m128i dft_Temp_Buf[15];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;

	in_span = 15;
	out_span = 15;
	radix12_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r12twiddle);
	for (i=1;i<15;i++)
	{
		radix12_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r12twiddle, (__m128i *)r1080twiddle + 2*2*i, 6);
	}

	in_span = 1;
	out_span = 12;
	for (i=0;i<12;i++)
	{
		radix15((__m128i *)(OutTemp) + i*15, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r15twiddle);
	}
}

static void dft192(__m128i *InBuf, __m128i *OutBuf, __m128i *r16twiddle, __m128i *r12twiddle, __m128i *r1152twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[192];
	__m128i dft_Temp_Buf[16];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11;

	in_span = 12;
	out_span = 12;
	radix16_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r16twiddle);
	for (i=1;i<12;i++)
	{
		radix16_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r16twiddle, (__m128i *)r1152twiddle + 2*2*i, 6);
	}

	in_span = 1;
	out_span = 16;
	for (i=0;i<16;i++)
	{
		radix12((__m128i *)(OutTemp) + i*12, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r12twiddle);
	}
}

static void dft216(__m128i *InBuf, __m128i *OutBuf, __m128i *r18twiddle, __m128i *r12twiddle, __m128i *r864twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[216];
	__m128i dft_Temp_Buf[18];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 12;
	out_span = 12;
	radix18_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r18twiddle);
	for (i=1;i<12;i++)
	{
		radix18_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r18twiddle, (__m128i *)r864twiddle + 2*2*i, 4);
	}

	in_span = 1;
	out_span = 18;
	for (i=0;i<18;i++)
	{
		radix12((__m128i *)(OutTemp) + i*12, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r12twiddle);
	}
}

static void dft240(__m128i *InBuf, __m128i *OutBuf, __m128i *r20twiddle, __m128i *r12twiddle, __m128i *r960twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[240];
	__m128i dft_Temp_Buf[20];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;

	in_span = 12;
	out_span = 12;
	radix20_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r20twiddle);
	for (i=1;i<12;i++)
	{
		radix20_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r20twiddle, (__m128i *)r960twiddle + 2*2*i, 4);
	}

	in_span = 1;
	out_span = 20;
	for (i=0;i<20;i++)
	{
		radix12((__m128i *)(OutTemp) + i*12, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r12twiddle);
	}
}

static void dft288(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle, __m128i *r24twiddle, __m128i *r864twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[288];
	__m128i dft_Temp_Buf[24];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 24;
	out_span = 24;
	radix12_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r12twiddle);
	for (i=1;i<24;i++)
	{
		radix12_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r12twiddle, (__m128i *)r864twiddle + 2*i, 3);
	}

	in_span = 1;
	out_span = 12;
	for (i=0;i<12;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft300(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle, __m128i *r25twiddle, __m128i *r1200twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[300];
	__m128i dft_Temp_Buf[25];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;

	in_span = 25;
	out_span = 25;
	radix12_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r12twiddle);
	for (i=1;i<25;i++)
	{
		radix12_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r12twiddle, (__m128i *)r1200twiddle + 2*i, 4);
	}

	in_span = 1;
	out_span = 12;
	for (i=0;i<12;i++)
	{
		radix25((__m128i *)(OutTemp) + i*25, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r25twiddle);
	}
}

static void dft324(__m128i *InBuf, __m128i *OutBuf, __m128i *r12twiddle, __m128i *r27twiddle, __m128i *r972twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[324];
	__m128i dft_Temp_Buf[27];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 27;
	out_span = 27;
	radix12_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r12twiddle);
	for (i=1;i<27;i++)
	{
		radix12_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r12twiddle, (__m128i *)r972twiddle + 2*i, 3);
	}

	in_span = 1;
	out_span = 12;
	for (i=0;i<12;i++)
	{
		radix27((__m128i *)(OutTemp) + i*27, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r27twiddle);
	}
}

static void dft360(__m128i *InBuf, __m128i *OutBuf, __m128i *r18twiddle, __m128i *r20twiddle, __m128i *r720twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[360];
	__m128i dft_Temp_Buf[20];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 20;
	out_span = 20;
	radix18_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r18twiddle);
	for (i=1;i<20;i++)
	{
		radix18_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r18twiddle, (__m128i *)r720twiddle + 2*i, 2);
	}

	in_span = 1;
	out_span = 18;
	for (i=0;i<18;i++)
	{
		radix20((__m128i *)(OutTemp) + i*20, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r20twiddle);
	}
}

static void dft384(__m128i *InBuf, __m128i *OutBuf, __m128i *r16twiddle, __m128i *r24twiddle, __m128i *r1152twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[384];
	__m128i dft_Temp_Buf[24];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 24;
	out_span = 24;
	radix16_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r16twiddle);
	for (i=1;i<24;i++)
	{
		radix16_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r16twiddle, (__m128i *)r1152twiddle + 2*i, 3);
	}

	in_span = 1;
	out_span = 16;
	for (i=0;i<16;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft432(__m128i *InBuf, __m128i *OutBuf, __m128i *r18twiddle, __m128i *r24twiddle, __m128i *r864twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[432];
	__m128i dft_Temp_Buf[24];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 24;
	out_span = 24;
	radix18_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r18twiddle);
	for (i=1;i<24;i++)
	{
		radix18_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r18twiddle, (__m128i *)r864twiddle + 2*i, 2);
	}

	in_span = 1;
	out_span = 18;
	for (i=0;i<18;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft480(__m128i *InBuf, __m128i *OutBuf, __m128i *r20twiddle, __m128i *r24twiddle, __m128i *r960twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[960];
	__m128i dft_Temp_Buf[24];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 24;
	out_span = 24;
	radix20_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r20twiddle);
	for (i=1;i<24;i++)
	{
		radix20_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r20twiddle, (__m128i *)r960twiddle + 2*i, 2);
	}

	in_span = 1;
	out_span = 20;
	for (i=0;i<20;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft540(__m128i *InBuf, __m128i *OutBuf, __m128i *r18twiddle, __m128i *r30twiddle, __m128i *r1080twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[1080];
	__m128i dft_Temp_Buf[30];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12,m128_t13;

	in_span = 30;
	out_span = 30;
	radix18_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r18twiddle);
	for (i=1;i<30;i++)
	{
		radix18_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r18twiddle, (__m128i *)r1080twiddle + 2*i, 2);
	}

	in_span = 1;
	out_span = 18;
	for (i=0;i<18;i++)
	{
		radix30((__m128i *)(OutTemp) + i*30, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r30twiddle);
	}
}

static void dft576(__m128i *InBuf, __m128i *OutBuf, __m128i *r24twiddle, __m128i *r1152twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[576];
	__m128i dft_Temp_Buf[24];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 24;
	out_span = 24;
	radix24_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r24twiddle);
	for (i=1;i<24;i++)
	{
		radix24_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r24twiddle, (__m128i *)r1152twiddle + 2*i, 2);
	}

	in_span = 1;
	out_span = 24;
	for (i=0;i<24;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft600(__m128i *InBuf, __m128i *OutBuf, __m128i *r24twiddle, __m128i *r25twiddle, __m128i *r1200twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[600];
	__m128i dft_Temp_Buf[25];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 25;
	out_span = 25;
	radix24_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r24twiddle);
	for (i=1;i<25;i++)
	{
		radix24_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r24twiddle, (__m128i *)r1200twiddle + 2*i, 2);
	}

	in_span = 1;
	out_span = 24;
	for (i=0;i<24;i++)
	{
		radix25((__m128i *)(OutTemp) + i*25, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r25twiddle);
	}
}

static void dft648(__m128i *InBuf, __m128i *OutBuf, __m128i *r27twiddle, __m128i *r24twiddle, __m128i *r648twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[648];
	__m128i dft_Temp_Buf[27];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 24;
	out_span = 24;
	radix27_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r27twiddle);
	for (i=1;i<24;i++)
	{
		radix27_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r27twiddle, (__m128i *)r648twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 27;
	for (i=0;i<27;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft720(__m128i *InBuf, __m128i *OutBuf, __m128i *r36twiddle, __m128i *r20twiddle, __m128i *r720twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[720];
	__m128i dft_Temp_Buf[36];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 20;
	out_span = 20;
	radix36_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r36twiddle);
	for (i=1;i<20;i++)
	{
		radix36_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r36twiddle, (__m128i *)r720twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 36;
	for (i=0;i<36;i++)
	{
		radix20((__m128i *)(OutTemp) + i*20, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r20twiddle);
	}
}

static void dft768(__m128i *InBuf, __m128i *OutBuf, __m128i *r48twiddle, __m128i *r16twiddle, __m128i *r768twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[768];
	__m128i dft_Temp_Buf[48];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 16;
	out_span = 16;
	radix48_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r48twiddle);
	for (i=1;i<16;i++)
	{
		radix48_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r48twiddle, (__m128i *)r768twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 48;
	for (i=0;i<48;i++)
	{
		radix16((__m128i *)(OutTemp) + i*16, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r16twiddle);
	}
}

static void dft864(__m128i *InBuf, __m128i *OutBuf, __m128i *r36twiddle, __m128i *r24twiddle, __m128i *r864twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[864];
	__m128i dft_Temp_Buf[36];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 24;
	out_span = 24;
	radix36_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r36twiddle);
	for (i=1;i<24;i++)
	{
		radix36_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r36twiddle, (__m128i *)r864twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 36;
	for (i=0;i<36;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft900(__m128i *InBuf, __m128i *OutBuf, __m128i *r36twiddle, __m128i *r25twiddle, __m128i *r900twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[900];
	__m128i dft_Temp_Buf[36];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 25;
	out_span = 25;
	radix36_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r36twiddle);
	for (i=1;i<25;i++)
	{
		radix36_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r36twiddle, (__m128i *)r900twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 36;
	for (i=0;i<36;i++)
	{
		radix25((__m128i *)(OutTemp) + i*25, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r25twiddle);
	}
}

static void dft960(__m128i *InBuf, __m128i *OutBuf, __m128i *r40twiddle, __m128i *r24twiddle, __m128i *r960twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[960];
	__m128i dft_Temp_Buf[40];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 24;
	out_span = 24;
	radix40_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r40twiddle);
	for (i=1;i<24;i++)
	{
		radix40_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r40twiddle, (__m128i *)r960twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 40;
	for (i=0;i<40;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft972(__m128i *InBuf, __m128i *OutBuf, __m128i *r36twiddle, __m128i *r27twiddle, __m128i *r972twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[972];
	__m128i dft_Temp_Buf[36];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 27;
	out_span = 27;
	radix36_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r36twiddle);
	for (i=1;i<27;i++)
	{
		radix36_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r36twiddle, (__m128i *)r972twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 36;
	for (i=0;i<36;i++)
	{
		radix27((__m128i *)(OutTemp) + i*27, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r27twiddle);
	}
}

static void dft1080(__m128i *InBuf, __m128i *OutBuf, __m128i *r36twiddle, __m128i *r30twiddle, __m128i *r1080twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[1080];
	__m128i dft_Temp_Buf[36];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 30;
	out_span = 30;
	radix36_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r36twiddle);
	for (i=1;i<30;i++)
	{
		radix36_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r36twiddle, (__m128i *)r1080twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 36;
	for (i=0;i<36;i++)
	{
		radix30((__m128i *)(OutTemp) + i*30, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r30twiddle);
	}
}

static void dft1152(__m128i *InBuf, __m128i *OutBuf, __m128i *r48twiddle, __m128i *r24twiddle, __m128i *r1152twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[1152];
	__m128i dft_Temp_Buf[48];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 24;
	out_span = 24;
	radix48_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r48twiddle);
	for (i=1;i<24;i++)
	{
		radix48_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r48twiddle, (__m128i *)r1152twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 48;
	for (i=0;i<48;i++)
	{
		radix24((__m128i *)(OutTemp) + i*24, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r24twiddle);
	}
}

static void dft1200(__m128i *InBuf, __m128i *OutBuf, __m128i *r48twiddle, __m128i *r25twiddle, __m128i *r1200twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[1200];
	__m128i dft_Temp_Buf[48];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 25;
	out_span = 25;
	radix48_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r48twiddle);
	for (i=1;i<25;i++)
	{
		radix48_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r48twiddle, (__m128i *)r1200twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 48;
	for (i=0;i<48;i++)
	{
		radix25((__m128i *)(OutTemp) + i*25, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r25twiddle);
	}
}

static void dft1536(__m128i *InBuf, __m128i *OutBuf, __m128i *r48twiddle, __m128i *r32twiddle, __m128i *r1536twiddle)
{
    WORD32 i;
    WORD32 in_span, out_span;
	__m128i OutTemp[1536];
    __m128i dft_Temp_Buf[48];
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12, m128_t13;

	in_span = 32;
    out_span = 32;
    radix48_0_zeromul((__m128i *)(InBuf), in_span, (__m128i *)OutTemp, out_span, (__m128i *)r48twiddle);
	for (i=1;i<32;i++)
	{
		radix48_0_mul((__m128i *)(InBuf) + i, in_span, (__m128i *)OutTemp + i, out_span, (__m128i *)r48twiddle, (__m128i *)r1536twiddle + 2*i, 1);
	}

	in_span = 1;
	out_span = 48;
	for (i=0;i<48;i++)
	{
		radix32((__m128i *)(OutTemp) + i*32, in_span,(__m128i *)OutBuf + i, out_span, (__m128i *)r32twiddle);
	}
}

void gen_dft(__m128i *InBuf, __m128i *OutBuf, DFTtwiddleStruct * psTwiddleBuffer, WORD32 dft_idx)
{
    switch(dft_idx)
    {
    case 1536:
        dft1536((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r48twiddle, (__m128i *)psTwiddleBuffer->pDft_r32twiddle, (__m128i *)psTwiddleBuffer->pDft_r1536twiddle);
        break;
    case 1200:
        dft1200((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r48twiddle, (__m128i *)psTwiddleBuffer->pDft_r25twiddle, (__m128i *)psTwiddleBuffer->pDft_r1200twiddle);
        break;
    case 1152:
        dft1152((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r48twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r1152twiddle);
        break;
    case 1080:
        dft1080((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r36twiddle, (__m128i *)psTwiddleBuffer->pDft_r30twiddle, (__m128i *)psTwiddleBuffer->pDft_r1080twiddle);
        break;
    case 972:
        dft972((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r36twiddle, (__m128i *)psTwiddleBuffer->pDft_r27twiddle, (__m128i *)psTwiddleBuffer->pDft_r972twiddle);
        break;
    case 960:
        dft960((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r40twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r960twiddle);
        break;
    case 900:
        dft900((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r36twiddle, (__m128i *)psTwiddleBuffer->pDft_r25twiddle, (__m128i *)psTwiddleBuffer->pDft_r900twiddle);
        break;
    case 864:
        dft864((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r36twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r864twiddle);
        break;
    case 768:
        dft768((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r48twiddle, (__m128i *)psTwiddleBuffer->pDft_r16twiddle, (__m128i *)psTwiddleBuffer->pDft_r768twiddle);
        break;
    case 720:
        dft720((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r36twiddle, (__m128i *)psTwiddleBuffer->pDft_r20twiddle, (__m128i *)psTwiddleBuffer->pDft_r720twiddle);
        break;
    case 648:
        dft648((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r27twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r648twiddle);
        break;
    case 600:
        dft600((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r25twiddle, (__m128i *)psTwiddleBuffer->pDft_r1200twiddle);
        break;
    case 576:
        dft576((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r1152twiddle);
        break;
    case 540:
        dft540((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r18twiddle, (__m128i *)psTwiddleBuffer->pDft_r30twiddle, (__m128i *)psTwiddleBuffer->pDft_r1080twiddle);
        break;
    case 480:
        dft480((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r20twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r960twiddle);
        break;
    case 432:
        dft432((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r18twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r864twiddle);
        break;
    case 384:
        dft384((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r16twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r1152twiddle);
        break;
    case 360:
        dft360((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r18twiddle, (__m128i *)psTwiddleBuffer->pDft_r20twiddle, (__m128i *)psTwiddleBuffer->pDft_r720twiddle);
        break;
    case 324:
        dft324((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r27twiddle, (__m128i *)psTwiddleBuffer->pDft_r972twiddle);
        break;
    case 300:
        dft300((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r25twiddle, (__m128i *)psTwiddleBuffer->pDft_r1200twiddle);
        break;
    case 288:
        dft288((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, (__m128i *)psTwiddleBuffer->pDft_r864twiddle);
        break;
    case 240:
        dft240((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r20twiddle, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r960twiddle);
        break;
    case 216:
        dft216((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r18twiddle, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r864twiddle);
        break;
    case 192:
        dft192((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r16twiddle, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r1152twiddle);
        break;
    case 180:
        dft180((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r15twiddle, (__m128i *)psTwiddleBuffer->pDft_r1080twiddle);
        break;
    case 144:
        dft144((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r864twiddle);
        break;
    case 120:
        dft120((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r24twiddle, NULL, (__m128i *)psTwiddleBuffer->pDft_r1200twiddle);
        break;
    case 108:
        dft108((__m128i *)InBuf, (__m128i *)OutBuf, NULL, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, (__m128i *)psTwiddleBuffer->pDft_r648twiddle);
        break;
    case 96:
        dft96((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, NULL, (__m128i *)psTwiddleBuffer->pDft_r768twiddle);
        break;
    case 72:
        dft72((__m128i *)InBuf, (__m128i *)OutBuf, NULL, NULL, (__m128i *)psTwiddleBuffer->pDft_r648twiddle);
        break;
    case 60:
        dft60((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle, NULL, (__m128i *)psTwiddleBuffer->pDft_r720twiddle);
        break;
    case 48:
        dft48((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r48twiddle);
        break;
    case 36:
        dft36((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r36twiddle);
        break;
    case 24:
        dft24((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r24twiddle);
        break;
    case 12:
        dft12((__m128i *)InBuf, (__m128i *)OutBuf, (__m128i *)psTwiddleBuffer->pDft_r12twiddle);
        break;
    default:
        printf("the points number %d of DFT is wrong, please check it!\n", dft_idx); 
        break;
    }
    return;
}
