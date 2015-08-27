#ifndef _COMMON_STRUCTURE_H
#define _COMMON_STRUCTURE_H

#define FFTLIB_ALIGNTO 64

#include "common_typedef.h"

#define _IDFT_1200_     (1200)
#define _IDFT_1152_     (1152)
#define _IDFT_1080_     (1080)
#define _IDFT_972_      (972) 
#define _IDFT_960_      (960) 
#define _IDFT_900_      (900) 
#define _IDFT_864_      (864) 
#define _IDFT_768_      (768) 
#define _IDFT_720_      (720) 
#define _IDFT_648_      (648) 
#define _IDFT_600_      (600) 
#define _IDFT_576_      (576) 
#define _IDFT_540_      (540) 
#define _IDFT_480_      (480) 
#define _IDFT_432_      (432) 
#define _IDFT_384_      (384) 
#define _IDFT_360_      (360) 
#define _IDFT_324_      (324) 
#define _IDFT_300_      (300) 
#define _IDFT_288_      (288) 
#define _IDFT_240_      (240) 
#define _IDFT_216_      (216) 
#define _IDFT_192_      (192) 
#define _IDFT_180_      (180) 
#define _IDFT_144_      (144) 
#define _IDFT_120_      (120) 
#define _IDFT_108_      (108) 
#define _IDFT_96_       (96)  
#define _IDFT_72_       (72)  
#define _IDFT_60_       (60)  
#define _IDFT_48_       (48)  
#define _IDFT_36_       (36)  
#define _IDFT_24_       (24)  
#define _IDFT_12_       (12)  

typedef struct
{
    WORD16 re;				// real part
    WORD16 im;				// imag part
}Complex;

typedef struct
{
    // DFT buffers
    __m128i * pDft_r12twiddle; //[12][2];
    __m128i * pDft_r15twiddle; //[15][2];
    __m128i * pDft_r16twiddle; //[16][2];
    __m128i * pDft_r18twiddle; //[18][2];
    __m128i * pDft_r20twiddle; //[20][2];
    __m128i * pDft_r24twiddle; //[24][2];
    __m128i * pDft_r25twiddle; //[25][2];
    __m128i * pDft_r27twiddle; //[27][2];
    __m128i * pDft_r30twiddle; //[30][2];
    __m128i * pDft_r32twiddle; //[32][2];
    __m128i * pDft_r36twiddle; //[36][2];
    __m128i * pDft_r40twiddle; //[40][2];
    __m128i * pDft_r48twiddle; //[48][2];

    __m128i * pDft_r648twiddle; //[648][2];
    __m128i * pDft_r720twiddle; //[720][2];
    __m128i * pDft_r768twiddle; //[768][2];
    __m128i * pDft_r864twiddle; //[864][2];
    __m128i * pDft_r900twiddle; //[900][2];
    __m128i * pDft_r960twiddle; //[960][2];
    __m128i * pDft_r972twiddle; //[972][2];
    __m128i * pDft_r1080twiddle; //[1080][2];
    __m128i * pDft_r1152twiddle; //[1152][2];
    __m128i * pDft_r1200twiddle; //[1200][2];
    __m128i * pDft_r1536twiddle; //[1536][2];

    // IDFT buffers

    __m128i * pIdft_r12twiddle; //[12][2];
    __m128i * pIdft_r15twiddle; //[15][2];
    __m128i * pIdft_r16twiddle; //[16][2];
    __m128i * pIdft_r18twiddle; //[18][2];
    __m128i * pIdft_r20twiddle; //[20][2];
    __m128i * pIdft_r24twiddle; //[24][2];
    __m128i * pIdft_r25twiddle; //[25][2];
    __m128i * pIdft_r27twiddle; //[27][2];
    __m128i * pIdft_r30twiddle; //[30][2];
    __m128i * pIdft_r32twiddle; //[32][2];
    __m128i * pIdft_r36twiddle; //[36][2];
    __m128i * pIdft_r40twiddle; //[40][2];
    __m128i * pIdft_r48twiddle; //[48][2];

    __m128i * pIdft_r648twiddle; //[648][2];
    __m128i * pIdft_r720twiddle; //[720][2];
    __m128i * pIdft_r768twiddle; //[768][2];
    __m128i * pIdft_r864twiddle; //[864][2];
    __m128i * pIdft_r900twiddle; //[900][2];
    __m128i * pIdft_r960twiddle; //[960][2];
    __m128i * pIdft_r972twiddle; //[972][2];
    __m128i * pIdft_r1080twiddle; //[1080][2];
    __m128i * pIdft_r1152twiddle; //[1152][2];
    __m128i * pIdft_r1200twiddle; //[1200][2]
    __m128i * pIdft_r1536twiddle; //[1536][2]

}DFTtwiddleStruct;

#endif
