#ifndef _COMMON_FUNCTION_H_
#define _COMMON_FUNCTION_H_
#include "common_typedef.h"

void init_dft12_twiddle_factor(WORD16 *r12twiddle);
void init_dft15_twiddle_factor(WORD16 *r15twiddle);
void init_dft16_twiddle_factor(WORD16 *r16twiddle);
void init_dft18_twiddle_factor(WORD16 *r18twiddle);
void init_dft20_twiddle_factor(WORD16 *r20twiddle);
void init_dft24_twiddle_factor(WORD16 *r24twiddle);
void init_dft25_twiddle_factor(WORD16 *r25twiddle);
void init_dft27_twiddle_factor(WORD16 *r27twiddle);
void init_dft30_twiddle_factor(WORD16 *r30twiddle);
void init_dft32_twiddle_factor(WORD16 *r32twiddle);
void init_dft36_twiddle_factor(WORD16 *r36twiddle);
void init_dft40_twiddle_factor(WORD16 *r40twiddle);
void init_dft48_twiddle_factor(WORD16 *r48twiddle);
void init_dft648_twiddle_factor(WORD16 *r648twiddle);
void init_dft720_twiddle_factor(WORD16 *r720twiddle);
void init_dft768_twiddle_factor(WORD16 *r768twiddle);
void init_dft864_twiddle_factor(WORD16 *r864twiddle);
void init_dft900_twiddle_factor(WORD16 *r900twiddle);
void init_dft960_twiddle_factor(WORD16 *r960twiddle);
void init_dft972_twiddle_factor(WORD16 *r972twiddle);
void init_dft1080_twiddle_factor(WORD16 *r1080twiddle);
void init_dft1152_twiddle_factor(WORD16 *r1152twiddle);
void init_dft1200_twiddle_factor(WORD16 *r1200twiddle);
void init_dft1536_twiddle_factor(WORD16 *r1536twiddle);
void gen_dft(__m128i *InBuf, __m128i *OutBuf, DFTtwiddleStruct*psTwiddleBuffer, WORD32 dft_idx);

// IDFT
void init_idft12_twiddle_factor(WORD16 *r12twiddle);
void init_idft15_twiddle_factor(WORD16 *r15twiddle);
void init_idft16_twiddle_factor(WORD16 *r16twiddle);
void init_idft18_twiddle_factor(WORD16 *r18twiddle);
void init_idft20_twiddle_factor(WORD16 *r20twiddle);
void init_idft24_twiddle_factor(WORD16 *r24twiddle);
void init_idft25_twiddle_factor(WORD16 *r25twiddle);
void init_idft27_twiddle_factor(WORD16 *r27twiddle);
void init_idft30_twiddle_factor(WORD16 *r30twiddle);
void init_idft32_twiddle_factor(WORD16 *r32twiddle);
void init_idft36_twiddle_factor(WORD16 *r36twiddle);
void init_idft40_twiddle_factor(WORD16 *r40twiddle);
void init_idft48_twiddle_factor(WORD16 *r48twiddle);
void init_idft648_twiddle_factor(WORD16 *r648twiddle);
void init_idft720_twiddle_factor(WORD16 *r720twiddle);
void init_idft768_twiddle_factor(WORD16 *r768twiddle);
void init_idft864_twiddle_factor(WORD16 *r864twiddle);
void init_idft900_twiddle_factor(WORD16 *r900twiddle);
void init_idft960_twiddle_factor(WORD16 *r960twiddle);
void init_idft972_twiddle_factor(WORD16 *r972twiddle);
void init_idft1080_twiddle_factor(WORD16 *r1080twiddle);
void init_idft1152_twiddle_factor(WORD16 *r1152twiddle);
void init_idft1200_twiddle_factor(WORD16 *r1200twiddle);
void init_idft1536_twiddle_factor(WORD16 *r1536twiddle);
void gen_idft(__m128i *InBuf, __m128i *OutBuf, DFTtwiddleStruct *psTwiddleBuffer,  WORD32 idft_idx);

    
// IFFT
void init_ifft2048_twiddle_factor(WORD16 * ifft2048_r2048twiddle, 
    WORD16 * ifft2048_r32twiddle, WORD16* ifft2048_r64twiddle);
void ifft2048(__m128i *InBuf, __m128i *OutBuf, __m128i *ifft2048_r32twiddle, 
    __m128i *ifft2048_r64twiddle, __m128i *ifft2048_r2048twiddle);
void init_ifft1024_twiddle_factor(WORD16 * ifft1024_r1024twiddle, WORD16 * ifft1024_r32twiddle);
void ifft1024(__m128i *InBuf, __m128i *OutBuf, __m128i *ifft1024_r32twiddle, __m128i *ifft1024_r1024twiddle);
    
    
// FFT

extern "C" void fft2048(__m128i *InBuf, __m128i *OutBuf, __m128i *r32twiddle, __m128i *r64twiddle, __m128i *r2048twiddle);

void fft2048_core(__m128i *InBuf, __m128i *OutBuf, 
    __m128i *r32twiddle, __m128i *r64twiddle, __m128i *r2048twiddle);
void fft2048_core_cpy(__m128i *InBuf, __m128i *OutBuf, __m128i *r32twiddle, __m128i *r64twiddle, __m128i *r2048twiddle, __m128i *InBufCpy);
void init_twiddle_factor(WORD16 * r2048twiddle, WORD16 * r32twiddle, 
    WORD16* r64twiddle);
void init_fft2048_twiddle_factor(WORD16 * r2048twiddle, WORD16 * r32twiddle,
    WORD16* r64twiddle);
void fft1024_core(__m128i *InBuf, __m128i *OutBuf, __m128i *r32twiddle, __m128i *r32twiddle_core, __m128i *r1024twiddle);
void fft1024_core_cpy(__m128i *InBuf, __m128i *OutBuf, __m128i *r32twiddle, __m128i *r32twiddle_core, __m128i *r1024twiddle,  __m128i *InBufCpy);
void init_fft1024_wHscShift_twiddle_factor(WORD16 * r1024twiddle, WORD16 * r32twiddle, WORD16 * r32twiddle_core);

 void fftshift2048(Complex *pSymbol);
 void fftshift1024(Complex *pSymbol);


//dct
void init_idct_dct_table(WORD16 *psIdctDctTable, WORD16 idctDctSize);
void gen_dct(__m128i *InBuf,__m128i *OutBuf,WORD32 dctSize);
void gen_idct(__m128i *InBuf,__m128i *OutBuf,WORD32 dctSize);

extern "C" void findCmax2048( void *in, int *iMax, int *avg );
extern "C" void mul( void *ina, void *inb, void *out, int len );
extern "C" void mulconj( void *ina, void *inb, void *out, int len );
extern "C" void fft8192( void *in, void *out );
extern "C" void fft2048_n( void *in, void *out );
extern "C" void ifft2048( void *in, void *out );

#endif

