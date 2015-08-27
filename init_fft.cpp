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

// FFT twiddle
__m128i g_fft2048_r2048twiddle[512][2], g_fft2048_r32twiddle[32][2], g_fft2048_r64twiddle[64][2];
__m128i g_FFTr2048twiddle[512], g_FFTr32twiddle[32], g_FFTr64twiddle[64];
__m128i g_r1024twiddle[256][2], g_r32twiddle[32][2];
__m128i g_FFT1024r1024twiddle[256][2], g_FFT1024r32twiddle[32], g_FFT1024r32twiddle_core[32];

// IFFT twiddle 
__m128i g_ifft2048_r2048twiddle[512][2], g_ifft2048_r32twiddle[32][2], g_ifft2048_r64twiddle[64][2];
__m128i g_ifft1024_r1024twiddle[256][2], g_ifft1024_r32twiddle[32][2];


DFTtwiddleStruct g_sDFTtwiddle;

static __m128i Dft_r12twiddle[12][2];
static __m128i Dft_r15twiddle[15][2];
static __m128i Dft_r16twiddle[16][2];
static __m128i Dft_r18twiddle[18][2];
static __m128i Dft_r20twiddle[20][2];
static __m128i Dft_r24twiddle[24][2];
static __m128i Dft_r25twiddle[25][2];
static __m128i Dft_r27twiddle[27][2];
static __m128i Dft_r30twiddle[30][2];
static __m128i Dft_r32twiddle[32][2];
static __m128i Dft_r36twiddle[36][2];
static __m128i Dft_r40twiddle[40][2];
static __m128i Dft_r48twiddle[48][2];
static __m128i Dft_r648twiddle[648][2];
static __m128i Dft_r720twiddle[720][2];
static __m128i Dft_r768twiddle[768][2];
static __m128i Dft_r864twiddle[864][2];
static __m128i Dft_r900twiddle[900][2];
static __m128i Dft_r960twiddle[960][2];
static __m128i Dft_r972twiddle[972][2];
static __m128i Dft_r1080twiddle[1080][2];
static __m128i Dft_r1152twiddle[1152][2];
static __m128i Dft_r1200twiddle[1200][2];
static __m128i Dft_r1536twiddle[1536][2];

static __m128i Idft_r12twiddle[12][2];
static __m128i Idft_r15twiddle[15][2];
static __m128i Idft_r16twiddle[16][2];
static __m128i Idft_r18twiddle[18][2];
static __m128i Idft_r20twiddle[20][2];
static __m128i Idft_r24twiddle[24][2];
static __m128i Idft_r25twiddle[25][2];
static __m128i Idft_r27twiddle[27][2];
static __m128i Idft_r30twiddle[30][2];
static __m128i Idft_r32twiddle[32][2];
static __m128i Idft_r36twiddle[36][2];
static __m128i Idft_r40twiddle[40][2];
static __m128i Idft_r48twiddle[48][2];
static __m128i Idft_r648twiddle[648][2];
static __m128i Idft_r720twiddle[720][2];
static __m128i Idft_r768twiddle[768][2];
static __m128i Idft_r864twiddle[864][2];
static __m128i Idft_r900twiddle[900][2];
static __m128i Idft_r960twiddle[960][2];
static __m128i Idft_r972twiddle[972][2];
static __m128i Idft_r1080twiddle[1080][2];
static __m128i Idft_r1152twiddle[1152][2];
static __m128i Idft_r1200twiddle[1200][2];
static __m128i Idft_r1536twiddle[1536][2];

__m128i aIdctDctTable1200[300] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable1152[288] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable1080[270] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable972[243] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable960[240] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable900[225] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable864[216] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable768[192] __attribute__((aligned(FFTLIB_ALIGNTO)));

__m128i aIdctDctTable720[180] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable648[162] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable600[150] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable576[144] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable540[135] __attribute__((aligned(FFTLIB_ALIGNTO)));

__m128i aIdctDctTable480[120] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable432[108] __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable384[96]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable360[90]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable324[81]  __attribute__((aligned(FFTLIB_ALIGNTO)));
                             
__m128i aIdctDctTable300[75]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable288[72]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable240[60]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable216[54]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable192[48]  __attribute__((aligned(FFTLIB_ALIGNTO)));

__m128i aIdctDctTable180[45]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable144[36]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable120[30]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable108[27]  __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable96[24]   __attribute__((aligned(FFTLIB_ALIGNTO))); 
                             
__m128i aIdctDctTable72[18]   __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable60[15]   __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable48[12]   __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable36[9]    __attribute__((aligned(FFTLIB_ALIGNTO)));
__m128i aIdctDctTable24[6]    __attribute__((aligned(FFTLIB_ALIGNTO)));

extern "C" void InitFFTLIBTables()
{
    // FFT twiddle 
    init_twiddle_factor((WORD16 *)g_FFTr2048twiddle,
                            (WORD16 *)g_FFTr32twiddle,
                            (WORD16 *)g_FFTr64twiddle);
    init_fft1024_wHscShift_twiddle_factor((WORD16 *)g_FFT1024r1024twiddle,
                            (WORD16 *)g_FFT1024r32twiddle,
                            (WORD16 *)g_FFT1024r32twiddle_core);
    
    init_fft2048_twiddle_factor((WORD16 *)g_fft2048_r2048twiddle, 
                (WORD16 *)g_fft2048_r32twiddle, (WORD16 *)g_fft2048_r64twiddle);

    // IFFT twiddle
    init_ifft2048_twiddle_factor((WORD16 * )g_ifft2048_r2048twiddle, 
                (WORD16 * )g_ifft2048_r32twiddle, (WORD16 * )g_ifft2048_r64twiddle);
    init_ifft1024_twiddle_factor((WORD16 * )g_ifft1024_r1024twiddle,
                (WORD16 * )g_ifft1024_r32twiddle);

    // DFT twiddle
    init_dft12_twiddle_factor((WORD16 *)(&Dft_r12twiddle[0][0]));
    init_dft15_twiddle_factor((WORD16 *)(&Dft_r15twiddle[0][0]));
    init_dft16_twiddle_factor((WORD16 *)(&Dft_r16twiddle[0][0]));
    init_dft18_twiddle_factor((WORD16 *)(&Dft_r18twiddle[0][0]));
    init_dft20_twiddle_factor((WORD16 *)(&Dft_r20twiddle[0][0]));
    init_dft24_twiddle_factor((WORD16 *)(&Dft_r24twiddle[0][0]));
    init_dft25_twiddle_factor((WORD16 *)(&Dft_r25twiddle[0][0]));
    init_dft27_twiddle_factor((WORD16 *)(&Dft_r27twiddle[0][0]));
    init_dft30_twiddle_factor((WORD16 *)(&Dft_r30twiddle[0][0]));
    init_dft32_twiddle_factor((WORD16 *)(&Dft_r32twiddle[0][0]));
    init_dft36_twiddle_factor((WORD16 *)(&Dft_r36twiddle[0][0]));
    init_dft40_twiddle_factor((WORD16 *)(&Dft_r40twiddle[0][0]));
    init_dft48_twiddle_factor((WORD16 *)(&Dft_r48twiddle[0][0]));
    init_dft648_twiddle_factor((WORD16 *)(&Dft_r648twiddle[0][0]));
    init_dft720_twiddle_factor((WORD16 *)(&Dft_r720twiddle[0][0]));
    init_dft768_twiddle_factor((WORD16 *)(&Dft_r768twiddle[0][0]));
    init_dft864_twiddle_factor((WORD16 *)(&Dft_r864twiddle[0][0]));
    init_dft900_twiddle_factor((WORD16 *)(&Dft_r900twiddle[0][0]));
    init_dft960_twiddle_factor((WORD16 *)(&Dft_r960twiddle[0][0]));
    init_dft972_twiddle_factor((WORD16 *)(&Dft_r972twiddle[0][0]));
    init_dft1080_twiddle_factor((WORD16 *)(&Dft_r1080twiddle[0][0]));
    init_dft1152_twiddle_factor((WORD16 *)(&Dft_r1152twiddle[0][0]));
    init_dft1200_twiddle_factor((WORD16 *)(&Dft_r1200twiddle[0][0]));
    init_dft1536_twiddle_factor((WORD16 *)(&Dft_r1536twiddle[0][0]));
    g_sDFTtwiddle.pDft_r12twiddle = &Dft_r12twiddle[0][0];
    g_sDFTtwiddle.pDft_r15twiddle = &Dft_r15twiddle[0][0];
    g_sDFTtwiddle.pDft_r16twiddle = &Dft_r16twiddle[0][0];
    g_sDFTtwiddle.pDft_r18twiddle = &Dft_r18twiddle[0][0];
    g_sDFTtwiddle.pDft_r20twiddle = &Dft_r20twiddle[0][0];
    g_sDFTtwiddle.pDft_r24twiddle = &Dft_r24twiddle[0][0];
    g_sDFTtwiddle.pDft_r25twiddle = &Dft_r25twiddle[0][0];
    g_sDFTtwiddle.pDft_r27twiddle = &Dft_r27twiddle[0][0];
    g_sDFTtwiddle.pDft_r30twiddle = &Dft_r30twiddle[0][0];
    g_sDFTtwiddle.pDft_r32twiddle = &Dft_r32twiddle[0][0];
    g_sDFTtwiddle.pDft_r36twiddle = &Dft_r36twiddle[0][0];
    g_sDFTtwiddle.pDft_r40twiddle = &Dft_r40twiddle[0][0];
    g_sDFTtwiddle.pDft_r48twiddle = &Dft_r48twiddle[0][0];
    g_sDFTtwiddle.pDft_r648twiddle = &Dft_r648twiddle[0][0];
    g_sDFTtwiddle.pDft_r720twiddle = &Dft_r720twiddle[0][0];
    g_sDFTtwiddle.pDft_r768twiddle = &Dft_r768twiddle[0][0];
    g_sDFTtwiddle.pDft_r864twiddle = &Dft_r864twiddle[0][0];
    g_sDFTtwiddle.pDft_r900twiddle = &Dft_r900twiddle[0][0];
    g_sDFTtwiddle.pDft_r960twiddle = &Dft_r960twiddle[0][0];
    g_sDFTtwiddle.pDft_r972twiddle = &Dft_r972twiddle[0][0];
    g_sDFTtwiddle.pDft_r1080twiddle = &Dft_r1080twiddle[0][0];
    g_sDFTtwiddle.pDft_r1152twiddle = &Dft_r1152twiddle[0][0];
    g_sDFTtwiddle.pDft_r1200twiddle = &Dft_r1200twiddle[0][0];
    g_sDFTtwiddle.pDft_r1536twiddle = &Dft_r1536twiddle[0][0];
    // IDFT twiddle
    init_idft12_twiddle_factor((WORD16 *)(&Idft_r12twiddle[0][0]));
    init_idft15_twiddle_factor((WORD16 *)(&Idft_r15twiddle[0][0]));
    init_idft16_twiddle_factor((WORD16 *)(&Idft_r16twiddle[0][0]));
    init_idft18_twiddle_factor((WORD16 *)(&Idft_r18twiddle[0][0]));
    init_idft20_twiddle_factor((WORD16 *)(&Idft_r20twiddle[0][0]));
    init_idft24_twiddle_factor((WORD16 *)(&Idft_r24twiddle[0][0]));
    init_idft25_twiddle_factor((WORD16 *)(&Idft_r25twiddle[0][0]));
    init_idft27_twiddle_factor((WORD16 *)(&Idft_r27twiddle[0][0]));
    init_idft30_twiddle_factor((WORD16 *)(&Idft_r30twiddle[0][0]));
    init_idft32_twiddle_factor((WORD16 *)(&Idft_r32twiddle[0][0]));
    init_idft36_twiddle_factor((WORD16 *)(&Idft_r36twiddle[0][0]));
    init_idft40_twiddle_factor((WORD16 *)(&Idft_r40twiddle[0][0]));
    init_idft48_twiddle_factor((WORD16 *)(&Idft_r48twiddle[0][0]));
    init_idft648_twiddle_factor((WORD16 *)(&Idft_r648twiddle[0][0]));
    init_idft720_twiddle_factor((WORD16 *)(&Idft_r720twiddle[0][0]));
    init_idft768_twiddle_factor((WORD16 *)(&Idft_r768twiddle[0][0]));
    init_idft864_twiddle_factor((WORD16 *)(&Idft_r864twiddle[0][0]));
    init_idft900_twiddle_factor((WORD16 *)(&Idft_r900twiddle[0][0]));
    init_idft960_twiddle_factor((WORD16 *)(&Idft_r960twiddle[0][0]));
    init_idft972_twiddle_factor((WORD16 *)(&Idft_r972twiddle[0][0]));
    init_idft1080_twiddle_factor((WORD16 *)(&Idft_r1080twiddle[0][0]));
    init_idft1152_twiddle_factor((WORD16 *)(&Idft_r1152twiddle[0][0]));
    init_idft1200_twiddle_factor((WORD16 *)(&Idft_r1200twiddle[0][0]));
    init_idft1536_twiddle_factor((WORD16 *)(&Idft_r1536twiddle[0][0]));
    g_sDFTtwiddle.pIdft_r12twiddle = &Idft_r12twiddle[0][0];
    g_sDFTtwiddle.pIdft_r15twiddle = &Idft_r15twiddle[0][0];
    g_sDFTtwiddle.pIdft_r16twiddle = &Idft_r16twiddle[0][0];
    g_sDFTtwiddle.pIdft_r18twiddle = &Idft_r18twiddle[0][0];
    g_sDFTtwiddle.pIdft_r20twiddle = &Idft_r20twiddle[0][0];
    g_sDFTtwiddle.pIdft_r24twiddle = &Idft_r24twiddle[0][0];
    g_sDFTtwiddle.pIdft_r25twiddle = &Idft_r25twiddle[0][0];
    g_sDFTtwiddle.pIdft_r27twiddle = &Idft_r27twiddle[0][0];
    g_sDFTtwiddle.pIdft_r30twiddle = &Idft_r30twiddle[0][0];
    g_sDFTtwiddle.pIdft_r32twiddle = &Idft_r32twiddle[0][0];
    g_sDFTtwiddle.pIdft_r36twiddle = &Idft_r36twiddle[0][0];
    g_sDFTtwiddle.pIdft_r40twiddle = &Idft_r40twiddle[0][0];
    g_sDFTtwiddle.pIdft_r48twiddle = &Idft_r48twiddle[0][0];
    g_sDFTtwiddle.pIdft_r648twiddle = &Idft_r648twiddle[0][0];
    g_sDFTtwiddle.pIdft_r720twiddle = &Idft_r720twiddle[0][0];
    g_sDFTtwiddle.pIdft_r768twiddle = &Idft_r768twiddle[0][0];
    g_sDFTtwiddle.pIdft_r864twiddle = &Idft_r864twiddle[0][0];
    g_sDFTtwiddle.pIdft_r900twiddle = &Idft_r900twiddle[0][0];
    g_sDFTtwiddle.pIdft_r960twiddle = &Idft_r960twiddle[0][0];
    g_sDFTtwiddle.pIdft_r972twiddle = &Idft_r972twiddle[0][0];
    g_sDFTtwiddle.pIdft_r1080twiddle = &Idft_r1080twiddle[0][0];
    g_sDFTtwiddle.pIdft_r1152twiddle = &Idft_r1152twiddle[0][0];
    g_sDFTtwiddle.pIdft_r1200twiddle = &Idft_r1200twiddle[0][0];
    g_sDFTtwiddle.pIdft_r1536twiddle = &Idft_r1536twiddle[0][0];

    //idct dct table
    init_idct_dct_table((short *)aIdctDctTable1200, 1200);
    init_idct_dct_table((short *)aIdctDctTable1152, 1152);
    init_idct_dct_table((short *)aIdctDctTable1080, 1080);
    init_idct_dct_table((short *)aIdctDctTable972, 972);

    init_idct_dct_table((short *)aIdctDctTable960, 960);
    init_idct_dct_table((short *)aIdctDctTable900, 900);
    init_idct_dct_table((short *)aIdctDctTable864, 864);
    init_idct_dct_table((short *)aIdctDctTable768, 768);

    init_idct_dct_table((short *)aIdctDctTable720, 720);
    init_idct_dct_table((short *)aIdctDctTable648, 648);
    init_idct_dct_table((short *)aIdctDctTable600, 600);
    init_idct_dct_table((short *)aIdctDctTable576, 576);
    init_idct_dct_table((short *)aIdctDctTable540, 540);

    init_idct_dct_table((short *)aIdctDctTable480, 480);
    init_idct_dct_table((short *)aIdctDctTable432, 432);
    init_idct_dct_table((short *)aIdctDctTable384, 384);
    init_idct_dct_table((short *)aIdctDctTable360, 360);
    init_idct_dct_table((short *)aIdctDctTable324, 324);

    init_idct_dct_table((short *)aIdctDctTable300, 300);
    init_idct_dct_table((short *)aIdctDctTable288, 288);
    init_idct_dct_table((short *)aIdctDctTable240, 240);
    init_idct_dct_table((short *)aIdctDctTable216, 216);
    init_idct_dct_table((short *)aIdctDctTable192, 192);

    init_idct_dct_table((short *)aIdctDctTable180, 180);
    init_idct_dct_table((short *)aIdctDctTable144, 144);
    init_idct_dct_table((short *)aIdctDctTable120, 120);
    init_idct_dct_table((short *)aIdctDctTable108, 108);
    init_idct_dct_table((short *)aIdctDctTable96, 96);

    init_idct_dct_table((short *)aIdctDctTable72, 72);
    init_idct_dct_table((short *)aIdctDctTable60, 60);
    init_idct_dct_table((short *)aIdctDctTable48, 48);
    init_idct_dct_table((short *)aIdctDctTable36, 36);
    init_idct_dct_table((short *)aIdctDctTable24, 24);


}
