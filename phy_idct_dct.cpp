#include <stdio.h>
#include <math.h>

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE 2
#include <pmmintrin.h> // SSE 3
#include <tmmintrin.h> // SSSE 3
#include <smmintrin.h> // SSE 4 for media

#include "common_structure.h"
#include "common_function.h"

extern DFTtwiddleStruct g_sDFTtwiddle;
extern __m128i aIdctDctTable1200[300];
extern __m128i aIdctDctTable1152[288];
extern __m128i aIdctDctTable1080[270];
extern __m128i aIdctDctTable972[243];
extern __m128i aIdctDctTable960[240];
extern __m128i aIdctDctTable900[225];
extern __m128i aIdctDctTable864[216];
extern __m128i aIdctDctTable768[192];
extern __m128i aIdctDctTable720[180];
extern __m128i aIdctDctTable648[162];
extern __m128i aIdctDctTable600[150];
extern __m128i aIdctDctTable576[144];
extern __m128i aIdctDctTable540[135];
extern __m128i aIdctDctTable480[120];
extern __m128i aIdctDctTable432[108];
extern __m128i aIdctDctTable384[96];
extern __m128i aIdctDctTable360[90];
extern __m128i aIdctDctTable324[81];
extern __m128i aIdctDctTable300[75];
extern __m128i aIdctDctTable288[72];
extern __m128i aIdctDctTable240[60];
extern __m128i aIdctDctTable216[54];
extern __m128i aIdctDctTable192[48];
extern __m128i aIdctDctTable180[45];
extern __m128i aIdctDctTable144[36];
extern __m128i aIdctDctTable120[30];
extern __m128i aIdctDctTable108[27];
extern __m128i aIdctDctTable96[24];
extern __m128i aIdctDctTable72[18];
extern __m128i aIdctDctTable60[15];
extern __m128i aIdctDctTable48[12];
extern __m128i aIdctDctTable36[9];
extern __m128i aIdctDctTable24[6];

static __m128i m128_sw_r = _mm_setr_epi8(0, 1, 0, 1, 4, 5, 4, 5,
    8, 9, 8, 9, 12, 13, 12, 13);
static __m128i m128_sw_i = _mm_setr_epi8(2, 3, 2, 3, 6, 7, 6, 7,
    10, 11, 10, 11, 14, 15, 14, 15);
static __m128i m128IQ_switch = _mm_setr_epi8(2, 3, 0, 1, 6, 7, 4, 5,
    10, 11, 8, 9, 14, 15, 12, 13);

static __m128i m128Neg_I = _mm_setr_epi8( 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1,
    0xFF, 0xFF,0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1);

static __m128i vrep = _mm_setr_epi8(2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1);
//static __m128i vsign = _mm_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1);
static __m128i vsign = _mm_setr_epi8(0x01, 0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xff,0x01, 0x00, 0xff, 0xff);
//vsign = _mm_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1);
/* multiply complex: (A + iB)*(C+iD) = AC-BD + i(AD+BC) */
 #define COMPLEX_MULT(input0, input1, outPtr) \
 { \
    __m128i ReRe, ImIm, negImPosRe; \
    __m128i tmp1, tmp2, result; \
    /* select real or image part from a complex value */ \
    ReRe = _mm_shuffle_epi8(input0, m128_sw_r); \
    ImIm = _mm_shuffle_epi8(input0, m128_sw_i); \
    /* swap real or image part and negative image part from a complex value */ \
    /* switch IQ */ \
    tmp1 = _mm_shuffle_epi8(input1, m128IQ_switch);     /* t1,t0,t3,t2,t5,t4,t7,t6 */ \
    /* negative the Q part */ \
    negImPosRe = _mm_sign_epi16(tmp1, m128Neg_I);           /* -t1,t0,-t3,t2,-t5,t4,-t7,t6 */ \
    \
    /* multiply complex */ \
    tmp1 = _mm_mulhrs_epi16(ReRe, input1); \
    tmp2 = _mm_mulhrs_epi16(ImIm, negImPosRe); \
    result = _mm_adds_epi16(tmp1, tmp2); \
    _mm_store_si128((__m128i *) outPtr, result); \
 }

#ifndef PI
#define PI (3.14159265358979323846)
#endif
static WORD16 double2short(DOUBLE64 d)
{
    d = floor(0.5 + d);
    if (d >= 32767)    return 32767;
    if (d < -32768)     return -32768;
    return (WORD16)d;
}

void init_idct_dct_table(WORD16 *psIdctDctTable, WORD16 idctDctSize)
{
    WORD32 i;
    DOUBLE64 factor = DOUBLE64(32768)*0.707106;
    //printf("12-point idct initilization. Developed by Intel Lab China. 2012/04/17.\n");

    for (i=0; i<idctDctSize; ++i)
    {
        /* c+dj */

        *(WORD16 *)(psIdctDctTable + 2*i +0) = double2short( factor * cos(-FLOAT32(i) * PI / (2*idctDctSize)));
        *(WORD16 *)(psIdctDctTable + 2*i +1) = double2short( factor * sin(-FLOAT32(i) * PI / (2*idctDctSize)));
    }
}


WORD32 * DCTIDCT_table_select(WORD32 dct_idct_size)
{
    __m128i * p_table=NULL;
    switch(dct_idct_size)
    {
        case 1200:
        p_table = aIdctDctTable1200;
        break;
        case 1152:
        p_table = aIdctDctTable1152;
        break;
        case 1080:
        p_table = aIdctDctTable1080;
        break;
        case 972:
        p_table = aIdctDctTable972;
        break;
        case 960:
        p_table = aIdctDctTable960;
        break;
        case 900:
        p_table = aIdctDctTable900;
        break;
        case 864:
        p_table = aIdctDctTable864;
        break;
        case 768:
        p_table = aIdctDctTable768;
        break;
        case 720:
        p_table = aIdctDctTable720;
        break;
        case 648:
        p_table = aIdctDctTable648;
        break;
        case 600:
        p_table = aIdctDctTable600;
        break;
        case 576:
        p_table = aIdctDctTable576;
        break;
        case 540:
        p_table = aIdctDctTable540;
        break;
        case 480:
        p_table = aIdctDctTable480;
        break;
        case 432:
        p_table = aIdctDctTable432;
        break;
        case 384:
        p_table = aIdctDctTable384;
        break;
        case 360:
        p_table = aIdctDctTable360;
        break;
        case 320:
        p_table = aIdctDctTable324;
        break;
        case 300:
        p_table = aIdctDctTable300;
        break;
        case 288:
        p_table = aIdctDctTable288;
        break;
        case 240:
        p_table = aIdctDctTable240;
        break;
        case 216:
        p_table = aIdctDctTable216;
        break;
        case 192:
        p_table = aIdctDctTable192;
        break;
        case 180:
        p_table = aIdctDctTable180;
        break;
        case 144:
        p_table = aIdctDctTable144;
        break;
        case 120:
        p_table = aIdctDctTable120;
        break;
        case 108:
        p_table = aIdctDctTable108;
        break;
        case 96:
        p_table = aIdctDctTable96;
        break;
        case 72:
        p_table = aIdctDctTable72;
        break;
        case 60:
        p_table = aIdctDctTable60;
        break;
        case 48:
        p_table = aIdctDctTable48;
        break;
        case 36:
        p_table = aIdctDctTable36;
        break;
        case 24:
        p_table = aIdctDctTable24;
        break;
    }

    return (WORD32 *)p_table;
}
#define IDCT_SHIFT 0
#define DCT_SHIFT (3)

#define IDCT_OUT_RESCALE_VAL (2)
void gen_dct(__m128i *InBuf,__m128i *OutBuf, WORD32 dctSize)
{
    WORD32 iLoop;
    __m128i *pVfftIn=NULL;
    __m128i *pVfftOut=NULL;
    __m128i *pX=NULL;
    __m128i *pIn=NULL;
    __m128i *pOut=NULL;
    __m128i i128Temp0,i128Temp1,i128Temp2,i128Temp3,i128Temp4,i128Temp5;

    WORD32 *pRtable = NULL;

    __m128i xScale;
    WORD16 ScaleDown = 0,nFactor;
    //scale down factor
    if (dctSize <= 48)
    {
        nFactor = 2;
    }
    else if ((dctSize >= 60)&&(dctSize <= 120))
    {
        nFactor = 4;
        if (dctSize == 72)
            nFactor = 2;
    }
    else
    {
        nFactor = 8;
    }
    ScaleDown = (WORD16)((FLOAT32)nFactor / sqrt((FLOAT32)dctSize) * pow(FLOAT32(2), FLOAT32(15)));
    xScale =  _mm_set_epi16(ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown);
    //xScale =  _mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1);


    pX = InBuf;
    pVfftIn = OutBuf;
    pOut = OutBuf;
    pIn = InBuf;
    //1, reoder input
    //v(n)=x(2n)  0=<n<N/2
    for(iLoop=0;iLoop<(dctSize>>1);iLoop=iLoop+1)
    {
        pX = InBuf + 2*iLoop;
        i128Temp0 = _mm_load_si128(pX);
        i128Temp0 = _mm_srai_epi16 (i128Temp0, DCT_SHIFT);
        _mm_store_si128(pVfftIn, i128Temp0);
        pVfftIn = pVfftIn+1;
    }
    //v(n) = x(2N-2n-1)  N/2=<n<N
    for(iLoop=(dctSize>>1);iLoop<dctSize;iLoop=iLoop+1)
    {
        pX = InBuf + (dctSize<<1) - (iLoop<<1) - 1;
        i128Temp0 = _mm_load_si128(pX);
        i128Temp0 = _mm_srai_epi16 (i128Temp0, DCT_SHIFT);
        _mm_store_si128(pVfftIn, i128Temp0);
        pVfftIn = pVfftIn+1;
    }
    //2,FFT
    gen_dft(OutBuf, InBuf, (DFTtwiddleStruct *)&g_sDFTtwiddle, dctSize);
    //3,Get outpt DCT
    pVfftOut = InBuf;
    //pRtable = (WORD32*)pIdctDctTable->pR_1200_table;//need to modify later
    pRtable = DCTIDCT_table_select(dctSize);
    //iAssert(pRtable != NULL);
    i128Temp0 = *pVfftOut;//v(0)
    i128Temp1 =  _mm_mulhrs_epi16 (i128Temp0, xScale); //scale down
    i128Temp0 = _mm_slli_epi16 (i128Temp1, 0);
    _mm_store_si128(pOut, i128Temp0);

    //i128Temp1 =  _mm_mulhrs_epi16 (i128Temp0, xScale); //scale down
    //_mm_store_si128(pOut, i128Temp1);
    pOut++;
    for(iLoop=1; iLoop<dctSize; iLoop++)
    {
        i128Temp0 = _mm_load_si128(pVfftOut + iLoop);//v(k)
        i128Temp1 = _mm_load_si128(pVfftOut + dctSize - iLoop);//v(N-k)
        i128Temp2 = _mm_set1_epi32 ((WORD32) *(pRtable+iLoop));//R(k)
        i128Temp3 = _mm_sign_epi16 (i128Temp2, vsign);//conj(R(k))
        COMPLEX_MULT(i128Temp0,i128Temp2, &i128Temp4);//v(k)*R(k)
        COMPLEX_MULT(i128Temp1,i128Temp3, &i128Temp5);//v(N-k)*conj(R(k))
        i128Temp0 = _mm_adds_epi16(i128Temp4,i128Temp5);//v(k)*R(k)+v(N-k)*conj(R(k))

        i128Temp1 =  _mm_mulhrs_epi16 (i128Temp0, xScale); //scale down
        i128Temp0 = _mm_slli_epi16 (i128Temp1, 0);
        _mm_store_si128(pOut, i128Temp0);

        //i128Temp1 =  _mm_mulhrs_epi16 (i128Temp0, xScale); //scale down
        //_mm_store_si128(pOut, i128Temp1);
        pOut++;
    }
}

void gen_idct(__m128i *InBuf,__m128i *OutBuf,WORD32 dctSize)
{
    WORD32 iLoop;
    WORD32 *pRtable = NULL;

    __m128i *pFFTin = NULL;
    __m128i *pFFTout = NULL;
    __m128i *pIDCTout = NULL;

    __m128i i128Temp0,i128Temp1,i128Temp2,i128Temp3;

    __m128i xScale;
    WORD16 ScaleDown = 0,nFactor;
    //scale down factor
    if (dctSize <= 48)
    {
        nFactor = 2;
    }
    else if ((dctSize >= 60)&&(dctSize <= 120))
    {
        nFactor = 4;
    if (dctSize == 72)
        nFactor = 2;
    }
    else
    {
        nFactor = 8;
    }
    nFactor = nFactor *IDCT_OUT_RESCALE_VAL;
    ScaleDown = (WORD16)((FLOAT32)nFactor / sqrt((FLOAT32)dctSize) * pow(FLOAT32(2), FLOAT32(15)));
    xScale =  _mm_set_epi16(ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown, ScaleDown);

    //1, preprocessing for inBuf
    pFFTin = OutBuf;
    //pRtable = (WORD32*)pIdctDctTable->pR_1200_table;//need to modify later
    pRtable = DCTIDCT_table_select(dctSize);
    //iAssert(pRtable != NULL);
    i128Temp0 = _mm_load_si128(InBuf);//w(0)
    i128Temp2 = _mm_srai_epi16 (i128Temp0, IDCT_SHIFT);
    _mm_store_si128(pFFTin,i128Temp2);

    for(iLoop=1;iLoop<dctSize;iLoop++)
    {
        i128Temp0 = _mm_load_si128(InBuf + iLoop); //y(i)
        i128Temp1 = _mm_load_si128(InBuf + dctSize-iLoop); //y(N-i)
        i128Temp3 = _mm_shuffle_epi8(i128Temp1, m128IQ_switch);//
        i128Temp1 = _mm_sign_epi16(i128Temp3, vsign);//conj(y(N-i))
        i128Temp3 = _mm_adds_epi16(i128Temp0,i128Temp1);//

        i128Temp1 = _mm_set1_epi32((WORD32)*(pRtable + iLoop));   //R(i) QI QI QI QI
        i128Temp0 = _mm_sign_epi16(i128Temp1, vsign);//conj(y(N-i))

        COMPLEX_MULT(i128Temp0,i128Temp3, &i128Temp2);//w(N-k)*R(N-k)
        i128Temp2 = _mm_srai_epi16 (i128Temp2, IDCT_SHIFT);
        _mm_store_si128(pFFTin+iLoop,i128Temp2);
    }

    //2,IFFT
    gen_idft(OutBuf, InBuf, (DFTtwiddleStruct *)&g_sDFTtwiddle,dctSize);

    //3, re-ordering
    pFFTout = InBuf;
    pIDCTout = OutBuf;
    for(iLoop=0; iLoop<dctSize; iLoop=iLoop+2)
    {
        i128Temp0 = _mm_load_si128(pFFTout + (iLoop>>1));//y(k/2)
        i128Temp1 =  _mm_mulhrs_epi16 (i128Temp0, xScale); //scale down
        _mm_store_si128(pIDCTout+iLoop,i128Temp1);
    }
    for(iLoop=1; iLoop<dctSize; iLoop=iLoop+2)
    {
        i128Temp0 = _mm_load_si128(pFFTout + dctSize - ((iLoop+1)>>1));//y(N-(k+1)/2)
        i128Temp1 =  _mm_mulhrs_epi16 (i128Temp0, xScale); //scale down
        _mm_store_si128(pIDCTout+iLoop,i128Temp1);
    }
}
