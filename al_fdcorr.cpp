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
/*
__m128i al_fdcorr_in_fft[8192/4];
*/
__m128i g_al_fdcorr_fd[2048/4];
__m128i g_al_fdcorr_td[2048/4];
__m128i g_al_fdcorr_s[2048/4];

/*
static void al_mul_conj( __m128i *ina, __m128i *inb, __m128i *out, int shift )
{
	int left,right;
	if( shift>=0 ) // freq is low than 0
	{
		right = shift;
	}
	else
	{
		right = 2048/4+shift;
	}
	left = 2048/4 - right;
	mulconj( ina+right, inb, out, left*4 );
	mulconj( ina, inb+left, out+left, right*4 );	
} 
*/
extern "C" void al_fdcorr( void *fft, void *fpn, int shift, int *max, int *avg )
{
	int i;
	int tMax,tAvg;
	__m128i *pfft,*pfd,*ptd,*pfpn,*ps;
	int left,right;
	if( shift>=0 ) // freq is low than 0
	{
		right = shift;
	}
	else
	{
		right = 2048+shift;
	}
	left = 2048 - right;

	pfft = (__m128i*)fft;
	pfpn = (__m128i*)fpn;
	pfd = g_al_fdcorr_fd;
	ptd = g_al_fdcorr_td;
	ps = g_al_fdcorr_s;
	int *ifd, *ifft, *is;
	is = (int *)ps;
	ifft = (int *)fft;
	ifd = (int *)pfd;
	for( i=0;i<4;i++ )
	{
		memcpy( ifd, ifft+right, left*4 );
		memcpy( ifd+left, ifft, right*4 );
		
		mulconj( pfd, pfpn, pfd, 2048 );
		ifft2048( pfd, ptd );
		findCmax2048( ptd, max+i, avg+i );
		
		ifft += 2048;
	}	
}	

extern short stwiddle_2048[];
const static __m128i  IQ_switch = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
const static __m128i  Neg_I = _mm_setr_epi8(0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF);
const static __m128i  Neg_R = _mm_setr_epi8(0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1);

extern "C" int processFreqCorr( void *in, void *out, int nco, int freq, int len )
{
	__m128i *pin, *pout;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	int i,phase,t;
	int *twiddle = (int *)stwiddle_2048;
	
	pin = (__m128i *)in;
	pout = (__m128i *)out;
	
	for( i=0;i<(len+3)/4;i++ )
	{
		phase = (nco>>16)&0x7ff;
		m128_t0 = _mm_load_si128(pin);  pin = pin + 1;
		t = twiddle[phase];
		m128_t2 = _mm_set_epi32( t, t, t, t );
		
		m128_t1 = _mm_shuffle_epi8(m128_t0, IQ_switch);
    m128_t3 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2); 
    m128_t12  = _mm_madd_epi16(m128_t1, m128_t3);
     
    m128_t8  = _mm_srli_si128 (m128_t8, 2); 
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55);
     
    _mm_store_si128(pout, m128_t12);  pout = pout + 1; 
    nco += freq;		
	}	

} 

extern "C" void down2Sample( void *in, void *out, int len, int off )
{
	int i;
	int *iin,*iout;
	iin = (int*)in;
	iout = (int*)out;
	i = len/2;
	iin += off;
	while( i>0 )
	{
		*iout++ = *iin++; iin++; i--;	
	}	
}

extern "C" void xcorr( void *ina, void *inb, int len, int *out )
{
	int i,j;
	__m128i *pa, *pb, *po;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	__m128i sumi, sumq;
	int *ii,*iq;
	pa = (__m128i *)ina;
	pb = (__m128i *)inb;
	sumi = _mm_xor_si128( sumi, sumi );
	sumq = _mm_xor_si128( sumq, sumq );
	for( i=0;i<len/4;i++ )
	{
		m128_t0 = _mm_load_si128(pa);  pa = pa + 1;
		m128_t2 = _mm_load_si128(pb);  pb = pb + 1;
		
		m128_t1 = _mm_shuffle_epi8(m128_t0, IQ_switch);
    m128_t3 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2); 
    m128_t12  = _mm_madd_epi16(m128_t1, m128_t3);
  
  	sumi = _mm_add_epi32(sumi,m128_t8);
  	sumq = _mm_add_epi32(sumq,m128_t12);
  }
  ii = (int *)&sumi;
  iq = (int *)&sumq;
  out[0] = 0;
  out[1] = 0;
  for( i=0;i<4;i++ )
  {
  	out[0] += ii[i];
  	out[1] += iq[i];	
  }
}

extern "C" void xcorr2048Shift( void *ina, void *inb, int shift, int *out )
{
	int i;
	__m128i *pa, *pb;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	__m128i sumi, sumq;
	int *ii,*iq;
	int s0,s1,s2;
	shift &= 0x7ff;
	s0 = shift&3;
	s1 = shift>>2;
	s2 = (2048/4-s1);
	
	pa = (__m128i *)ina;
	pb = (__m128i *)inb;
	pb += (2048/4)*s0;
	
	sumi = _mm_xor_si128( sumi, sumi );
	sumq = _mm_xor_si128( sumq, sumq );
	
	for( i=0;i<s2;i++ )
	{
		m128_t0 = _mm_load_si128(pa+i);  
		m128_t2 = _mm_load_si128(pb+s1+i);
		
		m128_t1 = _mm_shuffle_epi8(m128_t0, IQ_switch);
    m128_t3 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2); 
    m128_t12  = _mm_madd_epi16(m128_t1, m128_t3);
  
  	sumi = _mm_add_epi32(sumi,m128_t8);
  	sumq = _mm_add_epi32(sumq,m128_t12);
  }
  
  for( i=0;i<s1;i++ )
	{
		m128_t0 = _mm_load_si128(pa+s2+i);  
		m128_t2 = _mm_load_si128(pb+i);
		
		m128_t1 = _mm_shuffle_epi8(m128_t0, IQ_switch);
    m128_t3 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2); 
    m128_t12  = _mm_madd_epi16(m128_t1, m128_t3);
  
  	sumi = _mm_add_epi32(sumi,m128_t8);
  	sumq = _mm_add_epi32(sumq,m128_t12);
  }
  
  ii = (int *)&sumi;
  iq = (int *)&sumq;
  out[0] = 0;
  out[1] = 0;
  for( i=0;i<4;i++ )
  {
  	out[0] += ii[i];
  	out[1] += iq[i];	
  }
}   