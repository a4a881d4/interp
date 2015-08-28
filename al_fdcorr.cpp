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
 