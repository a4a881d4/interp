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

#define BUFSIZE (512*256)
#define TABSIZELOG2 (5)
typedef struct ft_corr {
	int amp;
	int carrierf;
	int timingf;
	int carrierp;
	int timingp;
	int buf_end;
} ft_corr_t;
__m128i g_interpolationBuf[BUFSIZE/4];

extern "C" int interpolationPutBuf( void *in, int il, ft_corr_t *state )
{
	int pos;
	int *pb;
	pos = state->timingp >> 16;
	pb = (int*)g_interpolationBuf;
	memmove( pb, pb+pos, (state->buf_end-pos)*sizeof(int) );
	state->buf_end = state->buf_end-pos; 
	state->timingp &= 0xffff;
	memcpy( pb+state->buf_end, in, il*sizeof(int) );
	state->buf_end += il;
	return state->buf_end;
}

static void dump( __m128i a, const char name[] )
{
	short *pa;
	int i;
	pa = (short *)&a;
	printf("%s:",name);
	for( i=0;i<8;i++ )
		printf("%6d ",(int)pa[i]);
	printf("\n");
}

static void dumpint( __m128i a, const char name[] )
{
	int *pa;
	int i;
	pa = (int *)&a;
	printf("%s:",name);
	for( i=0;i<4;i++ )
		printf("%10d ",(int)pa[i]);
	printf("\n");
}
static void dumpTab( void *tab )
{
	__m128i t0;
	__m128i *pt = (__m128i *)tab;
	int i,j;
	for( i=0;i<32;i++ )
	{
		for(j=0;j<6;j++ )
		{
			t0 = _mm_load_si128( pt ); pt+=1;
			dump( t0, "" );
		}
		printf("\n");
	} 	
}
extern "C" int interpolationFreqCorr( void *out, void *tab, int ol, int oo, ft_corr_t *state )
// ol must div 4
// oo must div 4
// interpolation 6
{
	int pos;
	int i,j;
	int *pb;
	int cp,tp,t;
	__m128i *po,*pt;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	__m128i sumi, sumq, m128_amp;
	int *twiddle = (int *)stwiddle_2048;
	pt = (__m128i*)tab;
	po = (__m128i*)out;
	po += oo/4;
	pos = state->timingp >> 16;
	pb = (int*)g_interpolationBuf;
	//dumpTab( tab );
	short amp = (short)state->amp;
	m128_amp = _mm_set_epi16( amp, amp, amp, amp, amp, amp, amp, amp );
	for( i=0;i<ol/4;i++ )
	{
		cp = (state->carrierp>>16)&0x7ff;
		tp = (state->timingp>>(16-TABSIZELOG2))&((1<<TABSIZELOG2)-1);
		pos = state->timingp>>16;
		sumi = _mm_xor_si128( sumi, sumi );
		sumq = _mm_xor_si128( sumq, sumq );
		{
			m128_t0 = _mm_loadu_si128( (__m128i*)(pb+pos+0) );  // if(i==0) dump( m128_t0, "m128_t0");
			m128_t1 = _mm_loadu_si128( (__m128i*)(pb+pos+1) );  // if(i==0) dump( m128_t1, "m128_t1");
			                                                    //
			m128_t2 = _mm_load_si128( pt+tp*6 );                // if(i==0) dump( m128_t2, "m128_t2");
			m128_t3 = _mm_load_si128( pt+tp*6+1 );              // if(i==0) dump( m128_t3, "m128_t3");
			                                                    //
			m128_t0 = _mm_shuffle_epi8(m128_t0, IQ_switch);     // if(i==0) dump( m128_t0, "m128_t0");
    	m128_t4 = _mm_blend_epi16( m128_t1, m128_t0, 0x55 );// if(i==0) dump( m128_t4, "image m128_t4");// image
			m128_t5 = _mm_blend_epi16( m128_t0, m128_t1, 0x55 );// // real
			m128_t5 = _mm_shuffle_epi8(m128_t5, IQ_switch);     // if(i==0) dump( m128_t5, "real m128_t5");
    	                                                    //
			m128_t6 = _mm_blend_epi16( m128_t3, m128_t2, 0x55 );// if(i==0) dump( m128_t6, "m128_t6");
			                                                    //
			m128_t7 = _mm_madd_epi16(  m128_t4, m128_t6 );      // if(i==0) dumpint( m128_t7, "m128_t7");
			sumq = _mm_add_epi32( sumq, m128_t7 );              //
			                                                    //
			m128_t8 = _mm_madd_epi16(  m128_t5, m128_t6 );      // if(i==0) dumpint( m128_t8, "m128_t8");
			sumi = _mm_add_epi32( sumi, m128_t8 );              //
			                                                    //
			m128_t0 = _mm_loadu_si128( (__m128i*)(pb+pos+2) );  // if(i==0) dump( m128_t0, "m128_t0");
			m128_t1 = _mm_loadu_si128( (__m128i*)(pb+pos+3) );  // if(i==0) dump( m128_t1, "m128_t1");
			                                                    //                                    
			m128_t2 = _mm_load_si128( pt+tp*6+2 );              // if(i==0) dump( m128_t2, "m128_t2");
			m128_t3 = _mm_load_si128( pt+tp*6+3 );              // if(i==0) dump( m128_t3, "m128_t3");
			                                                    //
			m128_t0 = _mm_shuffle_epi8(m128_t0, IQ_switch);     // if(i==0) dump( m128_t0, "m128_t0");
    	m128_t4 = _mm_blend_epi16( m128_t1, m128_t0, 0x55 );// if(i==0) dump( m128_t4, "image m128_t4");// image
			m128_t5 = _mm_blend_epi16( m128_t0, m128_t1, 0x55 );// // real
			m128_t5 = _mm_shuffle_epi8(m128_t5, IQ_switch);     // if(i==0) dump( m128_t5, "real m128_t5");
    	                                                    //
			m128_t6 = _mm_blend_epi16( m128_t3, m128_t2, 0x55 );// if(i==0) dump( m128_t6, "m128_t6");
			                                                    //
			m128_t7 = _mm_madd_epi16(  m128_t4, m128_t6 );      //
			sumq = _mm_add_epi32( sumq, m128_t7 );              //
			                                                    //
			m128_t8 = _mm_madd_epi16(  m128_t5, m128_t6 );      //
			sumi = _mm_add_epi32( sumi, m128_t8 );              //
                                                          //
			m128_t0 = _mm_loadu_si128( (__m128i*)(pb+pos+4) );  // if(i==0) dump( m128_t0, "m128_t0");
			m128_t1 = _mm_loadu_si128( (__m128i*)(pb+pos+5) );  // if(i==0) dump( m128_t1, "m128_t1");
			                                                    //                                    
			m128_t2 = _mm_load_si128( pt+tp*6+4 );              // if(i==0) dump( m128_t2, "m128_t2");
			m128_t3 = _mm_load_si128( pt+tp*6+5 );              // if(i==0) dump( m128_t3, "m128_t3");
			                                                    //
			m128_t0 = _mm_shuffle_epi8(m128_t0, IQ_switch);     // if(i==0) dump( m128_t0, "m128_t0");
    	m128_t4 = _mm_blend_epi16( m128_t1, m128_t0, 0x55 );// if(i==0) dump( m128_t4, "image m128_t4");// image
			m128_t5 = _mm_blend_epi16( m128_t0, m128_t1, 0x55 );// // real
			m128_t5 = _mm_shuffle_epi8(m128_t5, IQ_switch);     // if(i==0) dump( m128_t5, "real m128_t5");
    	                                                    //
			m128_t6 = _mm_blend_epi16( m128_t3, m128_t2, 0x55 );// if(i==0) dump( m128_t6, "m128_t6");
			
			m128_t7 = _mm_madd_epi16(  m128_t4, m128_t6 );
			sumq = _mm_add_epi32( sumq, m128_t7 );
			
			m128_t8 = _mm_madd_epi16(  m128_t5, m128_t6 );
			sumi = _mm_add_epi32( sumi, m128_t8 );

			sumi  = _mm_srli_si128 (sumi, 2); 
    	m128_t0 = _mm_blend_epi16(sumq,sumi, 0x55);            //if(i<16) dump( m128_t0, "m128_t0");
	    m128_t0 = _mm_mulhi_epi16( m128_t0, m128_amp );
		}
		t = twiddle[cp];
		m128_t2 = _mm_set_epi32( t, t, t, t );
		
		m128_t1 = _mm_shuffle_epi8(m128_t0, IQ_switch);
    m128_t3 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2); 
    m128_t12  = _mm_madd_epi16(m128_t1, m128_t3);
     
    m128_t8  = _mm_srli_si128 (m128_t8, 2); 
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55);
     
    _mm_store_si128(po, m128_t12);  po = po + 1; 
    state->carrierp += state->carrierf;	
    state->timingp += state->timingf;	
	}
	pos = state->timingp >> 16;
	return state->buf_end-pos;
}
	
extern "C" void mulConst( void *in, void *pn, void *out, int len, int *coef )
{
	int i;
	__m128i *pin, *ppn, *pout;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	__m128i coef_r, coef_i;
	
	pin = (__m128i *)in;
	ppn = (__m128i *)pn;
	pout = (__m128i *)out;
	
	coef_r = _mm_set_epi32( coef[0], coef[0], coef[0], coef[0] );
	coef_i = _mm_set_epi32( coef[1], coef[1], coef[1], coef[1] );
	
	for( i=0;i<len/4;i++ )
	{
		m128_t0 = _mm_load_si128(pin); pin = pin + 1; 
		m128_t2 = _mm_load_si128(ppn); ppn = ppn + 1; ;
		
		m128_t1 = _mm_shuffle_epi8(m128_t0, IQ_switch);
    m128_t3 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2);  //real 
    m128_t12  = _mm_madd_epi16(m128_t1, m128_t3);  //imag
  
    m128_t4 = _mm_mullo_epi32( m128_t8, coef_r );
    m128_t5 = _mm_mullo_epi32( m128_t12, coef_i );
    
    m128_t4 = _mm_sub_epi32( m128_t4, m128_t5 ); // real
    
    m128_t6 = _mm_mullo_epi32( m128_t8, coef_i );
    m128_t7 = _mm_mullo_epi32( m128_t12, coef_r );
    
    m128_t12 = _mm_add_epi32( m128_t6, m128_t7 ); //imag
    
    m128_t8  = _mm_srli_si128 (m128_t4, 2);
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55);

		_mm_store_si128(pout, m128_t12);  pout = pout + 1;
  }
}

/*
def FHT(data):
	lN = int(np.log2(len(data))+0.5)
	N = 2**lN
	print 'FHT:',lN,N
	print '-------------------------------------------'
	x = np.zeros(N)
	x = data[:N]
	k1=N
	k2=1
	k3=N/2
	for i1 in range(lN):
		L1 = 0
		for i2 in range(k2):
			for i3 in range(k3):
				i = i3+L1
				j = i+k3
				temp1 = x[i]
				temp2 = x[j]
				x[i] = temp1 + temp2
				x[j] = temp1 - temp2
			L1 += k1
		k1 /= 2
		k2 *= 2
		k3 /= 2
	return x

*/

extern "C" void walsh( short *in, int level, int off )
{
	short *pin = in + off*2;
	int k1,k2,k3;
	int i1,i2,i3;
	int L1,i,j;
	int ta,tb;
	k1 = 2<<level;
	k2 = 128>>level;
	k3 = 1<<level;
	//for( i1=0;i1<level;i1++ )
	{
		L1 = 0;
		for( i2=0;i2<k2;i2++ )
		{
			for( i3=0;i3<k3;i3++ )
			{
				i = i3+L1;
				j = i +k3;
				ta = pin[2*i];
				tb = pin[2*j];
				pin[2*i] = (ta+tb)/2;
				pin[2*j] = (ta-tb)/2;
				ta = pin[2*i+1];
				tb = pin[2*j+1];
				pin[2*i+1] = (ta+tb)/2;
				pin[2*j+1] = (ta-tb)/2;
			}
			L1 += k1;
		}
		//k1 *= 2;
		//k2 /= 2;
		//k3 *= 2;
	}	
}


extern "C" void hadamard( short *in, int level, int off )
{
	short *pin = in + off*2;
	int k1,k2,k3;
	int i1,i2,i3;
	int L1,i,j;
	int ta,tb;
	k1 = 256;
	k2 = 1;
	k3 = 128;
	for( i1=0;i1<level;i1++ )
	{
		L1 = 0;
		for( i2=0;i2<k2;i2++ )
		{
			for( i3=0;i3<k3;i3++ )
			{
				i = i3+L1;
				j = i +k3;
				ta = pin[2*i];
				tb = pin[2*j];
				pin[2*i] = (ta+tb)/2;
				pin[2*j] = (ta-tb)/2;
				ta = pin[2*i+1];
				tb = pin[2*j+1];
				pin[2*i+1] = (ta+tb)/2;
				pin[2*j+1] = (ta-tb)/2;
			}
			L1 += k1;
		}
		k1 /= 2;
		k2 *= 2;
		k3 /= 2;
	}	
}
