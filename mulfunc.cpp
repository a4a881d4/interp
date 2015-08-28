#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE 2 
#include <pmmintrin.h> // SSE 3
#include <tmmintrin.h> // SSSE 3
#include <smmintrin.h> // SSE 4 for media

const static __m128i  IQ_switch = _mm_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
const static __m128i  Neg_I = _mm_setr_epi8(0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF);
const static __m128i  Neg_R = _mm_setr_epi8(0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1, 0xFF, 0xFF, 0x1, 0x1);

extern "C" void mul( void *ina, void *inb, void *out, int len )
{
	int i,j;
	__m128i *pa, *pb, *po;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	
	pa = (__m128i *)ina;
	pb = (__m128i *)inb;
	po = (__m128i *)out;
	for( i=0;i<len/4;i++ )
	{
		m128_t0 = _mm_load_si128(pa);  pa = pa + 1;
		m128_t2 = _mm_load_si128(pb);  pb = pb + 1;
		
		m128_t3 = _mm_shuffle_epi8(m128_t2, IQ_switch);
    m128_t2 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2); 
    m128_t12  = _mm_madd_epi16(m128_t0, m128_t3);
    
    m128_t8  = _mm_srli_si128 (m128_t8, 2); 
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55);
    
    _mm_store_si128(po, m128_t12);  po = po + 1; 
	}
}

extern "C" void mulconj( void *ina, void *inb, void *out, int len )
{
	int i,j;
	__m128i *pa, *pb, *po;
	__m128i m128_t0,m128_t1,m128_t2,m128_t3,m128_t4,m128_t5,m128_t6,m128_t7,m128_t8,m128_t9,m128_t10,m128_t11,m128_t12;
	
	pa = (__m128i *)ina;
	pb = (__m128i *)inb;
	po = (__m128i *)out;
	for( i=0;i<len/4;i++ )
	{
		m128_t0 = _mm_load_si128(pa);  pa = pa + 1;
		m128_t2 = _mm_load_si128(pb);  pb = pb + 1;
		
		m128_t1 = _mm_shuffle_epi8(m128_t0, IQ_switch);
    m128_t3 = _mm_sign_epi16  (m128_t2, Neg_I);

    m128_t8 = _mm_madd_epi16  (m128_t0, m128_t2); 
    m128_t12  = _mm_madd_epi16(m128_t1, m128_t3);
     
    m128_t8  = _mm_srli_si128 (m128_t8, 2); 
    m128_t12 = _mm_blend_epi16(m128_t12,m128_t8, 0x55);
     
    _mm_store_si128(po, m128_t12);  po = po + 1; 
	}
}

extern "C" void _mm_cabs( void *inI, void *inQ, void *out, int Length )
{
	__m128i zero,one,two,iabs,qabs,id,qd,od,min,max,*ibuf,*qbuf,*obuf;
	
	int i,j;
	one=_mm_set_epi16(1,1,1,1,1,1,1,1);
	two=_mm_set_epi16(2,2,2,2,2,2,2,2);
	ibuf=(__m128i *)inI;
	qbuf=(__m128i *)inQ;
	obuf=(__m128i *)out;
	for( i=0;i<Length;i++ )
	{
		zero=_mm_xor_si128(zero,zero);
		id=_mm_load_si128(ibuf);
		zero=_mm_cmpgt_epi16(zero,id);
		zero=_mm_mullo_epi16(zero,two);
		zero=_mm_add_epi16(one,zero);
		iabs=_mm_mullo_epi16(zero,id);

		zero=_mm_xor_si128(zero,zero);

		qd=_mm_load_si128(qbuf);

		zero=_mm_cmpgt_epi16(zero,qd);

		zero=_mm_mullo_epi16(zero,two);

		zero=_mm_add_epi16(one,zero);

		qabs=_mm_mullo_epi16(zero,qd);

		od=_mm_max_epi16(iabs,qabs);
		min=_mm_min_epi16(iabs,qabs);

		od=_mm_add_epi16(od,od);
		od=_mm_add_epi16(od,min);
		_mm_store_si128(obuf,od);

		ibuf++;
		qbuf++;
		obuf++;
	}
}

extern "C" void _mm_power( void *in, void *out, int Length, int *iMax, int *avg )
{
	__m128i *ibuf,*obuf;
	__m128i id,pow,max,sum;
	int *sp;
	int *ip;
	short *sip;
	int i,j;
	printf("%s(a4a881d4): sizeof int = %d\n",__FILE__,sizeof(int));
	ibuf=(__m128i *)in;
	obuf=(__m128i *)out;
	
	max = _mm_xor_si128(max,max);
	sum = _mm_xor_si128(sum,sum);
	
	for( i=0;i<Length/4;i++ )
	{
		id=_mm_load_si128(ibuf);
		pow = _mm_madd_epi16( id, id );
		max = _mm_max_epi32( max, pow );
		_mm_store_si128(obuf,pow);
	
	sp = (int *)&sum;
	
	for( j=0;j<4;j++ )
	{
		sip = (short*)ibuf;
		printf("sp[%d] = %d, %d, %d\n",j,sp[j],(int)sip[j*2],(int)sip[j*2+1]);
	}

		pow = _mm_srai_epi32(pow, 8); 
		sum = _mm_add_epi32(sum,pow);

		ibuf++;
		obuf++;
	}
	*iMax=0;
	*avg=0;
	sp = (int *)&max;
	for( i=0;i<4;i++ )
	{
		if( sp[i]>*iMax )
			*iMax=sp[i];
	}
	ip = (int *)&sum;
	for( i=0;i<4;i++ )
	{
		*avg+=(int)ip[i];
	}
}

extern "C" void _mm_findMax( void *inI, int Length, int *imax, int *avg )
{
	__m128i id,max,sum,*ibuf;
	__m128i ones;

	int i;
	short *sp;
	int *ip;
	ibuf=(__m128i *)inI;
	max=_mm_xor_si128(max,max);
	sum=_mm_xor_si128(sum,sum);
	ones=_mm_set_epi16(1,1,1,1,1,1,1,1);

	for( i=0;i<Length;i++ )
	{
		id=_mm_load_si128(ibuf);
		max=_mm_max_epi16(max,id);
		id=_mm_madd_epi16(id,ones);
		sum=_mm_add_epi32(sum,id);
		ibuf++;
	}
	sp=(short *)&max;
	*imax=0;
	*avg=0;
	for( i=0;i<8;i++ )
	{
		if( sp[i]>*imax )
			*imax=sp[i];
	}
	ip=(int *)&sum;
	for( i=0;i<4;i++ )
	{
		*avg+=(int)ip[i];
		
	}
	*avg/=8;
	*avg/=Length;
}

extern "C" void findCmax2048( void *in, int *iMax, int *avg )
{
	__m128i *ibuf,*obuf;
	__m128i id,pow,max,sum, index, mask, ones, spow;
	int *sp;
	int *ip;
	short *sip;
	int i,j;
	ones = _mm_set_epi32(4,4,4,4);
	mask = _mm_set_epi32(0xfffff800,0xfffff800,0xfffff800,0xfffff800);
	index = _mm_set_epi32(3,2,1,0);
	
	ibuf=(__m128i *)in;
	
	max = _mm_xor_si128(max,max);
	sum = _mm_xor_si128(sum,sum);
	
	for( i=0;i<2048/4;i++ )
	{
		id=_mm_load_si128(ibuf);
		pow = _mm_madd_epi16( id, id );
		spow = _mm_srai_epi32(pow, 8); 
		
		pow = _mm_and_si128( pow, mask );
		pow = _mm_or_si128( pow, index );
		max = _mm_max_epi32( max, pow );
		index = _mm_add_epi32( index, ones );
	
		sum = _mm_add_epi32(sum,spow);

		ibuf++;
		obuf++;
	}
	*iMax=0;
	*avg=0;
	sp = (int *)&max;
	for( i=0;i<4;i++ )
	{
		if( sp[i]>*iMax )
			*iMax=sp[i];
	}
	ip = (int *)&sum;
	for( i=0;i<4;i++ )
	{
		*avg+=(int)ip[i];
	}
}