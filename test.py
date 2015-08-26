from ctypes import *	
import time
import numpy as np

resample_lib = CDLL("resample.so")

class PSTR_Lagrange_para(Structure):
	_fields_ = [('interp_factor', c_double),
				('fracst', c_double),
				('timeo', c_double)
				]

def Largrane_para_gen( fracst, in_sr, out_sr ):
	
	Largrange_para = PSTR_Lagrange_para()
	Largrange_para.interp_factor = in_sr/out_sr
	Largrange_para.fracst = c_double( fracst )
	Largrange_para.timeo = c_double(0)
	return Largrange_para
	
def large( int_sig_real, int_sig_imag, sig_len1 ):
	
	larg_start = time.time()
	
	la_real = (c_double*len(deci_real[:]))()
	la_imag = (c_double*len(deci_real[:]))()
	la_out_len = (c_uint*1)()
	fracst = (c_double*1)()
	Largrange_para = Largrane_para_gen( 0., 30.72, 25. )
	la_result = resample_lib.Large_interp1(
		  int_sig_real
		, int_sig_imag
		, sig_len1
		, byref(Largrange_para)
		, la_real
		, la_imag
		, la_out_len
		, fracst
		, 1
		)
	

	
	larg_end = time.time()
	print 'la_time', larg_end - larg_start

	return la_out_len[0],larg_end - larg_start

x = np.exp(-2.*np.pi*25./30.75/8.*np.arange(1200))
r = x.real
i = x.imag

length,es = large( 
	  r.ctypes.data_as(POINTER(c_double))
	, i.ctypes.data_as(POINTER(c_double))
	, 1200 )
	 