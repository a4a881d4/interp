from ctypes import *	
import time
import numpy as np
#import matplotlib.pyplot as plt

resample_lib = CDLL("./resample.so")

class PSTR_Lagrange_para(Structure):
	_fields_ = [('interp_factor', c_double),
				('fracst', c_double),
				('timeo', c_double)
				]

def Largrane_para_gen( fracst, in_sr, out_sr ):
	
	Largrange_para = PSTR_Lagrange_para()
	Largrange_para.interp_factor = out_sr/in_sr
	Largrange_para.fracst = c_double( fracst )
	Largrange_para.timeo = c_double(0)
	return Largrange_para
	
def large( int_sig_real, int_sig_imag, sig_len1 ):
	test_num = 100
	size_out = int(sig_len1*25./30.72+1.)
	print size_out
	la_real_orig = (c_double*size_out)()
	la_imag_orig = (c_double*size_out)()
	la_out_len = (c_uint*1)()
	fracst_orig = (c_double*1)()
	Largrange_para = Largrane_para_gen( 3., 30.72, 25. )
	larg_start = time.time()
	for i in range(test_num):
		la_result = resample_lib.Large_interp1(
		  int_sig_real
		, int_sig_imag
		, sig_len1
		, byref(Largrange_para)
		, la_real_orig
		, la_imag_orig
		, la_out_len
		, fracst_orig
		, 1
		)
	larg_end = time.time()
	print 'orig_la_time', larg_end - larg_start, fracst_orig[0], la_out_len[0]
	
	la_real = (c_double*size_out)()
	la_imag = (c_double*size_out)()
	fracst = (c_double*1)()
	
	larg_start = time.time()
	for i in range(test_num):
		la_result = resample_lib.Large_interp_faster(
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
	print 'fast_la_time', larg_end - larg_start, fracst[0], la_out_len[0]
	

	 	
	#plt.plot(la_real)
	#plt.plot(la_imag)
	orig_y = [ complex(x,y) for(x,y) in zip(la_real_orig, la_imag_orig) ] 
	y = [ complex(x,y) for(x,y) in zip(la_real, la_imag) ] 
	return la_out_len[0],larg_end - larg_start,y,orig_y

test_len = 302000
x = np.exp(-2.*np.pi*25./30.75/8.*np.arange(test_len)*1j)
r = np.zeros( test_len )
i = np.zeros( test_len )
r[:] = x[:].real
i[:] = x[:].imag

length,es,y,y1 = large( 
	  r.ctypes.data_as(POINTER(c_double))
	, i.ctypes.data_as(POINTER(c_double))
	, test_len )
for i in range(16):
	print 'y[%d]'%i,y[i],y1[i]
for i in range(len(y)-16,len(y)):
	print 'y[%d]'%i,y[i],y1[i]

print np.argmax(abs(np.fft.fft(np.array(y))))
