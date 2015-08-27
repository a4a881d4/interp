from ctypes import *
import numpy as np

def m128iBuf( l ):
	alignment = 16
	buf_size = l*4 + alignment - 1
	raw_memory = bytearray(buf_size)
	ctypes_raw_type = (c_char * buf_size)
	ctypes_raw_memory = ctypes_raw_type.from_buffer(raw_memory)
	raw_address = addressof(ctypes_raw_memory)
	offset = raw_address % alignment
	offset_to_aligned = (alignment - offset) % alignment
	ctypes_aligned_type = (c_short * (l*2))
	ctypes_aligned_memory = ctypes_aligned_type.from_buffer(raw_memory, offset_to_aligned)
	return ctypes_aligned_memory

fftlib = CDLL('./fftlib.so')

print fftlib.__dict__

fftlib.InitFFTLIBTables()

k = 1.
c = np.arange(2048)
#d = np.exp(1j*k*c/2048.*2.*np.pi) * 64.
d = np.zeros(2048,dtype='complex')
d.real = np.arange(0,4096,2)
d.imag = np.arange(1,4096,2)

a = m128iBuf( 2048 )
b = m128iBuf( 2048 )
print hex(addressof(a)),hex(addressof(b))
for i in range(2048):
	a[2*i] = c_short(int(d[i].real))
	a[2*i+1] = c_short(int(d[i].imag))

fftlib.fft2048_n(byref(a),byref(b))

e =np.fft.fft(d)

print np.argmax(abs(e))
for i in range(128):
	print b[2*i],b[2*i+1],e[i]/8.

