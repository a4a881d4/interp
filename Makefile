CFLAGS = -O2 -msse4 -march=core2  -fPIC
OBJS = fft1024.o \
	ifft1024.o \
	init_fft.o \
	phy_dft.o \
	phy_fft1024_core_dw.o \
	phy_fft2048.o \
	phy_fft.o \
	phy_idct_dct.o \
	phy_idft.o \
	phy_ifft.o \
	mfft.o \
	twiddle8192.o \
	mulfunc.o \
	al_fdcorr.o

%.o : %.cpp
	gcc $(CFLAGS) -c -o $@ $<

fftlib.so : $(OBJS)
	gcc -shared -o fftlib.so $(OBJS)

all : fftlib.so

clean:
	rm *.o -f
	rm *.so -f
	rm *~


