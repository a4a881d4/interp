# -*- coding: cp936 -*-
import numpy as np

def calcCoef( factor, tab_size ):
	if factor<1.:
		point = np.array([ -5., -4., -3, -2, -1., 0., 1., 2. ]);
	else:
		point = np.array([ -1., 0., 1., 2., 3, 4, 5, 6 ]);
	
	y = np.arange(tab_size)/float(tab_size) + 0.5/tab_size

	alf = np.zeros((8,tab_size,16))
	
	for k in range(16):
		for i in range(8):
			alf[i,:,k] = 1.
			for j in range(8):
				if i!=j:
					alf[i,:,k] *= y - point[j]
					alf[i,:,k] /= point[i]-point[j]
		point += 1.
		y += factor
	print y
	return alf
	
tab_size = 32
factor = 15.36/12.5
alf = calcCoef( factor, tab_size )

y = np.arange(tab_size)/float(tab_size) + 0.5/tab_size
 
print "short down_Lagrange_table[%d][8][16] = {"%tab_size
for i in range(tab_size):
	print "/*",np.arange((float(i)+0.5)/tab_size,(float(i)+0.5)/tab_size+16.*factor,factor),"*/"
	for j in range(8):
		for k in range(16):
			print "%8d"%int(round(alf[j,i,k]*32767.)),',',
		print "//",i,j
	print ''
	y += factor
print "};"

factor = 12.5/15.36
alf = calcCoef( factor, tab_size )

y = np.arange(tab_size)/float(tab_size) + 0.5/tab_size
 
print "short up_Lagrange_table[%d][8][16] = {"%tab_size
for i in range(tab_size):
	print "/*",np.arange((float(i)+0.5)/tab_size,(float(i)+0.5)/tab_size+16.*factor,factor),"*/"
	for j in range(8):
		for k in range(16):
			print "%8d"%int(round(alf[j,i,k]*32767.)),',',
		print "//",i,j
	print ''
	y += factor
print "};"
