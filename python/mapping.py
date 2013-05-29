""" For constrained parameters we must map the -inf,inf range of scipy.optimize.leastsq to the desired range.  The functions in this module do that:

forwardmap:	return the starting guess for the mapped variable given the limits
remap:		return the actual variable from the mapped variable
mapfunc:	mapping function """

import scipy.optimize

def remap( a, lim ):
	""" convert mapped variable back to real units, given limits """

	a = a - 0.5
	t = a / ( abs(a)+1.0 )
	return ( lim['max']+lim['min'] )/2.0 + t*( lim['max']-lim['min'] )/2.0

def mapfunc(a,x,lim):
	""" maping function to convert variable from -inf,inf to desired range """

	return x - ( lim['min']+lim['max'] )/2.0 - a/( 1+abs(a) )*( lim['max']-lim['min'] )/2.0

def forwardmap(x,lim):
	""" Note that the offset of 0.5 is for the benefit of the leastsq routine. If I start it perfectly
	centered at zero, then the code gets confused regarding the appropriate step size and sometimes
	starts with 1e-16 and never reaches the actual minima. """

	return scipy.optimize.minpack.fsolve( mapfunc, 0, args=(x,lim) )+0.5