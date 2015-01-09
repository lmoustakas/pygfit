#!/usr/bin/python

import pyfits
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import data_histogram
import numpy as np
import calc
import wcs

# run 100 bootstraps for each bin to calculate error of standard deviation
def calc_std_err( vals ):
	n = vals.size
	if n == 1: return 0
	return np.std( [ np.std( vals[np.random.randint( 0, n-1, n )] ) for c in range( 100 ) ] )

pyplot.rcParams.update( {'legend.fontsize': 'medium', 'legend.numpoints': 1, 'figure.subplot.wspace': 0, 'figure.subplot.hspace': 0} )

zpt = 17.97566
sky = 0.0189
xmin = 13
xmax = 23
root = '/astro/data/siesta1/cmancone/sizes_wfc3/fits/pygfit/29'

cat = pyfits.open( '%s/ch1/sims/results.cat' % (root) )[1].data
m = (cat['mag']-cat['mag_brightest'] > 5) | ( cat['mag'] > 30)
cat = cat[~m]

cat = cat[(cat['re_arcsecs'] < 2.0) & (cat['n'] < 7.5) & (cat['mag_warning'] == 0)]
before = cat.size
ratio = 10.0**( -0.4*(cat['mag']-cat['nearest_mag']) )
cat = cat[( cat['nblend'] == 1 ) | ( (cat['nearest'] > 1.0) & (ratio > 0.25) )]
after = cat.size
print float( before-after )/before

# plot points
pyplot.subplot( 2,1,1 )
pyplot.plot( cat['mag_input'], cat['mag_input']-cat['mag'], 'ko', ms=2 )
pyplot.axhline( 0.0, color='black' )
pyplot.axis( [xmin,xmax,-1.999,1.999] )
pyplot.ylabel( 'Mag In - Mag Out' )

# errors as a function of mag
hist = data_histogram.data_histogram( cat['mag_input'], bin=0.5, min=xmin, max=xmax )
ns = hist.hist
errs = np.zeros( hist.nbins )
errs_err = np.zeros( hist.nbins )
errs_aper = np.zeros( hist.nbins )
errs_err_aper = np.zeros( hist.nbins )
for (j,inds) in enumerate(hist):
	if not len( inds ): continue
	errs[j] = np.std( (cat['mag_input']-cat['mag'])[inds] )
	errs_err[j] = calc_std_err( (cat['mag_input']-cat['mag'])[inds] )
	errs_aper[j] = np.std( (cat['mag_input']-cat['mag_aper'])[inds] )
	errs_err_aper[j] = calc_std_err( (cat['mag_input']-cat['mag_aper'])[inds] )

# second plot
pyplot.subplot( 2,1,2 )
m = ns > 3
pyplot.errorbar( hist.locations[m], errs[m], errs_err[m], fmt='o', color='black' )
pyplot.errorbar( hist.locations[m], errs_aper[m], errs_err_aper[m], fmt='o', color='black', mfc='white' )
pyplot.axis( [xmin, xmax, 0, 0.75] )
pyplot.ylabel( '$\sigma$Mag' )
axis = pyplot.axis()
pyplot.gca().xaxis.set_major_locator( ticker.MultipleLocator( 2 ) )
pyplot.xlabel( 'Mag In' )

# plot estimated limit due to sky noise
mags = np.linspace( xmin, xmax, 1000 )
fluxes = 10**(-0.4*(mags-zpt))
errs = -2.5*np.log10( (fluxes-sky)/fluxes )
#pyplot.plot( mags, errs, 'k-' )

pyplot.gcf().set_size_inches( (6,4) )
pyplot.savefig( 'ps/irac_sims_cut.eps' )