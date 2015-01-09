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

root = '/astro/data/siesta1/cmancone/sizes_wfc3/fits/pygfit/29'
sims = {'r':	{'xmin': 19, 'xmax': 27, 'zpt': 0, 'zpt_key': 'magzero', 'folder': 'r_NDWFSJ1431p3421', 'label': 'R', 'sky': {'1': 0, '2': 147.6698, '4': 535.8417}},
	'j':	{'xmin': 15, 'xmax': 25, 'zpt': 0, 'zpt_key': 'zptslr', 'folder': 'j_P64', 'label': 'J', 'sky': {'1': 0, '2': 7.8794, '4': 25.8295}},
	'ch1':	{'xmin': 13, 'xmax': 23, 'zpt': 17.97566, 'zpt_key': '', 'folder': 'ch1', 'label': r'$3.6\mu$m', 'sky': {'1': 0, '2': 0.0059, '4': 0.0189}}}
order = ['r','j','ch1']

nsims = len( order )

for (i,filt) in enumerate(order):
	
	config = sims[filt]
	
	cat = pyfits.open( '%s/%s/sims/results.cat' % (root,config['folder']) )[1].data
	m = (cat['mag']-cat['mag_brightest'] > 5) | ( cat['mag'] > 30)
	cat = cat[~m]
	
	ratio = 10.0**( -0.4*(cat['mag']-cat['nearest_mag']) )
	#cat = cat[( cat['nblend'] == 1 ) | ( (cat['nearest'] > 1.0) & (ratio > 0.25) )]
	
	# plot points
	pyplot.subplot( 2, nsims, i+1 )
	pyplot.plot( cat['mag_input'], cat['mag_input']-cat['mag'], 'ko', ms=2 )
	pyplot.axhline( 0.0, color='black' )
	pyplot.axis( [config['xmin'],config['xmax'],-1.99,1.99] )
	if i == 0:
		pyplot.ylabel( 'Mag In - Mag Out' )
		axis = pyplot.axis()
	else:
		pyplot.gca().yaxis.set_major_formatter( ticker.NullFormatter() )
	pyplot.gca().xaxis.set_major_locator( ticker.MultipleLocator( 2 ) )
	pyplot.gca().xaxis.set_major_formatter( ticker.NullFormatter() )
	
	# errors as a function of mag
	hist = data_histogram.data_histogram( cat['mag'], bin=0.5, min=config['xmin'], max=config['xmax'] )
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
	pyplot.annotate( config['label'], (0.05,0.9), (0.05,0.9), xycoords='axes fraction', textcoords='axes fraction' )
	
	# second plot
	pyplot.subplot( 2, nsims, nsims+i+1 )
	m = ns > 3
	pyplot.errorbar( hist.locations[m], errs[m], errs_err[m], fmt='o', color='black' )
	pyplot.errorbar( hist.locations[m], errs_aper[m], errs_err_aper[m], fmt='o', color='black', mfc='white' )
	pyplot.axis( [config['xmin'], config['xmax'], 0, 0.75] )
	if i == 0:
		pyplot.ylabel( '$\sigma$Mag' )
		axis = pyplot.axis()
		#pyplot.gca().yaxis.set_major_formatter( label_hider( axis[3], '%.1f' ) )
	else:
		pyplot.gca().yaxis.set_major_formatter( ticker.NullFormatter() )
	pyplot.gca().xaxis.set_major_locator( ticker.MultipleLocator( 2 ) )
	pyplot.xlabel( 'Mag Out' )
	
	# plot estimated limit due to sky noise
	
	# can we estimate what the sky value is?
	fits = pyfits.open( '%s/%s/%s.fits' % (root,config['folder'],filt) )
	#info = wcs.get_wcs_info( fits[0].header )
	#img = fits[0].data
	#pix = img[~np.isnan(img)]
	#sky = calc.sigrej( pix )*np.pi/info['scale']
	sky = config['sky']['4']
	
	# now figure out zeropoint...
	zpt = config['zpt'] if config['zpt'] else fits[0].header[config['zpt_key']]

	
	mags = np.linspace( config['xmin'], config['xmax'], 1000 )
	fluxes = 10**(-0.4*(mags-zpt))
	errs = -2.5*np.log10( (fluxes-sky)/fluxes )
	pyplot.plot( mags, errs, 'k-' )

pyplot.gcf().set_size_inches( (8,4) )
pyplot.savefig( 'ps/mag_sims.eps' )