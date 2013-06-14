#!/usr/bin/python

import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import pyfits

class label_hider(ticker.Formatter):
	def __init__( self, to_hide, format ):
		self.hide = to_hide
		self.format = format
	def __call__( self, a, pos=None ):
		return '' if a == self.hide else self.format % a

pyplot.rcParams.update( {'figure.subplot.wspace': 0} )

root = '/astro/data/siesta1/cmancone/sizes_wfc3/fits/pygfit/29'
sims = {'r':	{'mag_lim': 24.0, 'folder': 'r_NDWFSJ1431p3421', 'label': 'R'},
	'j':	{'mag_lim': 21.5, 'folder': 'j_P64', 'label': 'J'},
	'ch1':	{'mag_lim': 19.0, 'folder': 'ch1', 'label': r'$3.6\mu$m'}}

for ( filt, filt_data ) in sims.iteritems():
	
	cat = pyfits.open( '%s/%s/sims/results.cat' % (root,filt_data['folder']) )[1].data
	
	cat = cat[ (cat['mag'] < filt_data['mag_lim']) ]
	cat = cat[ (cat['mag']-cat['mag_brightest'] < 5) ]
	cat = cat[(cat['re_arcsecs'] < 2.0) & (cat['n'] < 7.5) & (cat['mag_warning'] == 0)]
	ratio = 10.0**( -0.4*(cat['mag']-cat['nearest_mag']) )
	cat = cat[( cat['nblend'] == 0 ) | ( (cat['nearest'] > 1.0) & (ratio > 0.25) )]
	
	mag_diff = cat['mag_input']-cat['mag']
	
	blend_fraction = 10.0**( -0.4*(cat['mag_input']-cat['total_mag']) )
	xs = [ cat['nblend'], cat['nearest'], blend_fraction, 10.0**( -0.4*(cat['mag_input']-cat['nearest_mag']) ) ]
	xmins = [2, 1, 0, 0]
	xmaxs = [20, 5, 1, 10]
	xlabels = ['# Blended','Separation (")','Blend Fraction','Flux Ratio']
	xticklabelformats = ['%.0f', '%.1f', '%.1f', '%.0f']
	nxs = len( xs )
	
	for ( i, x, xmin, xmax, xlabel, xticklabelformat ) in zip( range( nxs ), xs, xmins, xmaxs, xlabels, xticklabelformats ):
		
		pyplot.subplot( nxs/2, nxs/2, i+1 )
		pyplot.plot( x, mag_diff, 'ko', ms=3 )
		pyplot.axis( [xmin,xmax,-0.7,0.7] )
		
		if i % 2 == 1:
			pyplot.gca().yaxis.set_major_formatter( ticker.NullFormatter() )
			pyplot.gca().xaxis.set_major_formatter( label_hider( xmin, xticklabelformat ) )
		else:
			pyplot.gca().xaxis.set_major_formatter( label_hider( xmax, xticklabelformat ) )
			pyplot.ylabel( 'Mag In - Mag Out' )
		pyplot.xlabel( xlabel )
	
	pyplot.gcf().set_size_inches( (8,5.8) )
	pyplot.savefig( 'ps/%s_error_correlations.eps' % filt )
	pyplot.clf()