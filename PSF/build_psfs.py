#!/usr/bin/python

import psf,dataio,match,os
from astropy.io import fits as pyfits
import matplotlib.pyplot as pyplot

# Proper use:
#  1. Use ds9 to generate a list of stars that you wish to use as input (xy format, wcs decimal coords0
#  2. Run SExtractor to generate a catalog that contains XWIN_WORLD, YWIN_WORLD
#  3. Run build_psfs.py
#  4. Look at the output postscript figures to identify stars that should be excluded, modify star list,
#      and rerun build_psfs.py

# first make a long list of potential plotting line styles
colors = ['k','b','r','g','c','m']
syms = ['-','--',':']
styles = [ '%s%s' % (color,sym) for sym in syms for color in colors ]

# next set the inner and outer annuli (in pixels) for the background)
configs = {	'w1': {'size': 30, 'inner': 30, 'outer': 60},		#WISE 
		'w2': {'size': 30, 'inner': 30, 'outer': 60},		#WISE 
	        'w3': {'size': 30, 'inner': 30, 'outer': 60},		#WISE 
                'w4': {'size': 30, 'inner': 30, 'outer': 60},		#WISE 
		'ch1': {'size': 30, 'inner': 40, 'outer': 60},		#IRAC
                'ch2': {'size': 30, 'inner': 40, 'outer': 60},		#IRAC
                'F606W': {'size': 60, 'inner': 100, 'outer': 200},	#HST/ACS
                'F814W': {'size': 60, 'inner': 100, 'outer': 200} }	#HST/ACS

# Loop through filters and build a psf for each
#    Assumes run from upper level, with data and star lists in subdirectories:
#      $FILTER/stars   	   #star list
#      $FILTER/$FILTER.cat #source extractor catalog
#		SExtractor catalog must be generated first and ocntaine XWIN_WORLD,YWIN_WORLD
#    You must set the proper filters to loop through
for f in ['w1','w2','ch1','ch2']:
   if(os.path.exists(f)):
	# load star positions from psf star list
	pos = dataio.rascii( '%s/stars' % f, silent=True )

	# then load source extractor catalog for the whole field
	if(f=='i'): cat = pyfits.open( '%s/%s.cat' % (f,f) )[1].data
	else: cat = pyfits.open( '%s/%s.cat' % (f,f) )[2].data

	# match positions
	( ml, ms ) = match.match( pos[:,0], pos[:,1], cat['xwin_world'], cat['ywin_world'], radius=20.0/3600, degrees=True )

	# load psf builder, passing it the fits file to extract stars from
	builder = psf.psf( '%s/%s.fits' % (f,f) )
	builder.inner = configs[f]['inner']	# inner radius for sky calculation
	builder.outer = configs[f]['outer']	# outer radius for sky calculation
	# tell the psf builder where the stars are (using the coordinates from source extractor)
	builder.set_stars( cat['xwin_world'][ms], cat['ywin_world'][ms] )

	# fetch combined psf and save
	psf_img = builder.combine_stars( size=configs[f]['size'] )	# size of output psf (30x30 pixels)
	hdu = pyfits.PrimaryHDU( psf_img )
	hdu.writeto( '%s/%s_psf.fits' % (f,f), clobber=True )

	# plot profiles of input stars
	# generate normalized flux profiles and plot
	ymax = 0
	xmax = 0
	for (i,star) in enumerate(builder):
		normalized = (star.flux-star.sky)/(star.flux.max()-star.sky)
		pyplot.plot( star.locs, normalized, styles[i], label='Star #%d' % (i+1) )
		ymax = max( ymax, normalized[1] )
		xmax = max( xmax, star.locs.max() )

	# final stuff
	pyplot.legend()
	pyplot.axhline( 0.0 )
	pyplot.xlabel( 'r (Pixels)' )
	pyplot.ylabel( 'Normalized Flux' )
	pyplot.axis( [0,8,0,1] )
	pyplot.savefig( '%s/%s_psf.ps' % (f,f) )
	pyplot.clf()
