#!/usr/bin/python

import pyfits
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker

class label_hider(ticker.Formatter):
	def __init__( self, to_hide, format ):
		self.hide = to_hide
		self.format = format
	def __call__( self, a, pos=None ):
		return '' if a == self.hide else self.format % a

pyplot.rcParams.update( {'legend.fontsize': 'medium', 'legend.numpoints': 1, 'figure.subplot.wspace': 0, 'figure.subplot.hspace': 0} )

cat = pyfits.open( 'integration_test/all_integration/pygfit.cat' )[1].data
integration = cat[cat['model'] == 'sersic']

cat = pyfits.open( 'integration_test/no_integration/pygfit.cat' )[1].data
no_integration = cat[cat['model'] == 'sersic']

pyplot.subplot( 2,1,1 )
pyplot.plot( no_integration['n'], no_integration['mag_hres']-no_integration['mag_initial'], 'kp', mfc='white', ms=12, mew=2, label='Without Integration' )
pyplot.plot( integration['n'], integration['mag_hres']-integration['mag_initial'], 'bo', ms=5, label='With Integration' )
pyplot.ylabel( 'Mag - Model Mag' )
pyplot.gca().xaxis.set_major_formatter( ticker.NullFormatter() )
pyplot.gca().yaxis.set_major_formatter( label_hider( -2, '%.0f' ) )
pyplot.axis( xmax=8.5 )
pyplot.legend( loc='upper left' )

pyplot.subplot( 2,1,2 )
m = no_integration['mag_hres']-no_integration['mag_initial'] > 0.05
pyplot.plot( no_integration['n'], no_integration['re_arcsecs'], 'bo', label='Correct Integration', ms=5 )
pyplot.plot( no_integration['n'][m], no_integration['re_arcsecs'][m], 'r<', label='Failed Integration', ms=12 )
pyplot.legend( loc='upper left' )
pyplot.gca().yaxis.set_major_formatter( label_hider( 16, '%.0f' ) )
pyplot.axis( ymin=-1, xmax=8.5, ymax=6 )
pyplot.ylabel( '$r_e$ (")' )
pyplot.xlabel( '$n$' )

pyplot.gcf().set_size_inches( (8,4) )
pyplot.savefig( 'ps/integration.eps' )
pyplot.clf()