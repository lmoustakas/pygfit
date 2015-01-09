#!/usr/bin/python

import pyfits
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter 

pyplot.ticklabel_format(axis='x',useOffset=False)
pyplot.axis( ymin=-3, ymax=3, xmax=20, xmin=6e-2 )
formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_powerlimits((-13,3))
pyplot.gca().set_xscale( 'log' )
pyplot.savefig( 'ps/close_sim.eps' )
pyplot.clf()
