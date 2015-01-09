#!/usr/bin/python

#now applying e+k correctioin
#no longer plotting line

import matplotlib
import matplotlib.gridspec as gridspec
from pylab import *
from scipy import *
from numpy import *
from matplotlib import lines
from matplotlib import transforms
from matplotlib.pyplot import *
from matplotlib.patches import *
import matplotlib.axes
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import *
from matplotlib.spines import *


formatter = ScalarFormatter()
formatter.set_scientific(False)
#formatter.set_powerlimits((-13,3))

ax=subplot(111)
#ax.xaxis.set_major_formatter(formatter)
ax.set_xscale('log')
ax.axis([6e-2,20,-3,3])
#ax.set_xlim(0, 20)

ax.set_xticks([0.1,1,10])
ax.set_xticklabels(('0.1','1','10'))

#ax.ticklabel_format(style='plain')
#ax.ticklabel_format(style='sci',scilimits=(-13,13),axis='x')
ax.plot(linewidth=2, style='plain',scilimits=(-13,13),axis='x')

savefig('test.png')
savefig('test.eps')
savefig('test.pdf')
