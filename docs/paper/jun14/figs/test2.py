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

width=7.5
height=7.5
#rc("font",**{'family':'serif','serif':['Times']})
#rc("font",family='cursive')
rc("font",family='serif')

rc("figure.subplot", left=(22/72.27)/width)
rc("figure.subplot", right=(width-10/72.27)/width)
rc("figure.subplot", bottom=(14/72.27)/height)
rc("figure.subplot", top=(height-7/72.27)/height)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
figure(figsize=(width, height))
 
xbins=10
ybins=10
#xedges=linspace(3e15,5e15,xbins)
#yedges=linspace(0.1,0.25,ybins)
xedges=linspace(6e-2,20,xbins)
yedges=linspace(-3,3,ybins)

#fig, ax=subplots(ybins,xbins,figsize=(10,11),sharex=True,sharey=False)

#fig.subplots_adjust(hspace=0,wspace=0)


#Format tickmarks on Axes
majorLocator   = MultipleLocator(0.05)
majorFormatter = FormatStrFormatter('%0.3f')
xform = NullFormatter()
minorFormatter = FormatStrFormatter('%.1f')
minorLocator   = MultipleLocator(0.025)
#minorLocator = ticker.AutoMinorLocator(n=500)



#TOP PLOT
#Define plot and finish setting tick marks
#ax=subplot(211)
gs=gridspec.GridSpec(2,1,height_ratios=[3,1])
gs.update(hspace=0)
#ax=subplot2grid((1,2),(0,0),rowspan=2)
ax=subplot(gs[0])
#ax.axis([9e12,1.3e15,0.005,0.25])
ax.axis([7e13,6e14,0.005,0.25])
ax.axis([7e13,0.999e15,0.008,0.25])
ax.axis([7e13,5.999e14,0.1,1.3])
ax.axis([3e13,1.e15,0.01,1.3])
ax.axis([7e13,1.e15,0.05,1.])
ax.axis([6e-2,20,-3,3])
gcf().subplots_adjust(bottom=0.10)
gcf().subplots_adjust(left=0.17)
gcf().subplots_adjust(right=0.94)

ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_major_formatter(xform)
ax.yaxis.set_minor_locator(minorLocator)


formatter=ScalarFormatter()
#formatter=LogFormatter(labelOnlyBase=False)
#formatter=LogFormatter(labelOnlyBase=True)
formatter=FormatStrFormatter('%g')
#formatter.set_scientific(False)
ax.semilogx(major_locator=majorLocator,antialiased=True,linewidth=2) #,style='plain')
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_minor_formatter(formatter)

#ax.set_xticklabels((''))

ax.tick_params(axis='x',which='major',labelsize='large' ,length=13,width=2)
ax.tick_params(axis='y',which='major',labelsize='large' ,length=13,width=2)
ax.tick_params(axis='x',which='minor',labelsize='large' ,length=5,width=2)
ax.tick_params(axis='y',which='minor',labelsize='large' ,length=5,width=2)


#ax.annotate('$\mathrm{Excluding\ ICL}$',(1.e14,0.09),color='r',size='x-large',rotation=-0)
#ax.annotate('$\mathrm{All\ Stars}$',(1.e14,0.11),color='b',size='x-large',rotation=-0)
#ax.annotate('$\mathrm{Zhang\ et\ al.\ (2011)}$',(1.e14,0.073),color='k',size='x-large',rotation=-0)
#
#labels=setp(gca(), ylabel='$M_\mathrm{*}/M_\mathrm{gas}$',xlabel='$M_{500}$ [$\mathrm{M}_\odot$]')
#setp(labels,fontsize='xx-large')

savefig('test.png')
savefig('test.eps')
savefig('test.pdf')
