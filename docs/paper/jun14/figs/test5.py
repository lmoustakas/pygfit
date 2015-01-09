#!/usr/bin/python

import matplotlib as pylab


fig = pylab.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# Plotting the data
plot_ax1, = ax1.plot()
plot_ax2, = ax2.plot()

# Setting the label colors
ax2.yaxis.set_offset_position('right') # To set the power indicator of ax2 
ax1.yaxis.label.set_color(plot_ax1.get_color())
ax2.yaxis.label.set_color(plot_ax2.get_color())

# Setting the ticker properties     
tkw = dict(size=4, width=1.5)
ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')        
ax1.tick_params(axis='y', colors=plot_ax1.get_color(), **tkw)
ax2.tick_params(axis='y', colors=plot_ax2.get_color(), **tkw)
ax1.tick_params(axis='x', **tkw)

# Setting the legend
lines = [plot_ax1, plot_ax2]
ax1.legend(lines, [l.get_label() for l in lines],'upper left')
