import numpy as np
import match

class data_histogram:
	""" res = data_histogram( data, bin=1, min=np.array(), max=np.array(), nbins=np.array(), bins=np.array() stats=False )

	Generate and manipulate histograms of arrays.  Data array will be flattened.
	You can specify arbitrary bin locations with bins=np.array()
	"""

	bin = None	# bin width
	nbins = 1	# number of bins
	min = 0		# histogram min
	max = 0		# histogram max
	index = 0	# index of iterator
	inds = []	# indexes of original data for each bin
	hist = []	# actual histogram
	mins = []	# bin minimums
	locations = []	# bin locations

	means = []	# mean of data in each bin
	medians = []	# medians of data in each bin
	stddevs = []	# stddevs of data in each bin

	def __init__(self, datain, bin=None, min=None, max=None, nbins=None, stats=False, bins=None ):

		data = np.asarray( datain )

		# load defaults
		if min == None: min = data.min()
		if max == None: max = data.max()
		if bin != None: self.bin = bin

		if max == min: raise NameError('data.max() == data.min()')
		if data.size < 2: raise NameError('Must provide at least 2 data points')

		if bins is not None:
			bins = np.sort( np.asarray( bins ).ravel() )
			self.nbins = bins.size
		elif nbins == None:
			if bin == None: self.bin = 1
			self.nbins = int( np.ceil( (max - min)/self.bin ) )
		else:
			self.nbins = nbins
			if bin == None: self.bin = (max-min)/nbins

		if self.bin == None: self.bin = 1

		# store min and max in histogram
		self.min = min
		self.max = self.min + self.bin*self.nbins

		# set histogram and index holders (zeroes for now)
		self.inds = [0]*self.nbins
		self.hist = np.zeros( self.nbins )
		if stats:
			self.means = np.empty( self.nbins )
			self.stddevs = np.zeros( self.nbins )
			self.medians = np.empty( self.nbins )

		# unravel and sort data
		straight = data.ravel()
		myinds = np.argsort( straight )
		mydata = straight[myinds]

		# calculate bin locations
		if bins is not None:
			self.mins = bins
			self.locations = (self.mins + np.roll( self.mins, -1 ))/2.0
			# deal with the last bin
			self.locations[self.nbins-1] = self.mins[self.nbins-1]
			if max > self.mins[self.nbins-1]: self.locations[self.nbins-1] = (max+self.mins[self.nbins-1])/2.0
		else:
			self.mins = np.arange( self.nbins )*self.bin + min
			self.locations = self.mins + 0.5*self.bin

		# figure out what bin each datapoint belongs in
		if bins is not None:
			binned = match.nearest( self.mins, mydata, left=True )
			binned[mydata < self.mins[0]] = -1
			binned[mydata > max] = self.nbins+1
		else:
			binned = np.floor( (mydata - min)/self.bin )
		binned = binned.astype( 'int' )

		# detect bin changes
		diff = np.hstack( (np.array([1]), binned[1:] - binned[:-1]) )

		# loop through bins
		w, = np.where( diff )
		for i in range(w.size):

			# bin starting index
			sti = w[i]

			# which bin is this?
			bind = binned[sti]

			# skip it if it isn't a valid bin
			if ( bind < 0 ) | ( bind >= self.nbins ): continue

			# bin ending index
			if i == w.size-1:
				edi = binned.size
			else:
				edi = w[i+1]

			# record the number of elements in this bin
			self.hist[bind] = edi-sti

			# record the indexes of the data in the original array
			self.inds[bind] = myinds[sti:edi]

			if stats:
				self.means[bind] = np.mean( mydata[sti:edi] )
				self.medians[bind] = np.median( mydata[sti:edi] )
				if self.hist[bind] > 2: self.stddevs[bind] = np.std( mydata[sti:edi] )

	def fetch_bin(self, index, silent=False):
		if (index < 0) | (index > self.nbins-1):
			if silent: return []
			else: raise NameError('Bin out of bounds')

		if self.hist[index] == 0:
			return []
		else:
			return self.inds[index]

	def fetch_bins(self, indexes, silent=False):
		res = []
		for ind in indexes:
			try:
				inds = self.fetch_bin(ind, silent=silent)
			except:
				raise
			res.extend(inds)
		return res

	def __iter__(self):
		self.index = 0
		return self

	def next(self):

		if self.index == self.nbins:
			raise StopIteration

		index = self.index
		self.index += 1

		return self.fetch_bin( index )