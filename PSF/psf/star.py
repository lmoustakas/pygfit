import img_data,data_histogram
import numpy as np
import matplotlib.pyplot as pyplot
import scipy.ndimage

class star:
	ra = 0.0			# Right Ascension
	dec = 0.0			# Declination
	weight = 0.0			# Weight
	x = 0.0				# x coordinate
	y = 0.0				# y coordinate
	x_ind = 0			# x-index of central pixel in array
	y_ind = 0			# y-index of central pixel in array

	# a small cutout to work with directly
	cutout = np.array( [] )		# cutout
	cutout_xs = np.array( [] )	# x-coordinates for cutout pixels
	cutout_ys = np.array( [] )	# y-coordinates for cutout pixels
	dists = np.array( [] )		# distances to cutout pixels

	# sky value
	sky = 0.0			# sky value
	inner = 50.0			# inner radii used for calculating sky
	outer = 100.0			# outer radii used for calculating sky
	has_sky = False			# whether or not the sky has been measured

	# flux profile
	hist = []			# histogram object containing radial profile
	locs = []			# list of histogram bin locations
	flux = []			# list of histogram median fluxes
	sky_radius = 0.0		# radius at which flux drops to sky

	def __init__( self, ra, dec, weight=1, inner=None, outer=None ):
		self.ra = ra
		self.dec = dec
		self.weight = weight

		# x/y coordinate of star
		( self.x, self.y ) = img_data.rd2xy( self.ra, self.dec )
		( self.x_orig, self.y_orig ) = ( self.x, self.y )

		# index of central pixel, subtract 0.5 to move integer pixel coordinates to the corners of the pixels
		self.x_ind = int( np.floor( self.x - 0.5 ) )
		self.y_ind = int( np.floor( self.y - 0.5 ) )

		# inner/outer sky radii
		if inner is not None: self.inner = inner
		if outer is not None: self.outer = outer

		# extract an initial cutout to work with
		# size of cutout
		cutout_size = int( self.outer*2.5 )
		# blank canvas for cutout
		self.cutout = np.zeros( (cutout_size,cutout_size) )
		# image and cutout indexes
		( img_ys, img_ye, img_xs, img_xe, cut_ys, cut_ye, cut_xs, cut_xe ) = self._get_cutout_indexes( self.x_ind, self.y_ind, cutout_size, img_data.img.shape )
		# make cutout
		self.cutout[cut_ys:cut_ye,cut_xs:cut_xe] = img_data.img[img_ys:img_ye,img_xs:img_xe]

		# get x & y coordinates of cutout
		( self.cutout_ys, self.cutout_xs ) = np.mgrid[0:cutout_size,0:cutout_size]
		self.cutout_ys += img_ys - cut_ys + 1.0
		self.cutout_xs += img_xs - cut_xs + 1.0

		# correct coordinates to cutout coordinates
		self.x_ind -= ( img_xs - cut_xs )
		self.y_ind -= ( img_ys - cut_ys )
		#self.x -= ( img_xs - cut_xs ) + 0.5
		#self.y -= ( img_ys - cut_ys ) + 0.5

		# calculate distance to each cutout pixel
		self.dists = np.sqrt( (self.cutout_xs-self.x)**2.0 + (self.cutout_ys-self.y)**2.0 )

	def find_sky( self, force=False ):

		if self.has_sky and not force: return True

		# and find the sky as a median within the inner and outer radii
		self.sky = np.median( self.cutout[(self.dists < self.outer) & (self.dists > self.inner) & ~np.isnan(self.cutout)] )
		self.has_sky = True

	def get_sky_radius( self, force=False ):

		if self.sky_radius and not force: return self.sky_radius

		# find the sky value
		self.find_sky()

		# generate flux profile out to the outer sky radius

		# calculate distance to each pixel
		dists = self.dists.ravel()
		pixels = self.cutout.ravel()

		# cut at given radius
		w = np.where( dists < self.outer )[0]

		# and create histogram
		self.hist = data_histogram.data_histogram( dists[w] )
		self.locs = self.hist.locations

		# measure flux as a function of radius (in pixels)
		self.flux = np.zeros( self.locs.size )
		for (i,inds) in enumerate(self.hist):

			# make sure we have pixels in this bin...
			if len( inds ) == 0:
				self.flux[i] = 0
				continue

			# filter out nans
			m = ~np.isnan( pixels[w[inds]] )

			# store median
			self.flux[i] = np.median( pixels[w[inds[m]]] )

		# at what radial bin is the sky flux reached?
		cutind = (np.where( self.flux <= self.sky )[0])[0]

		# and the radius where the sky is reached
		self.sky_radius = self.hist.mins[cutind]

		return self.sky_radius

	def get_cutout( self, size=None, normalize=True, center=True, weight=False, subtract=True ):

		if size is None: size = int( np.ceil( self.get_sky_radius() ) )

		# get zeroed array of proper size
		cutout = np.zeros( (size, size) )

		# get indexes into the image and cutout
		( img_ys, img_ye, img_xs, img_xe, cut_ys, cut_ye, cut_xs, cut_xe ) = self._get_cutout_indexes( self.x_ind, self.y_ind, size, self.cutout.shape )

		# grab cutout
		cutout[cut_ys:cut_ye,cut_xs:cut_xe] = self.cutout[img_ys:img_ye,img_xs:img_xe]

		# shift the center of the star to the center of the cutout
		if center:
			shift_y = self.y - np.round( self.y )
			shift_x = np.round( self.x ) - self.x
			# and shift
			cutout = scipy.ndimage.shift( cutout, [shift_y,shift_x] )

		# subtract the sky
		if subtract: cutout -= self.sky

		# zero out negatives
		m = cutout < 0
		if m.sum(): cutout[m] = 0

		# normalize
		if normalize: cutout /= cutout.sum()

		# weight
		if weight: cutout *= self.weight

		return cutout

	def _get_cutout_indexes( self, xc, yc, size, shape ):

		down_size = int( size/2 )
		up_size = down_size if size % 2 == 1 else down_size - 1
		# get indexes into the image
		( img_ys, img_ye, img_xs, img_xe ) = ( yc - down_size, yc + up_size + 1, xc - down_size, xc + up_size + 1 )
		# get indexes into the cutout
		( cut_ys, cut_ye, cut_xs, cut_xe ) = ( 0, int( size ), 0, int( size ) )

		# adjust indexes if they went off the image border
		( img_height, img_width ) = shape
		if img_ys < 0:
			cut_ys -= img_ys
			img_ys = 0
		if img_xs < 0:
			cut_xs -= img_xs
			img_xs = 0
		if img_ye > img_height:
			cut_ye -= (img_ye - img_height)
			img_ye = img_height
		if img_xe > img_width:
			cut_xe -= (img_xe - img_width)
			img_xe = img_width
		# adjust indexes if they went off the cutout border
		if cut_ys < 0:
			img_ys -= cut_ys
			cut_ys = 0
		if cut_xs < 0:
			img_xs -= cut_xs
			cut_xs = 0
		if cut_ye > size:
			img_ye -= (cut_ye - size)
			cut_ye = size
		if cut_xe > size:
			img_xe -= (cut_xe - size)
			cut_xe = size

		return ( img_ys, img_ye, img_xs, img_xe, cut_ys, cut_ye, cut_xs, cut_xe )