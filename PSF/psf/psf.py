import os,pyfits,wcs,star,img_data
import scipy.stats
import numpy as np

class psf:
	img_file = ''		# input fits file
	img = []		# image array
	wcs = {}		# wcs info from image file

	psf_file = ''		# output psf file
	stars = []		# list of star objects
	nstars = 0		# number of stars
	current_star = -1	# counter for iterator

	inner = 50		# inner radius for calculating sky
	outer = 100		# outer radius for calculating sky

	size = 0		# size of cutout

	def __init__( self, file, ras=None, decs=None, weights=None, extension=None ):
		""" psf.psf( img_file, extension, ras=None, decs=None, weights=None ) """

		if not os.path.isfile( file ): raise ValueError( 'The image file %s does not exist!' % file )

		# load image file and wcs info (store in global data holder module)
		img_data.set_img( file, extension )

		self.img_data = img_data

		# was star info passed as well?
		if ras is not None and decs is not None: self.set_stars( ras, decs, weights )

	# return iterator for fetching star objects
	def __iter__( self ):
		self.current_star = -1
		return self

	# return next star in iterator
	def next( self ):

		# iterate through number of stars
		self.current_star += 1
		if self.current_star == self.nstars: raise StopIteration

		# return star object
		return self.stars[self.current_star]

	# set star list
	def set_stars( self, ras, decs, weights=None ):

		# begin normalizing content
		ras = np.asarray( ras )
		decs = np.asarray( decs )
		if weights is None: weights = np.ones( decs.size )

		# check shape
		if ras.shape != decs.shape or weights.shape != ras.shape: raise ValueError( 'RA, Dec, and weights must be arrays of the same size!' )

		# make sure it is all a list
		if len( np.asarray( ras ).shape ) == 0:
			ras = [ ras ]
			decs = [ decs ]
			weights = [ weights ]

		# now generate list of star objects
		self.stars = []
		for ( ra, dec, weight ) in zip( ras, decs, weights ): self.stars.append( star.star( ra, dec, weights, inner=self.inner, outer=self.outer ) )

		self.nstars = len( self.stars )

	# find the size of the cutout that encompasses the sky for everything
	def get_cutout_size( self, force=False ):

		if self.size and not force: return self.size

		if not self.nstars: raise ValueError( 'You must set some stars before doing anything!' )

		# find the sky radius for every star
		sizes = np.empty( self.nstars )
		for (i,star) in enumerate( self ): sizes[i] = star.get_sky_radius()

		# and set the cutout size to be the maximum sky radius of all the stars
		self.size = int( np.ceil( sizes.max() ) )

		# make sure the size is odd
		if not( self.size % 2 ): self.size += 1

		# and return
		return self.size

	# now get the combination of all the psfs
	def combine_stars( self, mode='median', size=None, normalize=True, center=True, weight=False, subtract=True ):

		# we have to fetch the cutout size to make sure the sky and profiles are fit for all the stars
		if not self.size: self.get_cutout_size()

		# now what size do we want to use?
		if size is None: size = self.get_cutout_size()
		mode = mode.lower()

		# data cube for storing all cutouts
		cube = np.empty( (self.nstars,size,size) )

		# get cutouts and store in cube
		tot_weight = 0
		if mode == 'weighted':
			weight = True
		else:
			weight = False

		for (i,star) in enumerate( self ):
			cube[i,:,:] = star.get_cutout( size, weight=weight, normalize=normalize, center=center, subtract=subtract )
			tot_weight += star.weight

		if self.nstars == 1: return np.squeeze( cube )

		# now combine and return
		if mode == 'median':
			return scipy.stats.nanmedian( cube, axis=0 )
		elif mode == 'mean' or mode == 'weighted':
			mean = scipy.stats.nanmean( cube, axis=0 )
			if weight: mean /= tot_weight
			return mean
		else:
			raise ValueError( 'unrecognized mode: %s' % mode )