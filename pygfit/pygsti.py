#!/usr/bin/env python

""" PYthon Galaxy Simulation Tool for Images: A tool for adding artifical galaxy profiles to images. """

import pyfits,utils,models,os
import numpy as np
import gpu

class pygsti(object):
	""" PYthon Galaxy Simulation Tool for Images: A tool for adding artifical galaxies to images. """

	# catalog stuff
	cat_file = ''		# filename for input catalog
	cat_out = ''		# filename for output catalog
	reg_out = ''		# filename for output region file
	cat = {}		# catalog
	ncat = 0		# number of objects in catalog

	# image stuff
	img_file = ''		# filename of image
	rms_file = ''		# filename of rms image
	psf_file = ''		# filename of psf image
	has_rms = False		# whether or not there is an input rms file
	img = np.array( [] )	# image
	hdr = np.array( [] )	# image header
	psf = np.array( [] )	# psf image
	mask = np.array( [] )	# image mask - True/False
	wcs = {}		# wcs info

	# object state
	data_loaded = False	# whether or not the input catalog has been loaded
	slops_loaded = False	# whether or not the slop objects have been loaded
	current_frame = -1	# index for iterator

	# config
	config = {}

	# required configuration fields
	required = ['catalog','image','psf','magzero','pad_length']
	# default configuration fields
	default = {'mask': '', 're_arcsec': True, 'model_type': 0, 'id': 1, 'ra': 2, 'dec': 3, 'mag': 4, 'point_mag': 4, 're': 5, 'n': 6, 'pa': 7, 'ba': 8, 'sim_dir': '', 'individual_dirs': False, 'as_is': True, 'random_mags': False, 'mix_parameters': False, 'random_positions': False, 'catalog_suffix': '_sim.cat', 'image_suffix': '_sim.fits', 'rms_suffix': '_sim_rms.fits', 'add_noise': 0.0, 'add_background': 0.0, 'region_suffix': '_sim.reg', 'number_frames': 10, 'gals_per_frame': 100, 'output_format': 'fits', 'model_size': 10, 'mask_value': False, 'output_models': False, 'model_dir': 'models', 'clobber': True, 'include_errors': False, 'x_offset': 0, 'y_offset': 0, 'use_integration': True, 'gpu': False, 'gpu_nthreads': 512}

	##############################
	## initialize pygsti object ##
	##############################
	def __init__( self, catalog=None, image=None, psf=None, magzero=None, pad_length=None, config='pygsti.config' ):
		""" pygsti( catalog=None, image=None, psf=None, magzero=None, pad_length=None, config='pygsti.config' )
		
		Wrapper class for slop.  Processes configuration file to simplify the creation of simulated galaxy fields.
		Will add objects in ascii file `catalog` to `image` using `psf` with zeropoint of `magzero` and padding length of `pad_length`.
		For more detailed options use config, which can be the filname of a config file or a dictionary of options/values. """

		self.config = {}
		# check for valid config file
		if type( config ) == type( {} ):
			self.config = config
		elif type( config ) == type( '' ):
			if not os.path.isfile( config ) and config != 'pygsti.config':
				raise ValueError( 'Could not find config file: %s' % config )
			elif os.path.isfile( config ):
				self.config = utils.load_config( config )
		elif type( config ) == type( True ):
			self.config = self.load_default_config()
		else:
			raise ValueError( 'Improper config value on pygsti init!' )

		# check for configs passed through init call
		if catalog is not None: self.config['catalog'] = catalog
		if image is not None: self.config['image'] = image
		if psf is not None: self.config['psf'] = psf
		if magzero is not None: self.config['magzero'] = magzero
		if pad_length is not None: self.config['pad_length'] = pad_length

		# check for required fields (set in class definition)
		for field in self.required:
			if not self.config.has_key( field ): raise ValueError( 'Required field, %s, missing from configuration!' % field )

		# Make sure padding length is an int
		self.config['pad_length'] = int( float( self.config['pad_length'] ) )

		# set default fields
		for key in self.default: self.config.setdefault( key, self.default[key] )

		# copy some of the configuration parameters into the object (for convenience)
		self.img_file = self.config['image']
		self.psf_file = self.config['psf']
		# check if an RMS image was passed
		if self.config.has_key( 'rms_image' ):
			if os.path.isfile( self.config['rms_image'] ):
				self.has_rms = True
				self.rms_file = self.config['rms_image']
			else:
				raise ValueError( 'Specified input RMS image does not exist!' )

		# these can't be empty
		if self.config['image_suffix'] == '': self.config['image_suffix'] = '_sim.fits'
		if self.config['rms_suffix'] == '': self.config['rms_suffix'] = '_sim_rms.fits'
		if self.config['catalog_suffix'] == '': self.config['catalog_suffix'] = '_sim.cat'
		if self.config['region_suffix'] == '': self.config['region_suffix'] = '_sim.reg'

		# default number of frames
		if self.config['as_is']: self.config['number_frames'] = 1

		# need re's in arcseconds
		if not self.config['re_arcsec'] and not self.config['pixscale']: raise ValueError( 'If re is not in arcseconds, then you must specify the pixel scale with pixscale!' )
	
		# gpu support?
		self.config['gpu_nthreads'] = int( self.config['gpu_nthreads'] )
		if self.config['gpu']:
			# if so, see if it is actually supported...
			if not gpu.has_pycuda:
				log.warn( 'This python distribution does not support GPU processing.  This is accomplished with the pycuda package, which could not be loaded.' )
				self.config['gpu'] = False
			# if gpu is supported then initialize a gpu object and store it in the config
			else:
				self.config['gpu'] = gpu.gpu( self.config['gpu_nthreads'] )
	
	# return iterator for fetching frame properties
	def __iter__( self ):
		self.current_frame = -1
		return self

	# return next gblend in iterator
	def next( self ):

		# iterate through number of frames
		self.current_frame += 1
		if self.current_frame == self.config['number_frames']: raise StopIteration

		# get frame output names
		names = self.get_output_names( self.current_frame, self.config['number_frames'] )
		names['index'] = self.current_frame
		return names

	#########################
	## load default config ##
	#########################
	def load_default_config( self ):

		return {'catalog': 'input.cat', 'image': 'image.fits', 'psf': 'psf.fits', 'mask': 'mask.fits', 'magzero': 20, 'pad_length': 25.0, 'pixscale': 0.0, 'min_size': 20, 'max_size': 50, 'catalog_output': 'sim.cat', 'image_output': 'sim.fits', 'rms_output': 'sim_rms.fits', 'region_output': 'sim.reg', 'output_models': True, 'position_border': [-360, 360, -90, 90], 'sim_dir': 'sims', 'mag_err': 9, 'point_mag_err': 10, 're_err': 11, 'pa_err': 12, 'ba_err': 13}

	#########
	## go! ##
	#########
	def go( self ):
		""" pygsti.go()

		Run the configured simulations. """

		# load catalog and images
		self.load_data()

		# generate the images
		self.make_images()

	###############
	## load data ##
	###############
	def load_data( self, catalog=None ):
		""" pygsti.load_data()
		
		Loads images and the catalog according to the configuration. """

		# load catalog
		if catalog is None:
			( catalog, valid, unknown ) = utils.load_data_catalog( self.config['catalog'], self.config )
		else:
			if type( catalog ) != type( {} ): raise ValueError( 'Please pass a valid catalog!' )

		# and store
		self.catalog = catalog
		self.ncat = catalog['id'].size

		# load and store the images
		( self.img, self.hdr ) = utils.load_image( self.img_file, header=True )
		self.wcs = utils.get_wcs_info( self.hdr )
		self.psf = utils.load_psf( self.psf_file )
		if self.config['mask']:
			if not os.path.isfile( self.config['mask'] ): raise ValueError( 'The specified mask file was not found!' )
			self.mask = utils.load_image( self.config['mask'] )
			if self.mask.shape != self.img.shape: raise ValueError( 'Mask file and image file have different dimensions!' )

		# max image size must be at least size of psf image
		if self.config.has_key( 'max_size' ):
			if self.config['max_size'] < self.psf.shape[0]: self.config['max_size'] = self.psf.shape[0]

		# convert re to simulation pixels
		# first convert to arcseconds if needed
		if not self.config['re_arcsec']:
			self.catalog['re'] *= self.config['pixscale']*3600
			if self.catalog.has_key( 're_err' ): self.catalog['re_err'] *= self.config['pixscale']*3600
		# now convert to pixels
		self.catalog['re_pix'] = self.catalog['re']/self.wcs['scale']
		if self.catalog.has_key( 're_err' ): self.catalog['re_pix_err'] = self.catalog['re_err']/self.wcs['scale']

		# finally fetch a mask image so we know where objects can be (used for randomizing positions and by slop)
		mask = self.mask if self.config['mask'] else self.img
		self.mask = get_mask( mask, self.config['mask_value'] )

		# position border?  Prepare corner locations now because we can do things in x/y instead of ra/dec
		if not self.config.has_key( 'position_border' ):
			self.config['position_border'] = False
		else:
			# must be a list with either 4 or 8 elements
			if type( self.config['position_border'] ) != type( [] ) or ( len( self.config['position_border'] ) != 4 and len( self.config['position_border'] ) != 8 ):
				self.config['position_border'] = False
				raise ValueError( 'Position border must be a list of ra/decs specifying the generation region' )

			# we want it to be a dictionary with corner locations.  If 4 elements are passed then make it such a dictionary
			if len( self.config['position_border'] ) == 4:
				# convert to x/y
				( minra, maxra, mindec, maxdec ) = tuple( self.config['position_border'] )
				# filter out default value
				if minra > -359 or maxra < 359 or mindec > 89 or maxdec < 89:
					( xs, ys ) = utils.rd2xy( self.wcs, [minra, maxra], [mindec, maxdec] )
					# store it as a dictionary by corner location: left, top, right, bottom.
					self.config['position_border'] = { 'left': { 'x': xs.min(), 'y': ys.max() }, 'top': { 'x': xs.max(), 'y': ys.max() }, 'right': { 'x': xs.max(), 'y': ys.max() }, 'bottom': { 'x': xs.min(), 'y': ys.min() } }
				else:
					self.config['position_border'] = False
			else:
				# figure out where each corner is - left, top, right, bottom
				border = np.array( self.config['position_border'] )
				( xs, ys ) = utils.rd2xy( self.wcs, border[[0,2,4,6]], border[[1,3,5,7]] )
				wl = xs.argmin(); wt = ys.argmax(); wr = xs.argmax(); wb = ys.argmin()
				self.config['position_border'] = { 'left': { 'x': xs[wl], 'y': ys[wl] }, 'top': { 'x': xs[wt], 'y': ys[wt] }, 'right': { 'x': xs[wr], 'y': ys[wr] }, 'bottom': { 'x': xs[wb], 'y': ys[wb] } }

		# okay...
		self.data_loaded = True

	################
	## load slops ##
	################
	def make_images( self ):
		""" pygsti.make_images()
		
		Generate the requested images. """

		# make sure data was loaded
		if not self.data_loaded: self.load_data()

		# how many frames are we going to generate?
		nframes = self.config['number_frames']
		if self.config['as_is']:
			per_frame = self.catalog['id'].size
		else:
			per_frame = self.config['gals_per_frame']

		# if we are mixing up parameters, then do some stuff outside of the for loop
		if self.config['mix_parameters']:

			# find sersics
			ws = (self.catalog['model'] == 'sersic')[0]
			ns = len( ws )

		# if positions are randomized then run where on mask outside of loop
		if self.config['random_positions']:
			wmask = np.where( self.mask )
			# border on random positions?
			if self.config.has_key( 'position_border' ) and self.config['position_border']:

				# rules to find things "inside" each face of the rectangle
				corners = self.config['position_border']
				rules = [ {'c1': 'left', 'c2': 'top', 'mult': -1, 'field': 'y'}, {'c1': 'top', 'c2': 'right', 'mult': -1, 'field': 'x'}, {'c1': 'right', 'c2': 'bottom', 'mult': 1, 'field': 'y'}, {'c1': 'bottom', 'c2': 'left', 'mult': 1, 'field': 'x'} ]

				# loop through the rules and mask out pixels that don't fall in the position border
				for rule in rules:
					# get position of corners for this side
					c1 = corners[rule['c1']]
					c2 = corners[rule['c2']]

					# now find pixels on the appropriate side of the line connecting these two corners

					#  non-vertical lines
					if c2['x'] != c1['x']:
						# calculate slope and y-intecept of line connecting corners
						m = (c2['y']-c1['y'])/(c2['x']-c1['x'])
						b = c2['y']-m*c2['x']

						# Find matching pixels.  Things with rule['mult'] == -1 basically use a < operator
						if rule['field'] == 'y':
							w = np.where( wmask[0]*rule['mult'] > (m*wmask[1] + b)*rule['mult'] )[0]
						else:
							w = np.where( wmask[1]*rule['mult'] > ((wmask[0] - b)/m)*rule['mult'] )[0]
					# vertical lines
					else:
						w = np.where( wmask[1]*rule['mult'] > c2['x']*rule['mult'] )[0]

					# and filter wmask appropriately
					wmask = ( wmask[0][w], wmask[1][w] )

		# rms file to pass to slop
		rms_file = self.rms_file if self.has_rms else None

		# start generating them.  Loop through pygsti object, retruning output files for each frame
		for output in self:

			# if clobber is false and this simulation is already created, then skip
			if not self.config['clobber'] and os.path.isfile( output['img_out'] ) and os.path.isfile( output['cat_out'] ): continue

			# load the slop object
			generator = slop( self.img, self.psf, self.config, self.hdr, rms_img=rms_file )

			# now build the input catalog based on config settings
			cat = {}

			# just use the given catalog
			if self.config['as_is']:
				cat = self.catalog.copy()

			# mix up the parameters
			elif self.config['mix_parameters']:

				# randomly select ids and model type
				rind = np.random.random_integers( 0, self.ncat-1, size=per_frame )
				cat['model'] = self.catalog['model'][rind]
				# set empty sersic parameters
				for field in sersics: cat[field] = np.zeros( per_frame )

				# mix up all ras, and decs
				for field in ['ra','dec']: cat[field] = self.catalog[field][np.random.random_integers( 0, self.ncat-1, size=per_frame )]
				# mix up mags
				inds = np.random.random_integers( 0, self.ncat-1, size=per_frame )
				cat['mag'] = self.catalog[field][inds]
				if cat.has_key( 'mag_err' ): cat['mag_err'] = self.catalog[field][inds]

				# mix up sersic parameters - only for things that are sersics
				wn = (cat['model'] == 'sersic')[0]
				nn = len( wn )
				if nn > 0:
					for field in ['re','re_pix','n','pa','ba']:
						err_field = '%s_err' % field
						inds = np.random.random_integers( 0, ns-1, size=nn )
						# mix up parameter
						cat[field][wn] = self.catalog[field][ws[inds]]
						# and its error (but keep the same error with the same parameter)
						if self.catalog.has_key( err_field ): cat[err_field][wn] = self.catalog[err_field][ws[inds]]

			# select from catalog
			else:
				rind = np.random.random_integers( 0, self.ncat-1, size=per_frame )
				# mix up parameters
				for field in ['model','id','mag','ra','dec','re','re_pix','n','pa','ba']:
					cat[field] = self.catalog[field][rind]
				# and errors (if they exist)
				for field in ['mag_err','re_err','re_pix_err','n_err','pa_err','ba_err']:
					if self.catalog.has_key( field ): cat[field] = self.catalog[field][rind]

			# generate random magnitudes
			if self.config['random_mags']:

				# first build cumulative distribution of input catalog
				sinds = self.catalog['mag'].argsort()
				mags = self.catalog['mag'][sinds]
				cumulative = mags.cumsum()/mags.sum()
				# set first element to zero to avoid out of range interpolation errors
				cumulative[0] = 0

				# generate random values from 0 to 1
				rand_vals = np.random.rand( cat['ra'].size )
				# and use interpolation and the cumulative sum to convert that to mag given LF
				cat['mag'] = np.interp( rand_vals, cumulative, mags )

			# generate ids
			ids = []
			for i in range( cat['ra'].size ): ids.append( 'sim_%d' % i )
			cat['id'] = np.array( ids )

			# randomize positions
			if self.config['random_positions']:
				n = len( wmask[0] )
				rind = np.random.random_integers( 0, n-1, size=per_frame )
				( ys, xs ) = ( wmask[0][rind], wmask[1][rind] )
				( cat['ra'], cat['dec'] ) = utils.xy2rd( self.wcs, np.array( xs )+1, np.array( ys )+1 )

			# output the region file
			utils.write_region( output['reg_out'], cat['ra'], cat['dec'], 1.5/3600 )

			# create models
			min_size = None
			max_size = None
			if self.config.has_key( 'min_size' ): min_size = self.config['min_size']
			if self.config.has_key( 'max_size' ): max_size = self.config['max_size']
			# just pass along the whole configuration, as many items are needed
			generator.create_models( cat, min_size=min_size, max_size=max_size )

			# output models if requested
			if self.config['output_models']:
				for (j,model_obj) in enumerate(generator.models):

					# get header with updated wcs
					# get min x/y inds for image
					( minx, maxx, miny, maxy ) = model_obj.slop_image_inds
					# copy header and correct wcs for cutout
					hdr = self.hdr.copy()
					hdr['crpix1'] -= minx
					hdr['crpix2'] -= miny
					# and write out
					file_out = '%s%s.fits' % (output['model_dir'],model_obj.id)
					if os.path.isfile( file_out ): os.unlink( file_out )
					pyfits.writeto( file_out, model_obj.model_img, hdr, clobber=True )

			# get image with all simulated galaxies
			img = generator.get_full_model()

			# add noise?
			if self.config['add_noise']: img += np.random.normal( scale=self.config['add_noise'], size=img.shape )

			# add in a background?
			if self.config['add_background']: img += self.config['add_background']

			# before writing out simulated image, delete if it already exists
			if os.path.isfile( output['img_out'] ): os.remove( output['img_out'] )
			pyfits.writeto( output['img_out'], img, self.hdr )

			# are we also modifying the RMS image?
			if self.has_rms:
				rms = generator.get_full_rms_model()
				if os.path.isfile( output['rms_out'] ): os.remove( output['rms_out'] )
				pyfits.writeto( output['rms_out'], rms, self.hdr )

			# now prepare simulated catalog for output

			# first add in flux field
			cat['flux'] = 10.0**( -0.4*( cat['mag'] - self.config['magzero'] ) )
			
			# also extract the flux_re and mag_re from the model objects (using the generator)
			#cat['flux_re'] = generator.

			# if errors were included, then go ahead and copy the actual parameters to new data fields, and add errors to the "input" parameters
			if self.config['include_errors']:
				# mag, re, and n all have correlated errors.
				# Therefore just use one normally distributed random variable to calculate the error in these quantities.
				correlated = np.random.normal( size=cat[field].size )

				# now calculate error in mag and n assuming the errors are 90% correlated
				cat['mag_real'] = cat['mag'].copy()
				cat['n_real'] = cat['n'].copy()
				cat['mag'] += 0.9*correlated*np.abs( cat['mag_err'] ) + 0.1*np.random.normal( size=cat[field].size )*np.abs( cat['mag_err'] )
				cat['n'] -= 0.9*correlated*np.abs( cat['n_err'] ) + 0.1*np.random.normal( size=cat[field].size )*np.abs( cat['n_err'] )
				# the same for re
				cat['re_real'] = cat['re'].copy()
				cat['re_pix_real'] = cat['re_pix'].copy()
				re_uncorrelated = np.random.normal( size=cat[field].size )
				cat['re'] -= 0.9*correlated*np.abs( cat['re_err'] ) + 0.1*re_uncorrelated*np.abs( cat['re_err'] )
				cat['re_pix'] -= (0.9*correlated*np.abs( cat['re_err'] ) + 0.1*re_uncorrelated*np.abs( cat['re_err'] ))/self.wcs['scale']
				# and update the flux fields accordingly
				cat['flux_real'] = cat['flux'].copy()
				cat['flux'] = 10.0**( -0.4*( cat['mag'] - self.config['magzero'] ) )

				# finally ba and pa don't correlate with the others
				for field in ['pa','ba']:
					real_field = '%s_real' % field
					err_field = '%s_err' % field
					cat[real_field] = cat[field].copy()
					cat[field] += np.random.normal( size=cat[field].size )*np.abs( cat[err_field] )

				# these parameters can't be less than zero
				for field in ['mag','n','ba','re','re_pix']: cat[field] = np.abs( cat[field] )
				# some also have upper limits
				for ( field, limit ) in zip( ['n','ba'], [8,1] ):
					m = cat[field] > limit
					if m.sum(): cat[field][m] = limit

			# Finally, output the catalog
			if self.config['output_format'].lower() == 'fits':
				utils.output_fits_catalog( cat, output['cat_out'], cat.keys() )
			else:
				utils.output_ascii_catalog( cat, output['cat_out'], cat.keys() )

	######################
	## get output names ##
	######################
	def get_output_names( self, n, nframes=1 ):
		""" output = pygsti.get_output_names( n, nframes )
		
		Returns the output paths in a dictionary given the current frame number, total frames, and current config """

		# output directory
		dir_out = ''
		if self.config.has_key( 'sim_dir' ):
			dir_out = self.config['sim_dir']
			# make sure it ends with a slash
			if len( dir_out ) == 0: dir_out = '.'
			if dir_out[-1] != os.sep: dir_out += os.sep

		# generate an id string
		number = ''
		format = '%%0%dd' % np.ceil( np.log10( nframes ) )
		number = format % n
		# are we working with individual directories?
		if self.config['individual_dirs']:
			dir_out = '%s%s%s' % (dir_out,number,os.sep)

		# make sure the directory exists
		if not os.path.isdir( dir_out ): os.makedirs( dir_out, mode=0755 )

		# now get output names
		img_out = self._get_name( number, nframes, 'fits', 'image_output', 'image_suffix' )
		rms_out = self._get_name( number, nframes, 'fits', 'rms_output', 'rms_suffix' )
		cat_out = self._get_name( number, nframes, 'cat', 'catalog_output', 'catalog_suffix' )
		reg_out = self._get_name( number, nframes, 'reg', 'region_output', 'region_suffix' )

		# also get the model directory
		model_dir = self.config['model_dir']
		if model_dir[-1] != os.sep: model_dir += os.sep
		# and make sure it exists (if needed)
		if self.config['output_models'] and not os.path.isdir( dir_out + model_dir ): os.makedirs( dir_out + model_dir, mode=0755 )

		# and return
		return { 'img_out': dir_out + img_out, 'rms_out': dir_out + rms_out, 'cat_out': dir_out + cat_out, 'reg_out': dir_out + reg_out, 'model_dir': dir_out + model_dir, 'dir': dir_out }

	##############
	## get name ##
	##############
	def _get_name( self, number, nframes, ext, output_key, suffix_key ):
		""" name = pygsti._get_name( number, nframes, ext, output_key, suffix_key )
		
		This makes sure the name of frame output will be unqiue, and based on the configuration.
		Pass it the current frame number, the total number of frames, extension, and relevant config key names. """

		# the user can specify a filename or a filename suffix.  As long as we are outputting to individual directories,
		# or only one frame is generated, these things work fine.  Otherwise, we have to include the frame number in the name.
		out = ''
		if self.config.has_key( output_key ):
			if self.config['individual_dirs'] or nframes == 1:
				out = self.config[output_key]
			else:
				out = utils.swap_extension( self.config[output_key], '.%s' % ext, '_%s.%s' % (number,ext) )
		else:
			if self.config['individual_dirs'] or nframes == 1:
				out = utils.swap_extension( os.path.basename( self.img_file ), '.fits', self.config[suffix_key] )
			else:
				out = utils.swap_extension( os.path.basename( self.img_file ), '.fits', '_%s%s' % (number,self.config[suffix_key]) )

		return out

	###################
	## output config ##
	###################
	def output_config( self, filename=None, config=None ):
		""" pygfit.output_config( filename=None, config=None )
		
		Output the given configuration to the given file.
		If no file is specified, config is printed to screen.
		If no config is specified, then the current configuration is used.
		"""

		# Irritating part: specify output for configuration file.  Groups things and specify comments for all configuration parameters, as well as output format.
		# also have to specify field order
		image_order = ['image', 'rms_image', 'psf', 'magzero', 'mask', 'mask_value']
		image = {	'image':		{'format': '%s', 'comment': 'Filename for input image'},
				'rms_image':		{'format': '%s', 'comment': 'Filename for input RMS image'},
				'psf':			{'format': '%s', 'comment': 'Filename for psf image'},
				'magzero':		{'format': '%7.4f', 'comment': 'Magnitude zeropoint for image'},
				'mask':			{'format': '%s', 'comment': 'Filename of mask image'},
				'mask_value':		{'format': '%7.4f', 'comment': 'Consider pixels with this value to be off image.  NaNs are always ignored'} }

		catalog_order = ['catalog', 're_arcsec', 'pixscale']
		catalog = {	'catalog':		{'format': '%s', 'comment': 'Filename for catalog of input sources'},
				're_arcsec':		{'format': '%s', 'comment': "Whether or not Re in catalog is in arcseconds (True/False)"},
				'pixscale':		{'format': '%12.6e', 'comment': 'Pixel scale for catalog in degrees per pixel (if RE_ARCSEC is False)'} }

		format_order = ['model_type', 'id', 'ra', 'dec', 'mag', 'point_mag', 're', 'n', 'pa', 'ba','mag_err','point_mag_err','re_err','n_err','pa_err','ba_err']
		form = '%d' if type( self.config['model_type'] ) == type( 0 ) else '%s'
		format = {	'model_type':		{'format': form, 'comment': "Galaxy model type (either 'sersic' or 'point')"},
				'id':			{'format': form, 'comment': 'Unique id'},
				'ra':			{'format': form, 'comment': 'Right Ascension'},
				'dec':			{'format': form, 'comment': 'Declination'},
				'mag':			{'format': form, 'comment': 'Magnitude for sersic models'},
				'point_mag':		{'format': form, 'comment': 'Magnitude for point models'},
				're':			{'format': form, 'comment': 'Effective radius (sersic only)'},
				'n':			{'format': form, 'comment': 'Sersic index (sersic only)'},
				'pa':			{'format': form, 'comment': 'Position Angle (sersic only)'},
				'ba':			{'format': form, 'comment': 'Axis Ratio (sersic only)'},
				'mag_err':		{'format': form, 'comment': 'Magnitude error for sersic models (only needed if INCLUDE_ERRORS is True)'},
				'point_mag_err':	{'format': form, 'comment': 'Magnitude error for point models (only needed if INCLUDE_ERRORS is True)'},
				're_err':		{'format': form, 'comment': 'Effective radius error (only needed if INCLUDE_ERRORS is True)'},
				'n_err':		{'format': form, 'comment': 'Sersic index error (only needed if INCLUDE_ERRORS is True)'},
				'pa_err':		{'format': form, 'comment': 'Position angle error (only needed if INCLUDE_ERRORS is True)'},
				'ba_err':		{'format': form, 'comment': 'Axis Ratio error (only needed if INCLUDE_ERRORS is True)'} }

		model_order = ['use_integration','gpu','gpu_nthreads','pad_length', 'add_noise', 'add_background', 'model_size', 'min_size', 'max_size']
		model = {	'use_integration':	{'format': '%s', 'comment': 'Whether or not to use integration to properly calculate hard-to-estimate sersic models'},
				'gpu':			{'format': '%s', 'comment': 'Whether or not to attempt to speed up calculations with a GPU'},
				'gpu_nthreads':		{'format': '%d', 'comment': 'The number of threads per block to execute on the GPU'},
				'pad_length':		{'format': '%d', 'comment': 'Padding region (in pixels) around model to allow room for interpolation/convolution'},
				'add_noise':		{'format': '%.4f', 'comment': 'Noise to add to image'},
				'add_background':	{'format': '%.4f', 'comment': 'Background to add to image'},
				'model_size':		{'format': '%d', 'comment': 'Image size to generate for sersic models in multiples of sersic re'},
				'min_size':		{'format': '%d', 'comment': 'Minimum model size (square)'},
				'max_size':		{'format': '%d', 'comment': 'Maximum model size (square).  Will always be at least the psf size'},
				'x_offset':		{'format': '%.4f', 'comment': 'Offset models by this in the x-direction (pixels)'},
				'y_offset':		{'format': '%.4f', 'comment': 'Offset models by this in the y-direction (pixels)'} }

		output_order = ['sim_dir', 'individual_dirs', 'catalog_output', 'catalog_suffix', 'image_output', 'image_suffix', 'region_output', 'region_suffix', 'output_format', 'output_models', 'model_dir', 'clobber']
		output = {	'sim_dir':		{'format': '%s', 'comment': 'Name of working directory for simulated image output'},
				'individual_dirs':	{'format': '%s', 'comment': 'Whether or not to place each simulated image in its own directory (True/False)'},
				'catalog_output':	{'format': '%s', 'comment': 'Filename for output catalog detailing simulated objects'},
				'catalog_suffix':	{'format': '%s', 'comment': 'Suffix for determining output catalog name, if CATALOG_OUTPUT is not specified'},
				'image_output':		{'format': '%s', 'comment': 'Filename for simulated image'},
				'image_suffix':		{'format': '%s', 'comment': 'Suffix for determining simulated image name, if IMAGE_OUTPUT is not specified'},
				'rms_output':		{'format': '%s', 'comment': 'Filename for simulated rms image'},
				'rms_suffix':		{'format': '%s', 'comment': 'Suffix for determining simulated rms image name, if RMS_OUTPUT is not specified'},
				'output_format':	{'format': '%s', 'comment': 'Output format for image catalog.  fits or ascii'},
				'region_output':	{'format': '%s', 'comment': 'Filename for region file showing simulated object positions.'},
				'region_suffix':	{'format': '%s', 'comment': 'Suffix for determining region file name, if REGION_OUTPUT is not specified'},
				'output_models':	{'format': '%s', 'comment': 'Whether or not to output fits images for all generated models (True/False)'},
				'model_dir':		{'format': '%s', 'comment': 'Directory which models will be outputted to (subdir of SIM_DIR)'},
				'clobber':		{'format': '%s', 'comment': 'If False a new simulation will not be generated if the output image already exists.'} }

		sampling_order = ['include_errors', 'random_positions', 'position_border', 'as_is', 'random_mags', 'mix_parameters', 'number_frames', 'gals_per_frame']
		sampling = {	'include_errors':	{'format': '%s', 'comment': 'If True, then errors are added to output catalog parameters'},
				'random_positions':	{'format': '%s', 'comment': 'If True, then ra/dec are ignored and positions are randomized'},
				'position_border':	{'format': '%.8f', 'comment': 'Specify a rectangle in the WCS in which random objects must be found: min ra, max ra, min dec, max dec'},
				'as_is':		{'format': '%s', 'comment': 'Use the catalog as is'},
				'random_mags':		{'format': '%s', 'comment': 'Randomly generate magnitudes but match the input catalog LF'},
				'mix_parameters':	{'format': '%s', 'comment': 'Create new galaxies by randomly selecting from catalog parameter distributions'},
				'number_frames':	{'format': '%d', 'comment': 'The number of simulated images to generate (only applies for AS_IS=False)'},
				'gals_per_frame':	{'format': '%d', 'comment': 'The number of galaxies per simulated image (only applies for AS_IS=False)'} }

		# put them all in a list
		groups = [ image, catalog, format, model, output, sampling ]
		orders = [ image_order, catalog_order, format_order, model_order, output_order, sampling_order ]
		# also label them
		labels = [ 'Image', 'Catalog', 'Catalog Format', 'Modeling', 'Output', 'Catalog Sampling' ]

		if config is None: config = self.config

		# okay, now we can write out the config file
		utils.write_config( filename, config, groups, labels, orders )

##############################################################
## slop class for generating images with aritifical objects ##
##############################################################
class slop(object):
	""" Simulating Lots of Profiles:  A class for adding artificial galaxies to images. """

	# configuration
	config = {}

	# image stuff
	img_out = ''			# output filename
	rms_out = ''			# output rms filename
	img = np.array( [] )		# image
	hdr = np.array( [] )		# image header
	psf = np.array( [] )		# psf image
	rms = np.array( [] )		# RMS image
	has_rms = False			# was an RMS image loaded?
	zeropoint = 0.0			# zeropoint
	pad = 25			# padding length

	# catalog
	catalog = {}			# catalog of objects to generate
	ncat = 0			# number of objects in catalog

	# models
	models = []			# list of model objects
	models_created = False		# whether or not the models have been created
	model = np.array( [] )		# model image
	is_modeled = False		# whether or not the full model image has been created
	rms_model = np.array( [] )	# rms model image
	rms_is_modeled = False		# whether or not the rms model image has been created

	##########
	## init ##
	##########
	def __init__( self, img, psf, config, hdr=None, mask=None, mask_value=False, rms_img=None ):
		""" slop( img, psf, magzero, pad_length, hdr=None, mask=None, mask_value=False )
		
		Returns an object for generating artifical galaxy images and adding them to fits files.
		image and psf can be a filename or numpy arrays.  img_out is the output file name.
		If image is a numpy array then you must also pass a fits header (loaded with pyfits) to hdr.
		Pass the magnitude zeropoint and padding length for model generation.
		Also you can pass a boolean True/False mask to let slop know where the good pixels are.
		If you don't, it will generate one itself.  In this case you can just tell it what to use as a bad pixel
		value with mask_value (if any).
		"""

		# load image
		if type( img ) == type( '' ):
			if os.path.isfile( img ):
				( self.img, self.hdr ) = utils.load_image( img, header=True )
			else:
				raise ValueError( 'Slop error: Cannot find image file: %s' % img )
		elif type( img ) == type( np.array( [] ) ):
			self.img = img.copy()
			if hdr is None:
				raise ValueError( 'Slop error: Must pass a pyfits header when the image is a numpy array' )
			else:
				self.hdr = hdr
		else:
			raise ValueError( 'Slop error: unrecognized image!' )

		# load configuration
		self.config = {}

		# RMS image?
		if rms_img is not None: self.set_rms_image( rms_img )

		# set mask
		if mask is not None:
			self.mask = mask.copy()
		else:
			self.mask = get_mask( self.img, mask_value )

		# load wcs
		self.wcs = utils.get_wcs_info( self.hdr )

		# load psf
		self.psf = utils.load_psf( psf )
		self.psf_shape = self.psf.shape

		self.zeropoint = float( config['magzero'] )
		self.pad = int( float( config['pad_length'] ) )

		# configuration defaults
		self.model_size = config['model_size'] if 'model_size' in config else 10
		self.x_offset = config['x_offset'] if 'x_offset' in config else 0
		self.y_offset = config['y_offset'] if 'y_offset' in config else 0
		self.gpu = config['gpu'] if 'gpu' in config else False
		self.use_integration = config['use_integration'] if 'use_integration' in config else True

	###################
	## set rms image ##
	###################
	def set_rms_image( self, img ):
		""" slop.set_rms_image( image )
		
		Pass a numpy array or filename.  Must be the real RMS image which matches pixel for pixel the science image """

		if type( img ) == type( '' ):
			if os.path.isfile( img ):
				rms = utils.load_image( img )
			else:
				raise ValueError( 'Slop error: Cannot find rms image file: %s' % img )
		elif type( img ) == type( np.array( [] ) ):
			rms = img.copy()
		else:
			raise ValueError( 'Slop error: unrecognized rms image %s!' % img )

		if rms.shape != self.img.shape: raise ValueError( 'Slop error: the science image and rms image are different sizes!' )

		self.rms = rms
		self.has_rms = True

	################
	## set output ##
	################
	def set_output( self, filename, rms_filename=None ):
		""" slop.set_output( filename, rms_filename=None )
		
		Set the ouput filename """

		self.img_out = filename
		if rms_filename is not None: self.rms_out = rms_filename

	###################
	## create models ##
	###################
	def create_models( self, catalog, min_size=None, max_size=None ):
		""" slop.set_models( catalog, min_size=None, max_size=None )
		
		Set the catalog of things to generate.  Expects a dictionary of numpy arrays:
		cat.keys() == ['id','model','ra','dec','mag','re','re_pix','n','ba','pa']
		
		Set the image size for sersic models in factors of re.
		If desired, set minimum or maximum model size. """

		if type( catalog ) != type( {} ): raise ValueError( 'Slop error: expected dictionary for catalog format!' )

		required = ['id','model','ra','dec','mag','re','re_pix','n','ba','pa']
		for field in required:
			if not catalog.has_key( field ): raise ValueError( 'Slop error: could not find field %s' % field )

		# make sure max_size is always at least as large as the psf
		if max_size is not None:
			if max_size < self.psf_shape[0]: max_size = self.psf_shape[0]

		self.catalog_list = catalog
		self.ncat = self.catalog_list['id'].size
		self.models = []
		self.is_modeled = False

		# for creating models we need each model to have a dictionary with its parameters.
		for i in range( self.ncat ):
			this = {}

			# copy to dictionary
			for key in self.catalog_list.keys(): this[key] = self.catalog_list[key][i]

			# Calculated desired size/shape for this model.
			if this['model'] == 'point':
				size = self.psf_shape[0]
			else:
				size = int( np.ceil( this['re_pix']*self.model_size ) )
			# Make it odd.
			if not size % 2: size += 1
			if min_size is not None and size < min_size: size = min_size
			if max_size is not None and size > max_size: size = max_size
			shape = ( size, size )

			# calculate ra/dec
			( this['x'], this['y'] ) = utils.rd2xy( self.wcs, this['ra'], this['dec'] )
			this['x'] += self.x_offset
			this['y'] += self.y_offset

			# convert x/y to index
			ind_x = ( np.round( this['x'] ) - 1 ).astype( 'int' )
			ind_y = ( np.round( this['y'] ) - 1 ).astype( 'int' )

			# min/max x/y indexes into model
			( mminx, mmaxx, mminy, mmaxy ) = ( 0, shape[0], 0, shape[1] )
			# calculate min/max x/y indexes to place model in image
			size = (shape[0]-1)/2
			miny = ind_y - size
			maxy = ind_y + size + 1
			minx = ind_x - size
			maxx = ind_x + size + 1

			# make sure we haven't gone off any borders
			if miny < 0:
				mminy -= miny
				miny = 0
			if minx < 0:
				mminx -= minx
				minx = 0
			if maxy > self.img.shape[0]:
				mmaxy -= (self.img.shape[0] - maxy)
				maxy = self.img.shape[0]
			if maxx > self.img.shape[1]:
				mmaxx -= (self.img.shape[1] - maxx)
				maxx = self.img.shape[1]

			# need a few more properties for modeling.
			# this first one is supposed to be where to center the object in cutout coordinates
			this['img_x'] = size + this['x'] - ( np.round( this['x'] ) - 1 ).astype( 'int' ) #this['x'] - minx
			this['img_y'] = size + this['x'] - ( np.round( this['x'] ) - 1 ).astype( 'int' ) #this['y'] - miny
			# and now the center as x/y indexes - where they fall in the actual array
			this['ind_x'] = size #( np.round( this['img_x'] ) - 1 ).astype( 'int' )
			this['ind_y'] = size #( np.round( this['img_y'] ) - 1 ).astype( 'int' )
			# not used for generating models but required by model objects
			this['lim_x'] = {}
			this['lim_y'] = {}

			# create model object
			model = models.get_model( this, self.zeropoint, gpu=self.gpu )

			# generate model
			model.generate_model( self.pad, shape, self.psf, use_integration=self.use_integration )

			# store image borders in model object for quick retrieval later
			model.slop_image_inds = ( minx, maxx, miny, maxy )
			model.slop_model_inds = ( mminx, mmaxx, mminy, mmaxy )

			# and store
			self.models.append( model )

		self.models_created = True

	####################
	## get full model ##
	####################
	def get_full_model( self, fileout=None ):
		""" slop.get_full_model( fileout=None )
		
		Write out generated models to image and return.
		Also write to file `fileout` if set. """

		if self.is_modeled:
			# write to the already specified output file if it has been set and something else wasn't specified
			if fileout is None and self.img_out: fileout = self.img_out
			# write to file if requested!
			if fileout is not None: pyfits.writeto( fileout, img, self.hdr, clobber=True )
			return self.model.copy()

		img = self.img.copy()
		for model_obj in self.models:

			# get x/y image indexes
			( minx, maxx, miny, maxy ) = model_obj.slop_image_inds
			( mminx, mmaxx, mminy, mmaxy ) = model_obj.slop_model_inds

			# only copy those parts that are good data pixels and which fall on the image
			w = np.where( self.mask[miny:maxy,minx:maxx] )
			img[miny:maxy,minx:maxx][w] += model_obj.model_img[mminy:mmaxy,mminx:mmaxx][w]

		# save in object
		self.model = img
		self.is_modeled = True

		# save to file?
		# write to the already specified output file if it has been set and something else wasn't specified
		if fileout is None and self.img_out: fileout = self.img_out
		# write to file if requested!
		if fileout is not None: pyfits.writeto( fileout, img, self.hdr, clobber=True )

		# okay, all done.  Return
		return img.copy()

	########################
	## generate rms image ##
	########################
	def get_full_rms_model( self, fileout=None ):
		""" slop.get_full_rms_model( fileout=None )

		Generate RMS map corresponding to models and return.
		Also write to file `fileout` if set.

		This requires that the image passed to slop is a real science image, and the RMS map is a real RMS map. """

		if not self.has_rms: raise ValueError( 'Cannot generate an RMS image - the real RMS image was not found!' )

		if self.rms_is_modeled: return self.rms_model.copy()

		# image to add rms to
		rms = self.rms.copy()

		# the way this works is by taking the pixel value in the model image
		# and finding the mean RMS of pixels with similar values in the actual image
		# first add sky to the model image (because it has none)
		# sky values and sky rms must be subtracted since those are already in the images

		# first get rms for every good sky pixel, which will be used for interpolation (and therefore must be sorted)
		pix_vals = self.img[self.mask].ravel()
		sinds = pix_vals.argsort()
		pix_vals = pix_vals[sinds]
		rms_vals = (self.rms[self.mask].ravel())[sinds]

		# now subtract the sky from the pixel values and rms values
		pix_vals -= np.median( pix_vals )
		rms_vals -= np.median( rms_vals )

		for model_obj in self.models:

			# find rms of pixels with similar pixel values as the model
			rms_model = np.interp( model_obj.model_img, pix_vals, rms_vals )

			# and add this to the RMS image normally

			# get x/y image indexes
			( minx, maxx, miny, maxy ) = model_obj.slop_image_inds
			( mminx, mmaxx, mminy, mmaxy ) = model_obj.slop_model_inds

			# only copy those parts that are good data pixels
			w = np.where( self.mask[miny:maxy,minx:maxx] )
			rms[miny:maxy,minx:maxx][w] += rms_model[mminy:mmaxy,mminx:mmaxx][w]

		# save in object
		self.rms_model = rms
		self.rms_is_modeled = True

		# save to file?
		# write to the already specified output file if it has been set and something else wasn't specified
		if fileout is None and self.rms_out: fileout = self.rms_out
		# write to file if requested!
		if fileout is not None: pyfits.writeto( fileout, rms, self.hdr, clobber=True )

		# okay, all done.  Return
		return rms.copy()

##############
## get mask ##
##############
def get_mask( img, mask_value=False ):
	""" pygsti.get_mask( image, mask_value=False )
	
	Generate a mask image (True/False) for useable pixels.
	Pass the image (np array) and, if desired, a bad pixel value """

	mask = np.ones( img.shape, dtype=np.bool )

	# find bad values on the image
	m = np.isnan( img ) | np.isinf( img )

	# use mask value if specified.  Any boolean value is an unacceptable mask value
	if type( mask_value ) != type( False ):
		m = m | ( img == mask_value )

	mask[m] = False

	return mask

def usage():
	print ""
	print "pygsti.py [--help --config config_file]"
	print ""
	print "all command line flags must come first"
	print "run pygsti with the parameters specified in config_file"
	print "config_file is optional and defaults to pygsti.config"
	print "set --config flag to dump a default configuration to stdout"
	print ""

if __name__ == '__main__':
	import sys,getopt

	# check arguments
	try:
		(opts, args) = getopt.getopt( sys.argv[1:], 'hc', ['help','config'] )
	except getopt.GetoptError, err:
		print str(err)
		usage()
		sys.exit(2)

	# check flags
	output_config = False
	for (o, a) in opts:
		if o == '-h' or o == '--help':
			usage()
			sys.exit(2)
		if o == '-c' or o == '--config':
			sti = pygsti( config=True )
			sti.output_config()
			sys.exit(2)

	# default files
	if len( args ) > 0:
		config = args[0]
	else:
		config = 'pygsti.config'
		if not os.path.isfile( config ):
			usage()
			sys.exit(2)

	pygsti = pygsti( config=config )
	pygsti.go()