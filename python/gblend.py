""" Galaxy BLEND is the workhorse for pygfit - it does the actual fitting.

Initializing a gblend object is a multi step process.  It involves storing the object number from the source extractor pass, the maximum an object can shift by, and various configuration parameters (passed along at init), giving it the necessary images (image, rms image, and segmenetation image) through set_images(), and then passing along a high resolution catalog for it to extract matching objects from with set_hres_catalog()
"""

import numpy as np
import os,pyfits,log,utils,models,traceback
import scipy.optimize
import log

class gblend(object):
	""" Galaxy BLEND is the workhorse for pygfit - it does the actual fitting.

	Initializing a gblend object is a multi step process.  It involves storing the object number from the source extractor pass, the maximum an object can shift by, and various configuration parameters (passed along at init), giving it the necessary images (image, rms image, and segmenetation image) through set_images(), and then passing along a high resolution catalog for it to extract matching objects from with set_hres_catalog()

	gblend_obj = gblend( number, max_shift_arcsecs, config )

	Initializes a gblend object.
	number: the object number from source extractor
	config: configuration dictionary from pygfit object
	"""

	# configuration stuff
	config = {}
	config_keys = ['image_dir', 'rms_dir', 'segmentation_dir', 'segmentation_mask_dir', 'model_dir', 'hres_rotangle', 'lres_magzero', 'pad_length', 'output_all_models', 'max_array_size', 'fit_sky', 'gpu', 'use_integration']
	config_defaults = {	'image_dir':			'cutouts/img/',
				'rms_dir':			'cutouts/rms/',
				'segmentation_dir':		'cutouts/seg/',
				'segmentation_mask_dir':	'cutouts/mask/',
				'model_dir':			'models/',
				'pad_length':			25,
				'output_all_models':		False }

	# fitting details
	info = {}			# object info (from source extractor)
	number = 0			# object number (from source extractor)
	max_shift_arcsecs = 0		# maximum allowed shift for hres objects (arcseconds)
	max_shift = 0			# maximum allowed shift for hres objects (pixels)
	max_shift_high = 0		# int( np.ceil( max_shift ) )
	xmin = 0			# minx for cutout
	xmax = 0			# maxx for cutout
	ymin = 0			# miny for cutout
	ymax = 0			# maxy for cutout
	pad = 0				# padding length (copied out of config['pad_length'] by set_images()
	nx = 0				# image width
	ny = 0				# image height

	# images
	img = np.array( [] )		# cutout of low resolution image
	img_padded = np.array( [] )	# padded image array
	rms = np.array( [] )		# cutout of low resolution rms image
	seg = np.array( [] )		# cutout of segmentation image
	psf = np.array( [] )		# psf image
	psf_nx = 0			# width of psf image
	psf_ny = 0			# height of psf image
	img_hdr = ''			# img header
	seg_m = np.array( [] )		# segmentation mask: (self.seg == self.number)
	fit_m = np.array( [] )		# mask for fitting region: (self.seg == self.number) | (self.seg == 0)
	fit_mask = np.array( [] )	# image of fitting region: 1 where fitting is done, 0 otherwise
	x = 0				# x locations for interpolation/etc (for use with self.img_padded)
	y = 0				# y locations for interpolation/etc (for use with self.img_padded)s
	wcs = {}			# wcs dictionary

	# high resolution catalog
	nhres = 0			# number of matching objects from high resolution catalog
	hres_models = []		# list of model objects for high resolution catalog
	hres_cat = {}			# dictionary with list of values for objects in high resolution catalog
	current_model = -1		# track current object for model iterator

	# object state
	images_loaded = False		# whether or not the low resolution images have been loaded: self.set_images()
	wcs_loaded = False		# whether or not the wcs has been loaded: self.set_images()
	hres_loaded = False		# whether or not the high resolution catalog has been loaded: self.set_hres_catalog()
	ready = False			# whether or not the object is ready for fitting: self.generate_models()
	is_fit = False			# whether or not the fitting has been done: self.fit()
	fit_modeled = False		# whether or not the best fitting model has been stored in the object
	full_fit_modeled = False	# whether or not the full best fitting model has been stored in the object

	# fit details
	fit_code = 0			# the result code returned by scipy.optimize.leastsq()
	best_fit = []			# the best fit parameters returned by scipy.optimize.leastsq()
	sky = 0				# fitted sky
	model = np.array( [] )		# the best fitting model
	full_model = np.array( [] )	# the full best fitting model - covers the entire cutout region
	chisq = 0.0			# chisq for the best fit
	nf = 0				# number of free parameters for the best fit

	##########
	## init ##
	##########
	def __init__( self, info, max_shift_arcsecs, config ):
		""" gblend_obj = gblend( info, max_shift_arcsecs, config )
		
		Initializes a gblend object.
		info: the info for this object from the lres catalog
		config: configuration dictionary from pygfit object
		"""

		self.info = info
		self.number = info['number']
		self.max_shift_arcsecs = max_shift_arcsecs
		self.set_config( config );

		if not self.number: raise ValueError( 'gblend object must be initialized with source extractor number!' )
		if not self.max_shift_arcsecs: raise ValueError( 'gblend object must be initialized with a maximum allowed position shift!' )

	# return iterator for looping through models
	def __iter__(self):
		self.current_model = -1
		return self

	# return next model in iterator
	def next(self):
		self.current_model += 1
		if self.current_model == self.nhres: raise StopIteration

		return self.hres_models[self.current_model]

	################
	## set config ##
	################
	def set_config( self, config={} ):
		""" gblend_obj.set_config() """

		self.config = {}

		# was anything passed?
		if len( config.keys() ) == 0: raise ValueError( 'You must pass along the pygfit configuration parameters when initializing a gblend object!' )

		# look for the keys specified in the object
		for key in self.config_keys:
			if config.has_key( key ): self.config[key] = config[key]

		# finally some defaults
		for key in self.config_defaults.keys(): self.config.setdefault( key, self.config_defaults[key] )

		# make sure directories end in a slash
		if self.config['image_dir'][-1] != os.sep: self.config['image_dir'] += os.sep
		if self.config['rms_dir'][-1] != os.sep: self.config['rms_dir'] += os.sep
		if self.config['segmentation_dir'][-1] != os.sep: self.config['segmentation_dir'] += os.sep
		if self.config['segmentation_mask_dir'][-1] != os.sep: self.config['segmentation_mask_dir'] += os.sep
		if self.config['model_dir'][-1] != os.sep: self.config['model_dir'] += os.sep

	################
	## set images ##
	################
	def set_images( self, img, img_hdr, seg, rms, psf, wcs={} ):
		""" gblend.set_images( lres_img, lres_img_hdr, lres_seg_img, lres_rms_img, lres_psf_img, wcs={} )
		
		All images should be numpy arrays.  All headers should be pyfits header objects.
		Optionally, pass along a wcs dictionary generated by utils.get_wcs_info()
		"""

		# store wcs info.  Calculate if not passed.
		if ( type( wcs ) != type( {} ) or len( wcs.keys() ) == 0 ):
			self.wcs = utils.get_wcs_info( img_hdr )
		else:
			self.wcs = wcs.copy()
		
		if len( self.wcs.keys() ) == 0: raise ValueError( 'Could not load wcs info for gblend object!' )
		self.wcs_loaded = True

		# convert the max shift to pixels
		self.max_shift = self.max_shift_arcsecs/self.wcs['scale']
		self.max_shift_high = int( np.ceil( self.max_shift ) )

		# source extractor often has a bug whereby it will give pixels the same
		# value in the segmentation image even when they are on opposite sides of the image
		# and aren't at all connected.  This dramatically hurts execution time
		# as the cutout (and therefore model) size is determined by the extent of the object
		# in the segmentation image.  seg is fixed in place to avoid excessive copying.
		utils.fix_extractor_seg_bug( seg, self.number )

		# find this object in the segmentation image
		m = (seg == self.number)

		# make sure the object was actually found...
		if not m.sum(): raise ValueError( 'The object %d was not found in the low resolution segmentation image!' % self.number )

		# now extract min/max x/y for this object from the segmentation image
		w = np.where( m )
		(xmin,xmax,ymin,ymax) = (w[1].min(),w[1].max(),w[0].min(),w[0].max())

		# try it Anthony's way...
		xmin=(w[1]).min()-int(self.max_shift)-2
		ymin=(w[0]).min()-int(self.max_shift)-2
		dx=(w[1]).max()+1+int(self.max_shift)-xmin+4
		dy=(w[0]).max()+1+int(self.max_shift)-ymin+4
		(MAX_Y,MAX_X) = img.shape

		xmin0=xmin
		#IMAGES MUST BE SQUARE -- modification below assures this, but yields asymmetry about object
		if(dx<dy):
			xmin=xmin-int((dy-dx)/2.)
			dx=dy
		elif(dy<dx):
			ymin=ymin-int((dx-dy)/2.)
			dy=dx
		if(xmin<0): xmin=0
		if(ymin<0): ymin=0
		xmax=xmin+dx
		ymax=ymin+dx
		#At boundaries squaring can fail, so shift box to abut edge.
		if(xmax>MAX_X):
			dfix=xmax-MAX_X
			xmax=MAX_X
			xmin=xmin-dfix
		if(ymax>MAX_Y):
			dfix=ymax-MAX_Y
			ymax=MAX_Y
			ymin=ymin-dfix

		# now store min/max and image cutouts
		(self.xmin,self.xmax,self.ymin,self.ymax) = (xmin,xmax,ymin,ymax)
		self.seg = seg[ymin:ymax,xmin:xmax].copy()
		self.seg_m = (self.seg == self.number)
		self.fit_m = (self.seg == 0) | (self.seg == self.number)
		self.fit_mask = np.where( self.fit_m, 1, 0 )
		self.rms = rms[ymin:ymax,xmin:xmax].copy()
		self.psf = psf.copy()
		( self.psf_ny, self.psf_nx ) = self.psf.shape
		( self.ny, self.nx ) = (ymax-ymin,xmax-xmin)

		# store image header
		self.img_hdr = img_hdr.copy()
		self.img_hdr['crpix1'] -= xmin
		self.img_hdr['crpix2'] -= ymin

		# and update the wcs info
		self.wcs['x0'] = self.img_hdr['crpix1']
		self.wcs['y0'] = self.img_hdr['crpix2']

		# update the padding length
		if self.config['pad_length'] < len( self.psf )/2.0 + 1: self.config['pad_length'] = len( self.psf )/2+1
		# copy to something a bit more convenient
		self.pad = self.config['pad_length']

		# update the object state
		self.images_loaded = True

		# create the images for fitting
		self.generate_fitting_images( img )

		# output cutouts
		self.generate_cutouts()

	############################
	## generate fitting image ##
	############################
	def generate_fitting_images( self, image ):
		""" gblend.generate_fitting_image( image )
		
		Generate and store the images needed for fitting from the given image.
		"""

		if not self.images_loaded: raise ValueError( 'You must load images before generating the padded image!' )

		try:
			self.img = image[self.ymin:self.ymax,self.xmin:self.xmax].copy()
			self.img_padded = np.zeros( (self.ny+2*self.pad,self.ny+2*self.pad) )
			self.img_padded[self.pad:self.ny+self.pad,self.pad:self.nx+self.pad] = self.img
		except KeyboardInterrupt:
			raise
		except:
			# on exception return the traceback string.
			return traceback.format_exc()

		return True

	#################################
	## set high resolution catalog ##
	#################################
	def set_hres_catalog( self, hres_cat ):
		""" nhres = gblend.set_hres_catalog( pygfit.hres_catalog ) """

		if not self.images_loaded: raise ValueError( 'You must load images before matching the high resolution catalog!' )

		# convert high resolution x & y to image coordinates
		hres_cat['img_x'] = hres_cat['x'] - self.xmin
		hres_cat['img_y'] = hres_cat['y'] - self.ymin
		# calculate x/y limits
		hres_cat['lim_x'] = {'min': hres_cat['img_x']-self.max_shift, 'max': hres_cat['img_x']+self.max_shift}
		hres_cat['lim_y'] = {'min': hres_cat['img_y']-self.max_shift, 'max': hres_cat['img_y']+self.max_shift}
		# and the x/y indexes - where they fall in the actual array
		hres_cat['ind_x'] = ( np.round( hres_cat['img_x'] ) - 1 ).astype( 'int' )
		hres_cat['ind_y'] = ( np.round( hres_cat['img_y'] ) - 1 ).astype( 'int' )

		# first filter out anything that falls outside the image
		w = (np.where( (hres_cat['ind_x'] < self.nx) & (hres_cat['ind_x'] >= 0) & (hres_cat['ind_y'] < self.ny) & (hres_cat['ind_y'] >= 0) ))[0]
		# we're done if nothing is on the image
		if len( w ) == 0:
			self.hres_loaded = True
			return 0

		# which high resolution objects fall in the low resolution object itself?
		wmatch = w[ np.where( self.seg[hres_cat['ind_y'][w],hres_cat['ind_x'][w]] == self.number ) ]
		self.nhres = len( wmatch )

		# again, we're done if nothing was found
		if self.nhres == 0:
			self.hres_loaded = True
			return 0

		# a little inconvenient - create an individual dictionary for each object to pass to the model generator
		# also keep the trimmed down hres_cat dictionary for convenience
		if not os.path.isdir( self.config['model_dir'] ): os.makedirs( self.config['model_dir'], mode=0755 )
		hres_models = []
		new_cat = {}
		for i in range( self.nhres ):

			this = {}
			for key in hres_cat.keys():

				# for copying catalog, most keys are numpy arrays.  Those are easy to copy
				if type( hres_cat[key] ) == type( np.array([]) ):

					# create trimmed down hres_cat.
					if i == 0: new_cat[key] = hres_cat[key][wmatch]

					# build dictionary for this particular object
					this[key] = hres_cat[key][wmatch[i]]

				# the only non-arrays are the limit keys.  Copy those manually
				else:
					# copy for trimmed down hres_cat
					if i == 0: new_cat[key] = {'min': hres_cat[key]['min'][wmatch], 'max': hres_cat[key]['max'][wmatch]}

					# copy for this objects dictionary
					this[key] = {'min': hres_cat[key]['min'][wmatch[i]], 'max': hres_cat[key]['max'][wmatch][i]}

			# generate model object and store in object and store
			# models.get_model will return the appropriate model object (point, sersic)
			hres_models.append( models.get_model( this, self.config['lres_magzero'], max_array_size=self.config['max_array_size'], gpu=self.config['gpu'] ) )

		self.hres_cat = new_cat
		self.hres_models = hres_models

		# finally output a region file for the high resolution objects
		if not os.path.isdir( self.config['image_dir'] ): os.makedirs( self.config['image_dir'], mode=0755 )
		utils.write_region( '%s%d_hres.reg' % (self.config['image_dir'],self.number), self.hres_cat['ra'], self.hres_cat['dec'], self.hres_cat['re']/3600, titles=self.hres_cat['id'] )

		# that's really all there is to it
		self.hres_loaded = True

		return self.nhres

	#####################
	## generate models ##
	#####################
	def generate_models( self, queue=None, output_fits=True ):
		""" gblend.generate_models( queue=None, output_fits=True )
		
		Generate all the high resolution models used in the fitting process.
		The high resolution catalog must be set before models can be generated.
		If output_fits = True then a fits file will be generated with the inital models. This will be created in the model_dir.
		queue is used to return models when run in parallel mode. """

		if not self.hres_loaded: raise ValueError( 'You must set the high resolution catalog with gblend.set_hres_catalog() before generating models!' )

		imgs = []

		# loop through high resolution models
		for (i,model_obj) in enumerate(self):

			# catch any and all errors
			try:

				# generate models
				model_obj.generate_model( self.pad, (self.ny,self.nx), self.psf.copy(), use_integration=self.config['use_integration'] )
	
				# the rest is for outputting fits files of input models - skip if output_fits==False
				if not output_fits: continue
	
				# generate extension for outputting to fits
				imgs.append( pyfits.ImageHDU( model_obj.model_img.copy() ) )
	
				# add to complete model
				if i == 0:
					full_model = model_obj.model_img.copy()
				else:
					full_model += model_obj.model_img.copy()

			except KeyboardInterrupt:
				raise
			except:
				# on exception return the traceback string.  Also return the traceback string to the queue if was passed
				trace = traceback.format_exc()
				if queue is not None: queue.put( (self.number, trace) )
				return trace

		# output a fits file for the full model and individual models
		if output_fits:
			# prepare hdulist
			hdus = [pyfits.PrimaryHDU( full_model )]
			hdus.extend( imgs )
			hdulist = pyfits.HDUList( hdus )

			# write out file - delete first
			file = '%s%d_input.fits' % (self.config['model_dir'],self.number)
			if os.path.isfile( file ): os.remove( file )
			hdulist.writeto( file )

		# if queue is not None, then it is an ouput queue used to return the models to pygfit when running in parallel mode
		# that means I have to fetch the model from every model object and return them through the queue.
		# whatever I set in the queue will be passed back through gblend.set_models()
		if queue is not None:

			models = []
			for model_obj in self: models.append( model_obj.model_img )

			# and send it off!
			queue.put( (self.number, models) )

		# now we are ready for fitting!
		self.ready = True
		return True

	################
	## set models ##
	################
	def set_models( self, models ):
		""" gblend.set_models()
		
		receives a list of models (numpy arrays) and sets those to be the models for the model objects """

		if len( models ) != self.nhres: raise ValueError( 'The number of models passed to gblend.set_models() did not match the number of model objects!' )

		for (i,model_obj) in enumerate(self): model_obj.set_model( models[i] )
		self.ready = True

	######################
	## get dependencies ##
	######################
	def get_dependencies( self, ids, xs, ys, radii ):
		""" ( id, x, y, radius, dependencies ) = gblend.get_dependencies( ids, xs, ys, radii )
		
		Return information for calculating dependencies, as well as a list of objects which this is dependent on.
		This is used when running in parallel mode to make sure things aren't fit before bright neighbors are subtracted.
		"""

		# if anything in the catalog overlaps with this gblend then it is a dependent.

		# first calculate the center of this gblend
		coords = np.mgrid[0:self.nx,0:self.ny]
		yinds = coords[0]
		xinds = coords[1]
		x_cent = np.mean( xinds[self.fit_m] )
		y_cent = np.mean( yinds[self.fit_m] )

		# now how far away is the farthest pixel?
		radius = np.sqrt( (yinds[self.fit_m]-y_cent)**2.0 + (xinds[self.fit_m]-x_cent)**2.0 ).max()

		# now convert x_cent and y_cent from cutout coordinates to (roughly) image coordinates
		x_cent += self.xmin
		y_cent += self.ymin

		# figure out what this gblend depends on
		dependencies = []

		# if there is nothing in the catalog there can't be any dependencies!
		if len( ids ) == 0: return ( self.number, x_cent, y_cent, radius, dependencies )

		# calculate distance to everything that was passed
		dist = np.sqrt( (np.array( ys )-y_cent)**2.0 + (np.array( xs )-x_cent)**2.0 )

		# find any objects that overlap in radius
		w = np.where( dist < radius+np.array( radii ) )[0]

		# any matches are dependents
		if len( w ) > 0: dependencies = np.array( ids )[w].tolist()

		# all done!
		return ( self.number, x_cent, y_cent, radius, dependencies )

	##########
	## fit! ##
	##########
	def fit( self, queue=None ):
		""" gblend.fit()
		
		Perform the actual fitting.  Makes use of scipy.optimize.leastsq()
		"""

		# make sure all the necessary data has been loaded
		if not self.ready:
			if not self.generate_models(): raise ValueError( 'Gblend object not ready for fitting!' )

		try:

			# first loop through the models and retrieve the starting guesses
			guess = []
			for model in self: guess.extend( model.get_first_guess() )

			# append a starting guess for the sky if it is being fit
			if self.config['fit_sky']: guess.append( 0 )

			# now fit!
			res = scipy.optimize.leastsq( self.residuals, guess, ftol=1.e-13, xtol=1.e-13, full_output=1, epsfcn=1.e-1 )
			#res = scipy.optimize.fmin( self.residuals_alt, guess, ftol=1.e-13, xtol=1.e-13, full_output=1, disp=False )
			#res = scipy.optimize.fmin_powell( self.residuals_alt, guess, ftol=1.e-13, xtol=1.e-13, full_output=1, disp=False )

			# update object status / store result in object
			self.set_best_fit( res[-1], res[0] )

			# if self.fit() has been passed a queue object then this is running in parallel mode, and results must be returned through the queue
			if queue is not None: queue.put( (self.number, self.fit_code, self.best_fit) )

		except KeyboardInterrupt:
			raise
		except:
			# on exception return the traceback string.  Also return the traceback string to the queue if was passed
			trace = traceback.format_exc()
			if queue is not None: queue.put( (self.number, trace, False) )
			return ( trace, False )

		# return fit code and fit
		return ( self.fit_code, self.best_fit )

	##################
	## set best fit ##
	##################
	def set_best_fit( self, fit_code, fit ):
		""" gblend.set_best_fit( fit_code, fit )
		
		Stores the given result as the best fit.
		fit_code is the code returned by scipy.optimize.leastsq
		fit is the params array. """

		self.is_fit = True
		self.fit_modeled = False
		self.fit_code = fit_code
		self.best_fit = fit
		if self.config['fit_sky']: self.sky = self.best_fit[-1]

	#########################
	## calculate residuals ##
	#########################
	def residuals( self, params, absolute=False ):
		""" gblend.residuals( params, absolute=False )
		
		Given parameter list of high resolution model parameters, return residuals for low resolution image.
		Typically called by scipy.optimize.leastsq
		If absolute=True then absolute residuals will be returned - not divided by rms. """

		# fetch the full model for these parameters
		model = self.get_model( params )

		# residuals = ( img - model )
		residuals = ( self.img[self.fit_m] - model[self.fit_m] )

		# if not absolute residuals, divide by error
		if not absolute: residuals /= self.rms[self.fit_m]

		# check for infs and nans, replace with zero (so they don't mess things up)
		m = ~(np.isnan( residuals ) | np.isinf( residuals ))

		# and return
		return residuals[m]

	#########################
	## calculate residuals ##
	#########################
	def residuals_old( self, params, absolute=False ):
		""" gblend.residuals( params, absolute=False )
		
		Given parameter list of high resolution model parameters, return residuals for low resolution image.
		Typically called by scipy.optimize.leastsq
		If absolute=True then absolute residuals will be returned - not divided by rms. """

		# fetch the full model for these parameters
		model = self.get_model( params )

		# residuals = ( img - model )
		residuals = ( self.img[self.fit_m] - model[self.fit_m] )

		# if not absolute residuals, divide by error
		if not absolute: residuals /= self.rms[self.fit_m]

		# check for infs and nans, replace with zero (so they don't mess things up)
		m = np.isnan( residuals ) | np.isinf( residuals )
		if m.any(): residuals[m] = 0

		# and return
		return residuals

	###################
	## residuals alt ##
	###################
	def residuals_alt( self, params ):
		""" gblend.residuals_alt( params ) """

		return (self.residuals( params )**2.0).sum()

	###############
	## get model ##
	###############
	def get_model( self, params, full=False, sky=True ):
		""" gblend.get_model( params, full=False )
		
		Given parameter list of high resolution model parameters, return model for low resolution image.
		If full=True then it retuns the model is calculated over the full cutout region, not just the segmentation mask. """

		# parameter list to pass (will change with each pass)
		new_params = params

		# loop through the models and pass them the parameters list.
		for (i,model_obj) in enumerate(self):

			# fetch model corresponding to new parameters for each model object
			# with return_parameters=True it will return a new parameter list with its own parameters removed
			# this means that I can then just pass the new parameters list to the next model object
			( model, new_params ) = model_obj.model_parameters( new_params, return_parameters=True )

			# now build the full model
			if i == 0:
				if full:
					full_model = model.copy()
				else:
					full_model = np.zeros( model.shape )
					full_model[self.fit_m] = model[self.fit_m]
			else:
				if full:
					full_model += model
				else:
					full_model[self.fit_m] += model[self.fit_m]

		# there should be one parameter left over if sky fitting is on.  If so, add that in as the sky
		if self.config['fit_sky'] and len( new_params ) == 1 and sky:
			if full:
				full_model += new_params[0]
			else:
				full_model[self.fit_m] += new_params[0]

		return full_model

	##################
	## get best fit ##
	##################
	def get_best_fit( self, return_chisq=False, sky=True ):
		""" model = gblend.get_best_fit( return_chisq=False, sky=True )
		( model, chisq, nf ) = gblend.get_best_fit( return_chisq=True, sky=True )
		
		Returns the model, chisq, and number of free parameters for the best fit.
		Returned model has fitted sky added in if sky=True (and the sky was fit). """

		# make sure the fit has been found first
		if not self.is_fit: raise ValueError( 'You must fit the gblend before retrieving the best fit!' )

		# calculate the best fitting model if not calculated already
		if not self.fit_modeled:

			# fetch the best fit model
			model = self.get_model( self.best_fit )
	
			# calculate chisq: sum( ( (img-model)/err )**2.0 )
			chisq = np.sum( ( (self.img[self.fit_m] - model[self.fit_m]) / self.rms[self.fit_m] )**2.0 )
	
			# and the number of free parameters
			nf = self.fit_m.sum() - len( self.best_fit )

			# store these things in the object for quicker retrieval later
			self.fit_modeled = True
			self.model = model
			self.chisq = chisq
			self.nf = nf

		# the sky value is always included in the stored model.
		# so if sky is False then we have to subtract it off
		sky_subtract = 0 if sky else self.sky

		# return
		if return_chisq: return ( self.model - sky_subtract, self.chisq, self.nf )
		return self.model - sky_subtract


	#######################
	## get best fit full ##
	#######################
	def get_best_fit_full( self, sky=True ):
		""" model = gblend.get_best_fit_full( sky=True )
		
		Returns the full model, i.e. the model over the whole cutout region, not just the segmentation mask.
		Returned model has sky added in if sky=True """

		# make sure the fit has been found first
		if not self.is_fit: raise ValueError( 'You must fit the gblend before retrieving the best fit!' )

		# calculate the best fitting model if not calculated already
		if not self.full_fit_modeled:

			# fetch the best fit model
			model = self.get_model( self.best_fit, full=True )
	
			# store these things in the object for quicker retrieval later
			self.full_fit_modeled = True
			self.full_model = model

		# the sky value is always included in the stored model.
		# so if sky is False then we have to subtract it off
		sky_subtract = 0 if sky else self.sky

		# return
		return self.full_model.copy() - sky_subtract

	####################
	## subtract model ##
	####################
	def subtract_model( self, image, full=False, sky=False ):
		""" subtracted = gblend.subtract_model( image, full=False, sky=False )
		
		Subtract the model from the given image.
		This assumes that you have passed an image with the same shape as the original low resolution image.
		If full=False then it only subtracts over the segmentation mask, otherwise the whole cutout.
		If sky=True then the fitted sky will also be subtracted. """

		# fetch model image
		if full:
			fit = self.get_best_fit_full( sky=sky )
		else:
			fit = self.get_best_fit( sky=sky )

		# subtract
		image[self.ymin:self.ymax,self.xmin:self.xmax] -= fit

		# and return
		return image

	################
	## output fit ##
	################
	def output_fit( self ):
		""" gblend.output_fit()
		
		Outputs a .fits file with original image, model image, and residuals image """

		# prepare the various image extensions

		# original image
		img_hdu = pyfits.PrimaryHDU( self.img, self.img_hdr )

		# best fit model
		fit_hdu = pyfits.ImageHDU( self.get_best_fit_full(), self.img_hdr )

		# residuals
		res_hdu = pyfits.ImageHDU( self.img - self.get_best_fit_full(), self.img_hdr )

		# make list of hdus
		hdus = [img_hdu, fit_hdu, res_hdu]

		# if the configuration specified outputting all models, then do so
		if self.config['output_all_models']:

			# loop through model objects.  See self.get_model() for comments on how this works
			new_params = list( self.best_fit )
			for model_obj in self:

				# retrieve best fit model
				( model, new_params ) = model_obj.model_parameters( new_params, return_parameters=True )

				# create ImageHDU and append to list
				hdus.append( pyfits.ImageHDU( model, self.img_hdr ) )

		# generate pyfits hdu list
		hdulist = pyfits.HDUList( hdus )
		
		# and make sure extend is true
		hdulist.update_extend()

		# and write it out
		file = '%s%d_fit.fits' % (self.config['model_dir'],self.number)
		if os.path.isfile( file ): os.remove( file )
		hdulist.writeto( file )

	##########################
	## get position offsets ##
	##########################
	def get_position_offsets( self ):
		""" gblend.get_position_offsets()
		
		Returns an array of position offsets: out-in (pixels) """

		if not self.is_fit: raise ValueError( 'You must fit the gblend before retrieving position offsets!' )

		# loop through model objects.  In a way similar to self.get_model()
		new_params = list( self.best_fit )
		for (i,model_obj) in enumerate(self):

			# retrieve best fit offsets
			( shifts, new_params ) = model_obj.get_position_offset( new_params, return_parameters=True )

			# record shifts
			if i == 0:
				all_shifts = np.array( shifts )
			else:
				all_shifts = np.vstack( (all_shifts,shifts) )

		return all_shifts

	#################
	## get catalog ##
	#################
	def get_catalog( self ):
		""" gblend.get_catalog()
		
		Returns a catalog for the final fitted parameters of the high resolution objects in this model """

		if not self.is_fit: raise ValueError( 'You must fit the gblend before fetching the fitted catalog!' )

		# this returns a dictionary of key/values
		# each value is a list (not an array) with one value for
		# every hres object in the blend

		# loop through the model objects, in a very similar way to self.get_model()
		new_params = list( self.best_fit )
		for (i,model_obj) in enumerate(self):

			# get fitted parameter list
			( this_cat, new_params ) = model_obj.get_catalog( new_params, return_parameters=True )

			# record in catalog
			if i == 0:
				cat = this_cat
			else:
				for key in this_cat.keys(): cat[key].extend( this_cat[key] )

		# more stuff to add to catalog
		( model, chisq, nf ) = self.get_best_fit( return_chisq=True )
		cat['lres_id'] = [self.number]*self.nhres
		cat['chisq'] = [chisq]*self.nhres
		cat['nf'] = [nf]*self.nhres
		cat['chisq_nu'] = [chisq/(nf-1)]*self.nhres
		cat['nblend'] = [self.nhres]*self.nhres
		cat['mag_brightest'] = [min(cat['mag'])]*self.nhres
		cat['sky'] = [self.sky]*self.nhres

		# calculate distance to nearest object in blend
		# and the magnitude of the nearest object in the blend
		if self.nhres == 1:
			cat['nearest'] = [0]*self.nhres
			cat['nearest_mag'] = [0]*self.nhres
		else:
			nearest = []
			nearest_mag = []
			for i in range( self.nhres ):
				ra = cat['ra'][i]
				dec = cat['dec'][i]
				dist = np.sqrt( ( np.array(cat['ra'])-ra )**2.0*np.cos(dec/180.0*np.pi)**2.0 + (np.array(cat['dec'])-dec)**2.0 )*3600
				sinds = dist.argsort()
				nearest.append( dist[sinds[1]] )
				nearest_mag.append( cat['mag'][sinds[1]] )
			cat['nearest'] = nearest
			cat['nearest_mag'] = nearest_mag

		# calculate total flux of blend
		total_flux = np.sum( cat['flux'] )
		cat['total_flux'] = [total_flux]*self.nhres
		
		# and total magnitude
		cat['total_mag'] = [-2.5*np.log10( total_flux ) + self.config['lres_magzero']]*self.nhres
		
		# and then the fraction of the total flux accounted for by each object
		cat['blend_fraction'] = (np.asarray( cat['flux'] )/total_flux).tolist()
		
		# residuals and corresponding magnitude and fraction over segmentation region
		( mag, residuals, fraction ) = self.get_residuals_fraction( self.seg_m )
		cat['segmentation_mag'] = [mag]*self.nhres
		cat['segmentation_residuals'] = [residuals]*self.nhres
		cat['segmentation_fraction'] = [fraction]*self.nhres

		# residuals and corresponding magnitude and fraction over fitting region
		( mag, residuals, fraction ) = self.get_residuals_fraction( self.fit_m )
		cat['mask_mag'] = [mag]*self.nhres
		cat['mask_residuals'] = [residuals]*self.nhres
		cat['mask_fraction'] = [fraction]*self.nhres

		# convert model x/y coordinates back to global x/y coords
		cat['x'] = [self.xmin]*self.nhres
		cat['y'] = [self.ymin]*self.nhres
		for i in range( len( cat['img_x'] ) ):
			cat['x'][i] += cat['img_x'][i]
			cat['y'][i] += cat['img_y'][i]

		# finally return all catalog info from source extractor (except things that were already included)
		for field in self.info.array.names:
			field = field.lower()
			if field == 'number': continue
			cat[field] = [self.info[field]]*self.nhres

		return cat

	############################
	## get residuals fraction ##
	############################
	def get_residuals_fraction( self, m ):
		""" gblend.get_residuals_fraction( mask )
		
		Return the magnitude, magntiude residuals, and fractional residuals over a given mask to the image """

		# model image
		model = self.get_best_fit_full()

		# magnitude over image region
		mag = -2.5*np.log10( self.img[m].sum() ) + self.config['lres_magzero']

		# residuals over mask region
		residuals = (self.img[m] - model[m]).sum()

		# fractional residuals
		fraction = residuals/self.img[m].sum()

		# return
		return ( mag, residuals, fraction )

	######################
	## generate cutouts ##
	######################
	def generate_cutouts( self, img=None, rms=None, seg=None, seg_mask=None ):
		""" gblend.generate_stamps( img=None, rms=None, seg=None, seg_mask=None )
		
		Writes out fits images of cutouts for this gblend.
		img, rms, and seg keywords specify directory for output.  Configuration is used if they are None.
		Set to False to supress that particular cutout. """

		if not self.images_loaded: raise ValueError( 'You must load images before generating cutouts!' )

		if img is None: img = self.config['image_dir']
		if rms is None: rms = self.config['rms_dir']
		if seg is None: seg = self.config['segmentation_dir']
		if seg_mask is None: seg_mask = self.config['segmentation_mask_dir']

		# output the cutouts!
		if img is not False: self.output_cutout( img, '%d_img.fits' % self.number, self.img, self.img_hdr )
		if rms is not False: self.output_cutout( rms, '%d_rms.fits' % self.number, self.rms, self.img_hdr )
		if seg is not False: self.output_cutout( seg, '%d_seg.fits' % self.number, self.seg, self.img_hdr )
		if seg_mask is not False: self.output_cutout( seg_mask, '%d_seg_mask.fits' % self.number, self.fit_mask, self.img_hdr )

	###################
	## output cutout ##
	###################
	def output_cutout( self, dir, filename, img, hdr ):
		""" gblend.output_cutout( dir, filename, img, hdr ) """

		if not os.path.isdir( dir ): os.makedirs( dir, mode=0755 )
		file = '%s%s' % (dir,filename)
		# delete the file if it exists to avoid the notice when clobbering with pyfits
		if os.path.isfile( file ): os.remove( file )
		pyfits.writeto( file, img, hdr )