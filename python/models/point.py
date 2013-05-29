import numpy as np
import mapping
import scipy.ndimage

class point(object):
	""" point source model class.

	Initialize by passing a dictionary with data from high resolution catalog. """
	model_type = 'point'

	# model parameters
	zeropoint = 0		# the image zeropoint
	id = ''			# object id
	ra = 0			# right ascension
	dec = 0			# declination
	x = 0			# x coord (on low resolution image)
	y = 0			# y coord (on low resolution image)
	img_x = 0		# x coord (on low resolution cutout)
	img_y = 0		# y coord (on low resolution cutout)
	ind_x = 0		# x ind (to low resolution cutout array)
	ind_y = 0		# y ind (to low resolution cutout array)
	lim_x = {}		# x limit (dictionary: 'min', 'max')
	lim_y = {}		# y limit (dictionary: 'min', 'max')
	mag = 0			# magnitude
	flux = 0.0		# flux

	# gpu object for performing gpu calcuations
	# will be False if gpu calculations are not possible
	gpu = False

	# sersic parameters (will remain zero for point object)
	max_array_size = 1e7	# maximum number of data points to calculate radii for when computing models
	re = 0.0		# half light radius
	n = 0.0			# sersic index
	pa = 0.0		# position angle
	ba = 0.0		# axis ratio

	# a warning flag for when the sersic model failed to integrate properly
	# declared here because it needs to be output for point and sersic
	# models, even though only sersic models will ever change this
	mag_warning = False

	# object state
	modeled = False

	##########
	## init ##
	##########
	def __init__( self, data, zeropoint, max_array_size=1e7, gpu=False ):
		""" model = models.point( hres_object_dict, zeropoint, max_array_size=1e7 )
		
		Return a point source model object given a dictionary with data from high resolution catalog """

		# just copy out all the relevant data parameters
		self.id = data['id']
		self.ra = data['ra']
		self.dec = data['dec']
		self.x = data['x']
		self.y = data['y']
		self.img_x = data['img_x']
		self.img_y = data['img_y']
		self.ind_x = data['ind_x']
		self.ind_y = data['ind_y']
		self.lim_x = data['lim_x']
		self.lim_y = data['lim_y']
		self.mag = data['mag']
		self.zeropoint = zeropoint
		self.flux = 10**(-0.4*(self.mag-self.zeropoint))
		self.max_array_size = max_array_size
		
		# store gpu object (or False)
		self.gpu = gpu

	####################
	## generate model ##
	####################
	def generate_model( self, pad, img_shape, psf, shift=True, use_integration=True ):
		""" point.generate_model( pad, img_shape, psf, shift=True )
		
		Generate the model and store it in the object. Pass the padding length, image size, and the psf
		pad:		the padding length
		img_shape:	img.shape (for the lres cutout)
		psf:		the psf image
		shift = True: shift model to precisely reproduce centering
		"""

		# the basic model is just the psf * the surface brightness
		# this needs to be placed in the image at (roughly) the object location

		( x, y ) = self.generate_xy( img_shape, pad )

		model = np.zeros( x.shape )

		# object x/y position on padded image
		xc = self.ind_x + pad
		yc = self.ind_y + pad

		# now drop it onto the image
		xrad = (psf[0,:].size)/2
		yrad = (psf[:,0].size)/2

		# copy the psf onto the padded array and multiply by the flux
		model[yc-yrad:yc+yrad+1,xc-xrad:xc+xrad+1] = self.flux*psf.copy()

		# Center the point source on the object
		if shift: model = scipy.ndimage.shift( model, [ self.img_y-np.round(self.img_y), self.img_x-np.round(self.img_x) ] )

		# save just the part that matches the image cutout
		self.model_img = model[pad:int(img_shape[0]+pad),pad:int(img_shape[1]+pad)]
		self.mag_initial = -2.5*np.log10( self.model_img.sum() ) + self.zeropoint
		self.modeled = True

	#################
	## generate xy ##
	#################
	def generate_xy( self, img_shape, pad ):

		inds = np.mgrid[-1*pad+1:img_shape[0]+pad+1,-1*pad+1:img_shape[1]+pad+1]
		return ( inds[1], inds[0] )

	###############
	## set model ##
	###############
	def set_model( self, model_img ):
		""" point.set_model( model_img )
		
		Sets the given model in the model object """

		self.model_img = model_img
		self.mag_initial = -2.5*np.log10( self.model_img.sum() ) + self.zeropoint
		self.modeled = True

	#####################
	## get first guess ##
	#####################
	def get_first_guess( self ):
		""" point.get_first_guess()
		
		returns a list of guesses for all free parameters for this model. """

		# use mapping functions to retrieve first guess of x/y.
		xguess = mapping.forwardmap( self.img_x, self.lim_x )
		yguess = mapping.forwardmap( self.img_y, self.lim_y )

		# just pass along the flux as the first guess for flux
		fluxguess = self.flux

		# and return
		return [float( xguess ),float( yguess ),float( fluxguess )]

	######################
	## model parameters ##
	######################
	def model_parameters( self, params, return_parameters=False ):
		""" model = point.model_parameters( params, return_parameters=False )
		
		Returns the model with the parameters given in params.
		Expects params=[x,y,flux]
		if return_parameters=True then it will return ( model, new_parameters )
		where new_parameters is the input parameter array with the parameters used by this model removed. """

		# first make sure the model has been generated, since if it hasn't nothing can be done
		if not self.modeled: raise ValueError( 'Error modeling parameters: the model has not been created yet!' )

		if len( params ) < 3: raise ValueError( 'Did not receive enough parameters to build a model!' )

		# first remap x/y back to physical coordinates
		# location of parameters in params array is determined by return order of self.get_first_guess()
		new_x = mapping.remap( params[0], self.lim_x )
		new_y = mapping.remap( params[1], self.lim_y )

		# calculate position shift
		shift = [ new_y-self.img_y, new_x-self.img_x ]

		# calculate the flux ratio - new flux/old flux
		flux_ratio = np.abs( params[2]/self.flux )
		#if params[2] < 0: flux_ratio = 1e9

		# now adjust model
		new_model = flux_ratio*scipy.ndimage.shift( self.model_img, shift )

		# and return
		if not return_parameters: return new_model

		# return_parameters == True: also return parameters array with first three parameters gone
		return ( new_model, self.get_trimmed( params ) )

	#########################
	## get position offset ##
	#########################
	def get_position_offset( self, params, return_parameters=False ):
		""" (dy,dx) = point.get_position_offset( params, return_parameters=False )
		
		Returns the position offset (in pixels) between the parameters in the given list, and the original position.
		Expects params=[x,y,flux] (with x & y in mapped units)
		if return_parameters=True then it will return ( model, new_parameters )
		where new_parameters is the input parameter array with the parameters used by this model removed. """

		if len( params ) < 3: raise ValueError( 'Did not receive enough parameters to calculate offsets!' )

		# first remap x/y back to physical coordinates
		# location of parameters in params array is determined by return order of self.get_first_guess()
		new_x = mapping.remap( params[0], self.lim_x )
		new_y = mapping.remap( params[1], self.lim_y )

		# calculate position offset
		shifts = np.array( [ new_y-self.img_y, new_x-self.img_x ] )

		# return
		if not return_parameters: return shift

		# return_parameters == True: also return parameters array with first three parameters gone
		return ( shifts, self.get_trimmed( params ) )

	#################
	## get catalog ##
	#################
	def get_catalog( self, params, return_parameters=False ):
		""" [x,y,mag] = point.get_catalog( params, return_parameters=False )
		
		Returns a basic catalog of fit parameters.
		Expects params=[x,y,flux] (with x & y in mapped units)
		if return_parameters=True then it will return ( cat, new_parameters )
		where new_parameters is the input parameter array with the parameters used by this model removed. """

		if len( params ) < 3: raise ValueError( 'Did not receive enough parameters to return catalog!' )

		# first remap x/y back to physical coordinates
		# location of parameters in params array is determined by return order of self.get_first_guess()
		new_x = mapping.remap( params[0], self.lim_x )
		new_y = mapping.remap( params[1], self.lim_y )

		# calculate the flux in the model
		mag_image = -2.5*np.log10( self.model_parameters( params ).sum() ) + self.zeropoint

		# the sersic parameters will be zero for the point source model, and non-zero for sersic models
		cat = {'hres_id': [self.id], 'model': [self.model_type], 'img_x': [new_x], 'img_y': [new_y], 'flux': [params[2]], 'mag': [self.flux_to_mag( params[2] )], 'mag_image': [mag_image], 'mag_initial': [self.mag_initial], 'mag_hres': [self.mag], 're_lres': [self.re], 'n': [self.n], 'pa': [self.pa], 'ba': [self.ba], 'ra': [self.ra], 'dec': [self.dec], 'mag_warning': [self.mag_warning]}
		if not return_parameters: return cat

		# return_parameters == True: also return parameters array with first three parameters gone
		return ( cat, self.get_trimmed( params ) )

	#################
	## get trimmed ##
	#################
	def get_trimmed( self, params ):
		""" new_params = point.get_trimmed( params )
		
		returns a trimmed down parameter list so that those parameters that correspond to this model are removed.
		"""

		# this was created to leave room for changes later.
		# if I decide to make it so that the fixed/free parameters can be set on config,
		# having this in a function by itself will help.
		return params[3:]

	#################
	## flux to mag ##
	#################
	def flux_to_mag( self, flux ):
		""" mag = point.flux_to_mag( flux ) """

		return -2.5*np.log10( np.abs( flux ) ) + self.zeropoint