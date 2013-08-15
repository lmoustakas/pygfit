import point,mapping
import scipy
import scipy.ndimage
import scipy.signal
import scipy.special
import numpy as np
import traceback

def bfunc(x,twon):
	return 1.-2.*scipy.special.gammaincc(twon,x)

class sersic(point.point):
	""" sersic source model class.

	Initialize by passing a dictionary with data from high resolution catalog. """
	model_type = 'sersic'

	# additional model parameters
	re = 0.0	# half light radius
	n = 0.0		# sersic index
	pa = 0.0	# position angle
	ba = 0.0	# axis ratio
	b = 0.0		# b (calculated from sersic index)
	mu = 0.0	# surface brightness
	
	flux_re = 0.0	# flux within a circle of radius re
	mag_re = 0.0	# magnitude within a circle of radius re
	
	# computation parameters
	max_rebin_factor = 200
	sample_factor = 20.0	# Always want re to be sample_factor larger than the pixels
	reduction_factor = 10.0	# Don't need the sampling to be as good away from the object center (only 1/reduction_factor as good)
	nre = 2			# number of effective radii to treat with higher pixel sampling

	##########
	## init ##
	##########
	def __init__( self, data, zeropoint, max_array_size=1e7, gpu=False ):
		""" model = models.sersic( hres_object_dict, zeropoint, max_array_size=1e7 )
		
		Return a sersic model object given a dictionary with data from high resolution catalog """

		# the point source initializer will set the basic info (pos, mag, etc...)
		super( sersic, self ).__init__( data, zeropoint, gpu=gpu )

		# now copy the sersic specific parameters
		self.max_array_size = max_array_size
		if self.max_array_size is None: self.max_array_size = 1e7
		self.re = data['re_pix']
		self.n = data['n']
		self.b = self.get_b( self.n )
		self.pa = data['pa']
		self.ba = data['ba']
		self.c = 0
		if data.has_key( 'c' ): self.c = c

		# parameter checks
		if self.c + 2.0 == 0: raise ValueError( 'Invalid value for c!' )
		if self.re <= 0: raise ValueError( 'Re must be greater than 0!' )
		if self.n <= 0: raise ValueError( 'Sersic Index must be greater than 0!' )

		# now set the surface brightness
		self.mu = self.get_surface_brightness()

		# calculate the total flux from the surface brightness
		self.flux = self.get_total_flux()

	def get_b( self, n=None ):
		if n is None: n = self.n
		return scipy.optimize.minpack.fsolve( bfunc, 2.*n - 0.32, args=(2.*n) )

	############################
	## get surface brightness ##
	############################
	def get_surface_brightness( self, mag=None ):
		""" mu = models.sersic.get_surface_brightness( mag )
		
		Return the surface brightness of the sersic source in the model. """

		if mag == None: mag = self.mag

		g = scipy.special.gamma( 2*self.n )
		invc = 1.0/(self.c+2)
		beta = scipy.special.beta( invc, 1.0 + invc )
		rc = np.pi/( 4.0 * invc * beta )
		ftot = 10.0**( -0.4*( self.mag - self.zeropoint ) )

		return ftot / ( 2.0*np.pi*self.re**2.0*np.exp( self.b )*self.n*self.b**(-2.0*self.n)*g*self.ba*rc )

	####################
	## get total flux ##
	####################
	def get_total_flux( self ):
		""" flux = models.sersic.get_total_flux() """

		#Definition of total flux: f=2*pi*re**2*exp(b)*n*b**(-2*n)*gamma(2n)*q/R(c)
		g = scipy.special.gamma( 2*self.n )
		invc = 1.0/(self.c+2)
		beta = scipy.special.beta( invc, 1.0 + invc)
		rc = np.pi/( 4.0 * invc * beta )
		return 2.0*np.pi*self.re**2.0*abs(self.mu)*np.exp(self.b)*self.n*self.b**(-2.0*self.n)*g*self.ba*rc

	####################
	## generate model ##
	####################
	def generate_model( self, pad, img_shape, psf, shift=True ):
		""" sersic.generate_model( pad, img_shape, psf )
		
		Generate the model and store it in the object. Pass the padding length, the psf, and the x/y arrays (for the padded array, and gblend.y)
		pad:		the padding length
		img_shape:	img.shape (for the lres cutout)
		psf:		the psf image
		"""

		( x, y ) = self.generate_xy( img_shape, pad )

		(height,width) = x.shape

		# calculate sizes and scales
		rebin_factor = int( np.round( self.sample_factor/self.re ) )
		rebin_factor_small = int( float(rebin_factor) / self.reduction_factor )

		if( rebin_factor_small < 1 ): rebin_factor_small = 1
		if( rebin_factor > self.max_rebin_factor ): rebin_factor = self.max_rebin_factor
		if( rebin_factor_small > self.max_rebin_factor ): rebin_factor_small = self.max_rebin_factor

		x_cent = np.round( self.img_x )
		y_cent = np.round( self.img_y )

		centerpix = np.where( (x==x_cent) & (y==y_cent) )
		xcpix = int(centerpix[0])
		ycpix = int(centerpix[1])

		# size to treat with higher pixel sampling
		dr = int( np.round( self.nre*self.re ) )

		# generate first order model
		if rebin_factor_small == 1:
			# if rebin_factor_small is one, then just generate a sersic model on the current indices
			tmp = self.model( x, y, x_cent, y_cent, chunk=False )
			
		else:
			# if rebin_factor_small is greater than one, then initially generate the model with higher pixel sampling
			nxnew = rebin_factor_small*width
			xmax = width-1
			dx = (np.abs(x[0][0]-x[xmax][xmax])+1)/float(width)

			# generate list of x & y coordinates to generate model at.  Broadcasting will make both two dimensional for the final model.
			# generate as float32 for gpu support
			xx = x[0][0] - 0.5 + (np.arange( nxnew, dtype='float32' ) + 0.5)*dx/rebin_factor_small
			yy = y[0][0] - 0.5 + (np.arange( nxnew, dtype='float32' ).reshape( (nxnew,1) ) + 0.5)*dx/rebin_factor_small

			# now generate the resampled model and then average it back down to the image sampling
			tmp = self.shrink_array( self.model( xx, yy, x_cent, y_cent ), width, width )

		# now use a larger pixel sampling near the center
		if rebin_factor>rebin_factor_small:

			# Make sure in case re is strangely large (galfit error) that we don't exceed bounds of padded region
			if( dr > width*0.5-3.0 ): dr = width*0.5-3.0

			# Aim to avoid case where object too close to edges
			if( dr > x_cent ): dr = x_cent
			if( dr > y_cent ): dr = y_cent

			# dr must be at least 1 pixel in size
			if(dr==0): dr=1

			nxn = 2*dr+1
			nxnew = (nxn)*rebin_factor

			# generate list of x & y coordinates to generate model at.  Broadcasting will make both two dimensional for the final model.
			# generate as float32 for gpu support
			xx = x_cent - dr - 0.5 + (np.arange( nxnew, dtype='float32' ) + 0.5)/rebin_factor
			yy = y_cent - dr - 0.5 + (np.arange( nxnew, dtype='float32' ).reshape( (nxnew,1) ) + 0.5)/rebin_factor

			tmp[ xcpix-dr:xcpix-dr+nxn, ycpix-dr:ycpix-dr+nxn ] = self.shrink_array( self.model( xx, yy, x_cent, y_cent ), nxn, nxn )

		# integrate the central pixel
		# for large n the big peak in the center is hard to integrate so divide up into parts to minimize integral over very center
		if self.n > 6.5 and self.re < 1:
			cent = self.re/100
			tmp[ xcpix, ycpix ] = 0
			tmp[ xcpix, ycpix ] += scipy.integrate.quadpack.dblquad( self.model, y_cent-0.5, y_cent+0.5, lambda x: x_cent-0.5, lambda x: x_cent-cent, args=( x_cent, y_cent ), epsabs=1e-3 )[0]
			tmp[ xcpix, ycpix ] += scipy.integrate.quadpack.dblquad( self.model, y_cent-0.5, y_cent+0.5, lambda x: x_cent+cent, lambda x: x_cent+0.5, args=( x_cent, y_cent ), epsabs=1e-3 )[0]
			tmp[ xcpix, ycpix ] += scipy.integrate.quadpack.dblquad( self.model, y_cent-cent, y_cent-0.5, lambda x: x_cent-cent, lambda x: x_cent+cent, args=( x_cent, y_cent ), epsabs=1e-3 )[0]
			tmp[ xcpix, ycpix ] += scipy.integrate.quadpack.dblquad( self.model, y_cent+cent, y_cent+0.5, lambda x: x_cent-cent, lambda x: x_cent+cent, args=( x_cent, y_cent ), epsabs=1e-3 )[0]
			tmp[ xcpix, ycpix ] += scipy.integrate.quadpack.dblquad( self.model, y_cent-cent, y_cent+cent, lambda x: x_cent-cent, lambda x: x_cent+cent, args=( x_cent, y_cent ), epsabs=1e-3 )[0]
		else:
			tmp[ xcpix, ycpix ] = scipy.integrate.quadpack.dblquad( self.model, y_cent-0.5, y_cent+0.5, lambda x: x_cent-0.5, lambda x: x_cent+0.5, args=( x_cent, y_cent ), epsabs=1e-3 )[0]

		if shift: tmp = scipy.ndimage.shift( tmp, [self.img_y-y_cent, self.img_x-x_cent] )
		
		# calculate the flux and magnitude of the model within re.
		# this will be stored and later scaled to calculate flux_re and mag_re of best fitting model
		#self.flux_re = dblquad( self.model_integrate, 0, self.re, lambda x: 0, lambda x: 2*np.pi )[0]
		#self.mag_re = -2.5*np.log10( self.flux_re ) + self.zeropoint
		
		# finally convolve.  Use the GPU if available.  If not or if it fails, revert to scipy
		modeled = False
		if self.gpu:
			try:
				self.model_img = self.gpu.convolve2d( tmp, psf )
				modeled = True
			except:
				print "GPU processing failed.  Reverting to slower scipy fftconvolve.  Traceback:\n%s" % traceback.format_exc()
		
		if not modeled:
			self.model_img = scipy.signal.signaltools.fftconvolve( tmp, psf, mode='same' )
		
		# and extract just the part that overlaps with the cutout
		self.model_img = self.model_img[pad:int(img_shape[0]+pad),pad:int(img_shape[1]+pad)]
		self.mag_initial = -2.5*np.log10( self.model_img.sum() ) + self.zeropoint
		self.modeled = True

	###########
	## model ##
	###########
	def model( self, x, y, x_cent=None, y_cent=None, chunk=True ):
		""" models.sersic.model( x, y, x_cent=None, y_cent=None )
		
		Calculate the stored sersic model as a function of x/y for the sersic model centered at (x_cent,y_cent).
		x_cent and y_cent default to self.img_x and self.img_y """

		if x_cent is None: x_cent = self.img_x
		if y_cent is None: y_cent = self.img_y
		
		pa = (self.pa+90.)/180.0*np.pi # designed to match galfit
		cos_pa = np.cos( pa )
		sin_pa = np.sin( pa )
		c2 = self.c + 2.0

		xdiff = x-x_cent
		ydiff = y-y_cent
		
		# process with gpu if available, revert to python if not or if it fails.
		if self.gpu and type( x ) == type( np.array( [] ) ):
			# however, don't bother if it is just a small array or number
			if x.size > 20:
				try:
					# sometimes model is passed rows for x and y
					# with the intention that numpys broadcasting rules will fill them out
					# however, the gpu won't do that.  So we have to do it ourself
					if ydiff.shape[1] == 1 and len( xdiff.shape ) == 1:
						xdiff.shape = (1,xdiff.size)
						xdiff = xdiff.repeat( xdiff.size, axis=0 )
						ydiff = ydiff.repeat( ydiff.size, axis=1 )
					return self.gpu.sersic( xdiff, ydiff, self.mu, self.re, self.n, self.b, self.ba, cos_pa, sin_pa, c2 )
				except:
					print "GPU processing failed.  Reverting to python for sersic calculation.  Traceback:\n%s" % traceback.format_exc()

		# radius of each point in image.  Do this in a few chunks to save memory
		if type( x ) == type( np.array( [] ) ):
			nchunks = int( np.ceil( x.size**2.0/self.max_array_size ) )
		else:
			nchunks = 1

		if nchunks == 1 or not chunk:
			r = ( ( np.abs( xdiff*cos_pa + ydiff*sin_pa) )**c2 + ( np.abs( (ydiff*cos_pa - xdiff*sin_pa) / self.ba) )**c2 )**(1./c2)/self.re
			return self.mu * np.exp( -self.b*( r**(1.0/self.n) - 1 ) )
		else:
			# model array.  Make it a float32 so it is gpu ready
			model = np.empty( (x.size,x.size), dtype='float32' )

			# chunk starting/ending indexes
			inds = np.append( np.arange( nchunks )*model.shape[0]/nchunks, model.shape[0] )

			# calculate model in chunks
			for i in range( nchunks ):
				model[inds[i]:inds[i+1],:] = self.mu * np.exp( -self.b*( (( ( np.abs( xdiff*cos_pa + ydiff[inds[i]:inds[i+1],:]*sin_pa) )**c2 + ( np.abs( (ydiff[inds[i]:inds[i+1],:]*cos_pa - xdiff*sin_pa) / self.ba) )**c2 )**(1./c2)/self.re)**(1.0/self.n) - 1 ) )
			return model

	#####################
	## model_integrate ##
	#####################
	def model_integrate( self, theta, r ):
		""" models.sersic.model( theta, r )
		
		Calculate model as a function of theta and r.  Used with dblquad to integrate
		sersic model from 0 to some radius """
		pa = (self.pa+90.)/180.0*np.pi # designed to match galfit
		cos_pa = np.cos( pa )
		sin_pa = np.sin( pa )
		c2 = self.c + 2.0
		z = ( ( np.abs( r*np.cos(theta)*cos_pa + r*np.sin(theta)*sin_pa) )**c2 + ( np.abs( (r*np.sin(theta)*cos_pa - r*np.cos(theta)*sin_pa) / self.ba) )**c2 )**(1./c2)/self.re
		return self.mu * np.exp( -self.b*( z**(1.0/self.n) - 1 ) )

	def shrink_array( self, array, m, n ):
		""" shrink_array( array, m, n )
		
		Shrinks a 2-D array to size of (m,n)
		Equivalent to IDL rebin() when shrinking 2-d arrays. """

		M,N = array.shape
		Mm, Nn = M/m, N/n
		return np.sum( np.sum( np.reshape(array,(m,Mm,n,Nn) ),3 ),1 )/float( Mm*Nn )

	#####################
	## get first guess ##
	#####################
	def get_first_guess( self ):
		""" point.get_first_guess()
		
		returns a list of guesses for all free parameters for this model. """

		# use mapping functions to retrieve first guess of x/y.
		xguess = mapping.forwardmap( self.img_x, self.lim_x )
		yguess = mapping.forwardmap( self.img_y, self.lim_y )

		# just pass along the surface brightness as the first guess for flux
		fluxguess = self.flux

		# and return
		return [float( xguess ),float( yguess ),float( fluxguess )]
