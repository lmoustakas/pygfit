import numpy as np

# try importing pycuda
try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	has_pycuda = True
except ImportError:
	has_pycuda = False

class gpu(object):
	""" gpu class.  Used for running some pygfit functionality through a gpu with pycuda """
	
	# whether or not the cuda code has been processed by pycuda
	# we don't want this step to be done until needed, because
	# it takes a few seconds
	cuda_loaded = False
	
	# load/functions for various cuda routines
	convolve2d_loaded = False
	convolve2d_mod = ''
	convolve2d_func = ''
	sersic_loaded = False
	sersic_mod = ''
	sersic_func = ''
	shift_loaded = False
	shift_mod = ''
	shift_2coeffsX = ''
	shift_2coeffsY = ''
	
	##########
	## init ##
	##########
	def __init__ ( self, nthreads=512 ):
		""" gpu_object = gpu.gpu( nthreads=512 )
		
		Load a gpu object for executing gpu functionality.
		nthreads = The number of threads to execute per block. """
		
		# nothing to do if we don't support gpu processing
		if not has_pycuda:
			return None
		
		self.nthreads = int( nthreads )
	
	##############
	## force_le ##
	##############
	def force_le( self, array ):
		""" gup.force_le( array )
		
		Byteswaps the array to little endian if not already.
		CUDA requires all arrays to be little endian """
		
		# make sure we have a numpy array
		array = np.asarray( array )
		
		# return if it is already little endian
		# the check is done on the dtype so that we don't
		# have to bytswap the entire array if not necessary
		if array.dtype == array.dtype.newbyteorder('<'):
			return array
		
		# if we got this far then it is big endian and must be swapped
		return array.byteswap().newbyteorder()
	
	###################
	## force float32 ##
	###################
	def force_float32( self, array ):
		""" gpu.force_float32( array )
		
		Converts the array to a float32 numpy array if not already """
		
		# make sure we have a numpy array
		array = np.asarray( array )
		
		# return if it is already float32
		if array.dtype == np.dtype( 'float32' ):
			return array
		
		# otherwise convert
		return array.astype( 'float32' )
	
	########################
	## prepare convolve2d ##
	########################
	def _prepare_convolve2d( self ):
		""" gpu._prepare_convolve2d()
		
		Prepare the convolve2d function for cuda execution """
		
		# can we actually run pycuda?
		if not has_pycuda:
			return False
		
		# has this been loaded yet?
		if self.convolve2d_loaded:
			return True
		
		# load it
		self.convolve2d_mod = SourceModule( convolve_code )
		self.convolve2d_func = self.convolve2d_mod.get_function( 'convolve2d_float' )
		
		# update status
		self.convolve2d_loaded = True
		
		# and return
		return True
	
	################
	## convolve2d ##
	################
	def convolve2d( self, img, kernel ):
		""" img = gpu.convolve2d( img, kernel )
		
		Convolve an array with an arbitrary kernel, represented by another array """
		
		# make sure the cuda routine is ready to go
		if not self._prepare_convolve2d():
			raise ValueError( 'Cannot run convolve2d on gpu: gpu functionality is not supported' )
		
		# make sure both arrays are in little endian format and are float32 arrays
		img = self.force_float32( self.force_le( img ) )
		kernel = self.force_float32( self.force_le( kernel ) )
		
		# number of rows and columns in the image to convolve
		( nrows, ncols ) = img.shape
		
		# number of rows and columns in the kernel
		( knrows, kncols ) = kernel.shape
		
		# output array
		out = np.empty( img.shape, np.float32 )
		
		# figure out the number of blocks needed to execute this...
		blocks = int( np.ceil( float( img.size )/self.nthreads ) )
		
		# and now run
		self.convolve2d_func( drv.In(img), drv.Out(out), drv.In(kernel), np.int32(nrows), np.int32(ncols), np.int32(knrows), np.int32(kncols), grid=(blocks,1), block=(self.nthreads,1,1) )
		
		# and return
		return out

	####################
	## prepare sersic ##
	####################
	def _prepare_sersic( self ):
		""" gpu._prepare_sersic()
		
		Prepare the sersic function for cuda execution """
		
		# is pycuda available?
		if not has_pycuda:
			return False
		
		# has this been loaded yet?
		if self.sersic_loaded:
			return True
		
		# load it
		self.sersic_mod = SourceModule( sersic_code )
		self.sersic_func = self.sersic_mod.get_function( 'sersic_float' )
		
		# update status
		self.sersic_loaded = True
		
		# and return
		return True

	############
	## sersic ##
	############
	def sersic( self, xdiff, ydiff, mu, re, n, b, ba, cos_pa, sin_pa, c2 ):
		""" img = gpu.sersic( xdiff, ydiff, mu, re, n, b, ba, cos_pa, sin_pa, c2 )
		
		Calculate the sersic model at xdiff=(x-x_cent) and ydiff=(y-y_cent).  x & y can be arrays. """
		
		# make sure the cuda routine is ready to go
		if not self._prepare_sersic():
			raise ValueError( 'Cannot run sersic on gpu: gpu functionality is not supported' )
		
		# make sure both arrays are in little endian format and are float32 arrays
		xdiff = self.force_float32( self.force_le( xdiff ) )
		ydiff = self.force_float32( self.force_le( ydiff ) )
		
		# array size
		size = xdiff.size
		
		# output array
		out = np.empty( xdiff.shape, np.float32 )
		
		# figure out the number of blocks needed to execute this...
		blocks = int( np.ceil( float( size )/self.nthreads ) )
		
		# and now run
		self.sersic_func( drv.Out(out), drv.In(xdiff), drv.In(ydiff), np.int32(size), np.float32(mu), np.float32(re), np.float32(n), np.float32(1./n), np.float32(b), np.float32(ba), np.float32(cos_pa), np.float32(sin_pa), np.float32(c2), np.float32(1./c2), np.float32(np.e), grid=(blocks,1), block=(self.nthreads,1,1) )
		
		# and return
		return out

	###################
	## prepare shift ##
	###################
	def _prepare_shift( self ):
		""" gpu._prepare_shift()
		
		Prepare the shift function for cuda execution """
		
		# is pycuda available?
		if not has_pycuda:
			return False
		
		# has this been loaded yet?
		if self.shift_loaded:
			return True
		
		# load it.  First calculate constants Pole and Lambda and propogate into code
		# these were defined contstants in C, so I'll calculate them in python
		pole = np.sqrt( 3.0 ) - 2.0
		l = (1 - pole)*(1 - 1/pole)
		code = shift_code.replace( '@@POLE@@', '%ff' % pole ).replace( '@@LAMBDA@@', '%ff' % l )
		self.shift_mod = SourceModule( code )
		
		self.shift_2coeffsX = self.shift_mod.get_function( 'SamplesToCoefficients2DX' )
		self.shift_2coeffsY = self.shift_mod.get_function( 'SamplesToCoefficients2DY' )
		
		# update status
		self.shift_loaded = True
		
		# and return
		return True
	
	################
	## pre filter ##
	################
	def _pre_filter( self, data, return_gpu = True ):
		""" gpu._pre_filter( data, return_gpu = True )
		
		Prefilters a 2D array to prepare for spline interpolation.
		Returns the array in GPU memory if return_gpu = True """
		
		if not self._prepare_shift():
			return False
		
		# make sure the data array is float32 with little endian type
		data = self.force_float32( self.force_le( data ) )
		
		# get array size
		( height, width ) = data.shape
		
		# allocate memory on GPU with pitch
		( data_gpu, pitch ) = self._pitch_allocate( data )
		
		# and copy data array to gpu
		drv.memcpy_htod( data_gpu, data )
		
		# now prefilter in one direction
		threads_per_block = min( self._power_two_divider( height ), 64 )
		blocks = height/threads_per_block
		self.shift_2coeffsX( data_gpu, np.int32( pitch ), np.int32( width ), np.int32( height ), grid=(blocks,1), block=(threads_per_block,1,1) )
		
		# and then the other
		threads_per_block = min( self._power_two_divider( width ), 64 )
		blocks = width/threads_per_block
		self.shift_2coeffsY( data_gpu, np.int32( pitch ), np.int32( width ), np.int32( height ), grid=(blocks,1), block=(threads_per_block,1,1) )
		
		# return gpu array
		if return_gpu:
			return data_gpu
		
		# copy back from gpu
		drv.memcpy_dtoh( data, data_gpu )
		
		# free memory
		data_gpu.free()
		
		# and return
		return data
	
	####################
	## pitch allocate ##
	####################
	def _pitch_allocate( self, array ):
		""" ( alloc, pitch ) = gpu._pitch_allocate( array )
		
		Allocates memory space on the GPU (with pitch) to fit the passed array.
		Returns the gpu memory array and the pitch (the width of the array in bytes) """
		
		# array shape
		( height, width ) = array.shape
		
		# size of element (in bytes)
		size = array.nbytes/array.size
		
		return drv.mem_alloc_pitch( width*size, height, size )
	
	#######################
	## power two divider ##
	#######################
	def _power_two_divider( self, n ):
		if not n: return 0;
		divider = 1
		n = int( n )
		while (n & divider) == 0:
			divider = divider << 1
		return divider

convolve_code = """
__global__ void convolve2d_float(float *data, float *output, float *kernel, int rows, int cols, int kny, int knx)
{
	/*** need to pass rows and cols as int32(rows) and int32(cols)
	* same with kny, knx
	* i here is the array index for the output array
	*/

	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	int row = i/cols - kny/2;
	int col = i % cols - knx/2;
	int currRow = 0;
	int currCol = 0;
	/*** (row, col) = (row, col) at upper corner of kernel ***/
	output[i] = 0;

	/*** Boundary = pad with zeros.  Again double loop but this time
	* continue if outside of array bounds (no need to actually add 0 of
	* course).
	*/
	for (int j = 0; j < knx; j++)
	{
		for (int k = 0; k < kny; k++)
		{
			currCol = col + j;
			currRow = row + k;
			if (currCol < 0 || currCol >= cols || currRow < 0 || currRow >= rows) continue;
			output[i] += kernel[j+k*knx]*data[currRow*cols+currCol];
		}
	}
}"""

sersic_code = """
__global__ void sersic_float(float *output, float *xdiff, float *ydiff, int size, float mu, float re, float n, float ni, float b, float ba, float cos_pa, float sin_pa, float c2, float c2i, float e)
{
	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	
	if (i >= size) return;

	float p = xdiff[i]*cos_pa + ydiff[i]*sin_pa;
	p = p < 0 ? -p : p;
	float m = ydiff[i]*cos_pa - xdiff[i]*sin_pa;
	m = m < 0 ? -m : m;

	float r = pow( pow( p, c2 ) + pow( m/ba, c2 ), c2i )/re;
	
	output[i] = mu*exp( -b*( pow( r, ni ) - 1) );
}
"""

shift_code = """
__device__ float InitialCausalCoefficient( float* c, int DataLength, int step )
{
	const int Horizon = min(12, DataLength);

	// this initialization corresponds to clamping boundaries
	// accelerated loop
	float zn = @@POLE@@;
	float Sum = *c;
	for (int n = 0; n < Horizon; n++) {
		Sum += zn * *c;
		zn *= @@POLE@@;
		c = (float*)((char*)c + step);
	}
	return(Sum);
}

__device__ float InitialAntiCausalCoefficient( float* c, int DataLength, int step )
{
	// this initialization corresponds to clamping boundaries
	return((@@POLE@@ / (@@POLE@@ - 1.0f)) * *c);
}

__device__ void ConvertToInterpolationCoefficients( float* coeffs, int DataLength, int step )
{
	// causal initialization
	float* c = coeffs;
	float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
	*c = previous_c = @@LAMBDA@@ * InitialCausalCoefficient(c, DataLength, step);
	// causal recursion
	for (int n = 1; n < DataLength; n++) {
		c = (float*)((char*)c + step);
		*c = previous_c = @@LAMBDA@@ * *c + @@POLE@@ * previous_c;
	}
	// anticausal initialization
	*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
	// anticausal recursion
	for (int n = DataLength - 2; 0 <= n; n--) {
		c = (float*)((char*)c - step);
		*c = previous_c = @@POLE@@ * (previous_c - *c);
	}
}

__global__ void SamplesToCoefficients2DX( float* image, int pitch, int width, int height )
{
	// process lines in x-direction
	const int y = blockIdx.x * blockDim.x + threadIdx.x;
	float* line = (float*)((char*)image + y * pitch);

	ConvertToInterpolationCoefficients( line, width, sizeof(float) );
}

__global__ void SamplesToCoefficients2DY( float* image, int pitch, int width, int height )
{
	// process lines in y-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	float* line = image + x;

	ConvertToInterpolationCoefficients( line, height, pitch );
}
"""