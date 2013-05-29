import numpy as np

# try importing pycuda
try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	has_pycuda = True
except ImportError:
	has_pycuda = False

class spline(object):
	""" spline class.  Used for doing spline interpolation with PyCUDA """
	
	# whether or not the cuda code has been processed by pycuda
	# we don't want this step to be done until needed, because
	# it takes a few seconds
	cuda_loaded = False
	
	# the percision of np.float32
	percision = 1.1920929e-07
	
	##########
	## init ##
	##########
	def __init__ ( self, degree=3, nthreads=512 ):
		""" spline = spline.spline( degree=3, nthreads=512 )
		
		Load a spline object for performing spline interpolation
		with degree of `degree`.
		nthreads = The number of threads to execute per block. """
		
		# nothing to do if we don't support gpu processing
		if not has_pycuda:
			return None
		
		self.nthreads = int( nthreads )
		self.degree = degree
		self.prepare_pole_info( self.degree )
	
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
	
	################
	## pre filter ##
	################
	def pre_filter( self, array, return_gpu=True ):
		""" res = spline.pre_filter( array, return_gpu = True )
		
		Pre-filter a data array to prepare for spline interpolation.
		Returns an allocation object for the array in gpu memory if
		return_gpu = true, otherwise the pre filtered array """
		
		# make sure we are ready to execute
		if not self._prepare_cuda():
			return False
		
		import time
		now = time.time()
		
		# make sure the data array is little endian type
		array = self.force_le( array )
		
		# array size
		( ncols, nrows ) = array.shape
		
		# allocate memory for array
		array_gpu = drv.mem_alloc( array.nbytes )
		
		# and another for the transposed array
		t_array_gpu = drv.mem_alloc( array.nbytes )
		
		# and copy to gpu
		drv.memcpy_htod( array_gpu, array )
		
		# first apply gain.  Might as well apply it with GPU since
		# the array is already there
		blocks = int( np.ceil( float( array.size )/self.nthreads ) )
		self.cuda_apply_gain( array_gpu, np.int64( array.size ), grid=(blocks,1), block=(self.nthreads,1,1) )
		
		# now filter.  Rows are filtered individually.  Therefore
		# Each GPU thread will filter one row
		blocks = int( np.ceil( float( nrows )/self.nthreads ) )
		self.cuda_pre_filter_rows( array_gpu, np.int32( nrows ), np.int32( ncols ), grid=(blocks,1), block=(self.nthreads,1,1) )
		
		## now we need to transpose the array to filter in the other direction
		#blocks = int( np.ceil( float( array.size )/self.nthreads ) )
		#self.cuda_transpose( array_gpu, t_array_gpu, np.int32( nrows ), np.int32( ncols ), np.int32( array.size ), grid=(blocks,1), block=(self.nthreads,1,1) )
		
		## apply gain again.  Must be applied at both steps
		#blocks = int( np.ceil( float( array.size )/self.nthreads ) )
		#self.cuda_apply_gain( t_array_gpu, grid=(blocks,1), block=(self.nthreads,1,1) )
		
		## and run pre_filter on the transposed array
		#blocks = int( np.ceil( float( ncols )/self.nthreads ) )
		#self.cuda_pre_filter_rows( t_array_gpu, np.int32( ncols ), np.int32( nrows ), grid=(blocks,1), block=(self.nthreads,1,1) )
		
		## finally transpose back
		#blocks = int( np.ceil( float( array.size )/self.nthreads ) )
		#self.cuda_transpose( t_array_gpu, array_gpu, np.int32( ncols ), np.int32( nrows ), np.int32( array.size ), grid=(blocks,1), block=(self.nthreads,1,1) )
		
		# return gpu array
		if return_gpu:
			return array_gpu
		
		# otherwise fetch the array back out of GPU memory
		output = np.empty_like( array )
		drv.memcpy_dtoh( output, array_gpu )
		array_gpu.free()
		
		print time.time()-now
		
		# and return
		return output
	
	##################
	## prepare cuda ##
	##################
	def _prepare_cuda( self ):
		""" gpu._prepare_cuda()
		
		Compile the cuda code for spline interpolation """
		
		# can we actually run pycuda?
		if not has_pycuda:
			return False
		
		# has this been loaded yet?
		if self.cuda_loaded:
			return True
		
		# update code with pole info
		my_cuda = cuda
		for (key,val) in self.pole_info.iteritems():
			my_cuda = my_cuda.replace( '@@%s@@' % key.upper(), val )
		
		# load it
		self.cuda_mod = SourceModule( my_cuda )
		
		# and fetch individual functions
		self.cuda_apply_gain = self.cuda_mod.get_function( 'apply_gain' )
		self.cuda_pre_filter_rows = self.cuda_mod.get_function( 'pre_filter_rows' )
		self.cuda_transpose = self.cuda_mod.get_function( 'transpose' )
		
		# update status
		self.cuda_loaded = True
		
		# and return
		return True
	
	###################
	## get pole info ##
	###################
	def prepare_pole_info( self, degree ):
		""" spline.prepare_pole_info( degree ):
		
		Prepares a dict with CUDA code to replace into source before compiling.
		Parts of the CUDA are constants which depend on the spline
		degree.  This is best accomplished by precalculating these
		parts in python so it can all be compiled as constants in CUDA """
		
		if degree == 2:
			npoles = 1
			poles = np.array( [np.sqrt(8.0) - 3.0] )
		elif degree == 3:
			npoles = 1
			poles = np.array( [np.sqrt(3.0) - 2.0] )
		elif degree == 4:
			npoles = 2
			poles = np.array([	np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
						np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0] )
		elif degree == 5:
			npoles = 2
			poles = np.array([	np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0,
						np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0)) - np.sqrt(105.0 / 4.0) - 13.0 / 2.0] )
		elif degree == 6:
			npoles = 3
			poles = np.array([	-0.48829458930304475513011803888378906211227916123938,
						-0.081679271076237512597937765737059080653379610398148,
						-0.0014141518083258177510872439765585925278641690553467])
		elif degree == 7:
			npoles = 3
			poles = np.array([	-0.53528043079643816554240378168164607183392315234269,
						-0.12255461519232669051527226435935734360548654942730,
						-0.0091486948096082769285930216516478534156925639545994])
		elif degree == 8:
			npoles = 4
			poles = np.array([	-0.57468690924876543053013930412874542429066157804125,
						-0.16303526929728093524055189686073705223476814550830,
						-0.023632294694844850023403919296361320612665920854629,
						-0.00015382131064169091173935253018402160762964054070043])
		elif degree == 8:
			npoles = 4
			poles = np.array([	-0.60799738916862577900772082395428976943963471853991,
						-0.20175052019315323879606468505597043468089886575747,
						-0.043222608540481752133321142979429688265852380231497,
						-0.0021213069031808184203048965578486234220548560988624])
		else:
			raise ValueError( 'Spline degree must be an integer with 2 <= n <= 9' )
		
		# store gain
		self.gain = ((1-poles)*(1-1/poles)).prod()
		
		# make pole info dict
		self.pole_info = {}
		
		# and populate
		self.pole_info['npoles'] = '%d' % npoles
		self.pole_info['gain'] = '%.15e' % self.gain
		self.pole_info['tolerance'] = '%.15e' % self.percision
		
		self.pole_info['poles_define'] = ''
		for (i,pole) in enumerate( poles ):
			self.pole_info['poles_define'] += "pole[%d] = %.15e;\n" % (i,pole)
	
cuda = """
__global__ void apply_gain( double *data, long size )
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i >= size) return;
	data[i] *= @@GAIN@@f;
}

__global__ void transpose( double *data, double *output, int rows, int cols, long size )
{
	const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= size) return;
	int row = i/cols;
	int col = i % cols;
	output[col*rows+row] = data[i];
}

__global__ void pre_filter_rows( double *data, int rows, int cols )
{
	/* row to process */
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
	if (row >= rows) return;
	
	/* tolerance */
	double tolerance = @@TOLERANCE@@f;
	int horizon;
	double zn;
	
	/* pole array (based on spline degree) */
	int npoles = @@NPOLES@@;
	double pole[@@NPOLES@@];
	
	/* and define */
	@@POLES_DEFINE@@

	/* loop over poles */
	for (int k = 0; k <= npoles; k++)
	{
		/**** causal initialization ****/
		/* calculate horizon */
		horizon = (int) ceil( log(tolerance) / log( fabs(pole[k]) ) );
		zn = pole[k];
		for ( int n = 1; n < horizon; n++ )
		{
			data[cols*row] += zn * data[cols*row+n];
			zn *= pole[k];
		}
		
		/* causal recursion */
		for ( int n = 1; n < cols; n++ )
		{
			data[cols*row+n] += pole[k] * data[cols*row+(n-1)];
		}
		
		/* anticausal initialization */
		data[cols*row+(cols-1)] = ( (pole[k] / (pole[k] * pole[k] - 1.0)) * (pole[k] * data[cols*row+(cols-2)] + data[cols*row+(cols-1)]) );
		
		/* anticausal recursion */
		for ( int n = cols-2; 0 <= n; n-- )
		{
			data[cols*row+n] = pole[k] * (data[cols*row+(n+1)] - data[cols*row+n]);
		}
	}
}
"""
test = """

import pygfit.spline
import numpy as np
test = np.random.rand( 100, 100 )
spline = pygfit.spline.spline()
spline._prepare_cuda()
res = spline.pre_filter( test, return_gpu=False )


"""