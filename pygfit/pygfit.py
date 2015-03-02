#!/usr/bin/env python

""" PYthon Galaxy FIT: a tool for modeling low resolution images with the aid of high resolution galaxy models.

"""

import log,os,utils,pyfits,time,re,datetime,pygsti,shutil,traceback
import gblend
import numpy as np
import gpu

# try to import multiprocessing
try:
	import multiprocessing
	no_multiprocess = False
except ImportError:
	no_multiprocess = True

class pygfit(object):
	""" PYthon Galaxy FIT: a tool for modeling low resolution images with the aid of high resolution galaxy models.
	
	pygfit = pygfit( config_file='pygfit.config', log_file='pygfit.log', warn_file='pygfit.warn' ) """
	config = {}		# configuration
	timer = {}		# timer

	# border for selecting low and high resolution objects (degrees)
	border = 20.0/3600.0

	# low resolution stuff
	lres_img_file = ''	# filename of low resolution image
	lres_rms_file = ''	# filename of low resolution rms image
	lres_seg_file = ''	# filename of low resolution segmentation image
	lres_cat_file = ''	# filename of low resolution source extractor catalog
	lres_psf_file = ''	# filename of low resolution psf image
	lres_res_file = ''	# filename of low resolution residuals output file
	lres_back_file = ''	# filename of low resolution background image
	lres_img = []		# low resolution image
	lres_rms = []		# low resolution rms image
	lres_seg = []		# low resolution segmentation image
	lres_res = []		# low resolution residuals image
	lres_cat = {}		# low resolution catalog
	lres_align_cat = {}	# low resolution alignment catalog
	lres_psf = []		# low resolution psf image
	lres_back = []		# low resolution background image
	lres_border = {}	# min/max ra/dec for low resolution catalog (object range + self.border)
	lres_img_header = []	# header for the low resolution image
	lres_columns = []	# column data from source extractor catalog
	nlres = 0		# number of objects in the low resolution catalog (mag filtered)
	nlres_align = 0		# number of objects in the low resolution alignment catalog (mag filtered)

	# high resolution stuff
	hres_cat = {}		# high resolution catalog
	hres_border = {}	# min/max ra/dec for high resolution catalog (object range + self.border)
	hres_ra_shift = 0.0	# high resolution right ascension shift (low-high)
	hres_dec_shift = 0.0	# high resolution declination shift (low-high)
	rot_angle_diff = 0.0	# difference between rotation angles of low and high resolution images (low-high)
	y_offset = 0.0		# y offset between low and high resolution image (high-low) in arcseconds
	x_offset = 0.0		# x offset between low and high resolution image (high-low) in arcseconds
	use_point_catalog = False # whether or not to use a point source catalog

	# gblend objects
	gblends = {}		# dictionary of gblend objects for fitting
	good = {}		# dictionary specifying if each gblend is useable
	gblend_ids = []		# list of ids for gblend objects
	current_gblend = -1	# index for iterator

	# flags about object state
	extractor_done = False	# whether or not source extractor has been run
	lres_loaded = False	# whether or not the low resolution data has been loaded
	hres_loaded = False	# whether or not the high resolution data has been loaded
	cats_aligned = False	# whether or not the catalogs have been aligned
	gblends_loaded = False	# whether or not the gblend objects have been created
	is_fit = False		# whether or not the gblends have been fit

	# whether or not python supports multiprocessing
	can_multiprocess = False

	# wcs info from lres image header
	wcs = {}
	wcs_loaded = False

	# input/ouput checks for configuration
	# allowed output columns
	output_columns = ['lres_id','hres_id','nblend','nearest','nearest_mag','model','ra','dec','x','y','img_x','img_y','mag','mag_image','mag_initial','mag_hres','mag_brightest','mag_warning','flux','total_flux','total_mag','blend_fraction','sky','re_hres','re_lres','re_arcsecs','n','pa','ba','chisq_nu','chisq','nf','segmentation_mag','segmentation_residuals','segmentation_fraction','mask_mag','mask_residuals','mask_fraction']
	# required columns for source extractor to output
	required_extractor_columns = ['number','xpeak_image','ypeak_image','xwin_world','ywin_world','mag_auto','flags']
	# all columns for source extractor to output
	extractor_columns = []
	# number of mag_aper columns to expect from source extractor
	number_aperture_mags = 0
	# required configuration fields
	required = ['extractor_config','extractor_catalog','extractor_params','hres_catalog','hres_rotangle','hres_pixscale','lres_image','lres_rms','lres_psf','lres_magzero','min_mag','max_mag','align_min_mag','align_max_mag','pad_length']
	# default configuration fields
	default = {'extractor_cmd': 'sex', 'skip_extractor': True, 'model_type': 0, 'id': 1, 'ra': 2, 'dec': 3, 'mag': 4, 're': 5, 'n': 6, 'pa': 7, 'ba': 8, 'n_threads': 1, 'image_dir': 'cutouts/img', 'rms_dir': 'cutouts/rms', 'segmentation_dir': 'cutouts/seg', 'segmentation_mask_dir': 'cutouts/mask', 'model_dir': 'models', 'output_all_models': True, 'output_catalog': 'pygfit.cat', 'output_format': 'ascii', 'subtract_background': False, 'global_max_shift': 2.0, 'max_shift': 0.2, 'include_errors': False, 'max_array_size': 1e7, 'fit_sky': False, 'n_align': 25, 'gpu': False, 'gpu_nthreads': 512, 'use_integration': True}
	# allowed checkplot types
	check_plot_types = ['isolated','alignment','alignment_mag']

	# sim directory
	sim_dir = ''

	##############################
	## initialize pygfit object ##
	##############################
	def __init__( self, config_file='pygfit.config', log_file='pygfit.log', warn_file='pygfit.warn', logging=True ):
		""" pygfit( config_file='pygfit.config', log_file='pygfit.log', warn_file='pygfit.warn', logging=True )
		
		Returns an object for running pygfit """

		# load configuration
		self.load_config( config_file )

		# start up the log files
		if logging:
			cwd = os.getcwd()
			if cwd[-1] != os.sep: cwd += os.sep
			log.enable_logging()
			log.start( '%s%s' % (cwd,log_file), '%s%s' % (cwd,warn_file) )
			log.log( 'LOG of %s at %s' % (config_file,str(datetime.datetime.today())) )
			log.log( 'Run at path: %s' % os.getcwd() )
			log.warn( 'Warnings and errors for %s at %s' % (config_file,str(datetime.datetime.today())), silent=True )
		else:
			log.disable_logging()

	# return iterator for looping through gblends
	def __iter__(self):
		self.current_gblend = -1
		return self

	# return next gblend in iterator
	def next(self):

		# iteration is done with a list and a dictionary
		# the objects themselves are stored in a dictionary, but the list is used to preserve object order
		# the latter is important to make sure they are looped through in order of decreasing luminosity

		# return only good gblends
		while True:
			self.current_gblend += 1
			if self.current_gblend == len( self.gblend_ids ): raise StopIteration

			if self.good[self.gblend_ids[self.current_gblend]]: break

		return self.gblends[self.gblend_ids[self.current_gblend]]

	#####################
	## timer functions ##
	#####################
	def start_timer( self, name='default' ):
		self.timer[name] = time.time()

	def get_timer( self, name='default' ):
		return time.time()-self.timer[name]

	########################
	## load configuration ##
	########################
	def load_config( self, config_file ):
		""" pygfit.load_config( config_file ) """

		# special case - default configuration
		if type( config_file ) == type( True ) and config_file:
			self.config = self.load_default_config()
		else:
			# make sure config file exists
			if not os.path.isfile( config_file ) or not os.path.isfile( config_file ): raise ValueError( "The specified config file was not found!  Try 'pygfit.py --help' for usage instructions." )
			self.config = utils.load_config( config_file )

		self.config['checks'] = {}

		# check for required fields (set in class definition)
		for field in self.required:
			if not self.config.has_key( field ): raise ValueError( 'Required field, %s, missing from configuration!' % field )

		# set default fields
		for key in self.default: self.config.setdefault( key, self.default[key] )
		# point_mag will default to the mag field if it doesn't exist
		self.config.setdefault( 'point_mag', self.config['mag'] )

		# convert some things to the proper type.  Most are caught automatically by load_config, but these I want to be ints.
		self.config['pad_length'] = int( float( self.config['pad_length'] ) )
		self.config['n_align'] = int( float( self.config['n_align'] ) )
		self.config['n_threads'] = int( self.config['n_threads'] )
		self.config['gpu_nthreads'] = int( self.config['gpu_nthreads'] )

		# was GPU usage requested?
		if self.config['gpu']:
			# if so, see if it is actually supported...
			if not gpu.has_pycuda:
				log.warn( 'This python distribution does not support GPU processing.  This is accomplished with the pycuda package, which could not be loaded.' )
				self.config['gpu'] = False
			# if gpu is supported then initialize a gpu object and store it in the config
			else:
				self.config['gpu'] = gpu.gpu( self.config['gpu_nthreads'] )

		# check for multiprocessing (no_multiprocess is set at the top of this script near the imports)
		if no_multiprocess:
			if self.config['n_threads'] != 1:
				log.warn( 'This python distribution does not support multiprocessing.  This is done with the multiprocessing package, which was introduced in python v2.6.  Multithreading has been disabled for this pygfit run.' )
				self.config['n_threads'] = 1

		# set n_threads = n_cpus if n_threads == 0
		if self.config['n_threads'] == 0: self.config['n_threads'] = multiprocessing.cpu_count()

		# check output columns
		remove = []
		if self.config.has_key( 'output_columns' ):
			# make sure output columns is a list
			if type( self.config['output_columns'] ) != type( [] ): self.config['output_columns'] = [self.config['output_columns']]

			# convert requested columns to lowercase
			for i in range( len( self.config['output_columns'] ) ):

				self.config['output_columns'][i] = self.config['output_columns'][i].lower()

				# make sure this output column is allowed
				if not self.output_columns.count( self.config['output_columns'][i] ):
					log.warn( 'Column %s is not a recognized output column - skipping' % self.config['output_columns'][i] )
					remove.append( self.config['output_columns'][i] )

		for item in remove: self.config['output_columns'].remove( item )

		# default output columns
		if not self.config.has_key( 'output_columns' ) or len( self.config['output_columns'] ) == 0: self.config['output_columns'] = self.output_columns

		# build list of source extractor columns.  First copy required columns
		self.extractor_columns = list( self.required_extractor_columns )
		# then include anything in config
		if self.config.has_key( 'extractor_columns' ):
			if type( self.config['extractor_columns'] ) != type( [] ): self.config['extractor_columns'] = [self.config['extractor_columns']]
			for col in self.config['extractor_columns']:
				if not self.extractor_columns.count( col.lower() ): self.extractor_columns.append( col.lower() )

		# mag_aper and magerr_aper have to be handled a bit differently depending on the number of apertures specified.
		# so, load the source extractor configuration to figure out how many columns were specified
		if self.extractor_columns.count( 'mag_aper' ) or self.extractor_columns.count( 'magerr_aper' ):
			extractor_config = utils.load_config( self.config['extractor_config'] )
			# error handling if no phot_apertures are specified in the source extractor config
			if not extractor_config.has_key( 'phot_apertures' ):
				if self.extractor_columns.count( 'mag_aper' ): self.extractor_columns.remove( 'mag_aper' )
				if self.extractor_columns.count( 'magerr_aper' ): self.extractor_columns.remove( 'magerr_aper' )
				log.warn( 'MAG_APER output was requested but no PHOT_APERTURES were specified in the source extractor configuration!' )
			else:
				self.number_aperture_mags = len( extractor_config['phot_apertures'] ) if type( extractor_config['phot_apertures'] ) == type( [] ) else 1

		# make check plot types lower case for easy comparison.  Also make sure valid checkplots are selected
		if self.config.has_key( 'check_plots' ) and self.config.has_key( 'check_plot_files' ):
			# make sure check plots is a list
			if type( self.config['check_plots'] ) != type( [] ): self.config['check_plots'] = [self.config['check_plots']]
			if type( self.config['check_plot_files'] ) != type( [] ): self.config['check_plot_files'] = [self.config['check_plot_files']]

			# make sure we have the same number of checkplots and checkplot files
			if len( self.config['check_plots'] ) != len( self.config['check_plot_files'] ):
				log.log( 'Number of check plots and check plot files do not match!  Quitting!' )
				raise ValueError( 'Number of check plots and check plot files do not match!' )

			# convert to lower case and verify it is allowed
			for i in range( len( self.config['check_plots'] ) ):

				name = self.config['check_plots'][i].lower()

				# make sure this checkplot is allowed
				if not self.check_plot_types.count( name ):
					log.warn( 'Check plot %s is not a recognized check plot - skipping' % name )
					continue

				# store in list of checks
				self.config['checks'][name] = self.config['check_plot_files'][i]

		elif self.config.has_key( 'check_plots' ) or self.config.has_key( 'check_plot_files' ):
			# user must specify both check plots and check plot files
			log.log( 'You must specify both check_plots and check_plot_files to ouput any check plots!  Quitting.' )
			raise ValueError( 'You must specify both check_plots and check_plot_files to ouput any check plots!' )

		# copy some of the configuration parameters into the object (for convenience)
		self.lres_img_file = self.config['lres_image']
		self.lres_rms_file = self.config['lres_rms']
		self.lres_psf_file = self.config['lres_psf']

		# and now set some filenames according to naming convention
		if self.lres_img_file.find( '.fits' ) >= 0:
			self.lres_back_file = self.lres_img_file.replace( '.fits', '_back.fits' )
			self.lres_seg_file = self.lres_img_file.replace( '.fits', '_seg.fits' )
			self.lres_cat_file = self.lres_img_file.replace( '.fits', '.cat' )
			self.lres_res_file = self.lres_img_file.replace( '.fits', '_res.fits' )
		else:
			self.lres_back_file = "%s_back.fits" % self.lres_img_file
			self.lres_seg_file = "%s_seg.fits" % self.lres_img_file
			self.lres_cat_file = "%s.cat" % self.lres_img_file
			self.lres_res_file = "%s_res.fits" % self.lres_img_file
		if self.config.has_key( 'extractor_catalog' ): self.lres_cat_file = self.config['extractor_catalog']
		self.lres_reg = self.lres_cat_file.replace( '.cat', '.reg' )

		# high resolution catalog output region files
		self.config['hres_catalog_reg'] = utils.swap_extension( self.config['hres_catalog'], '.cat', '.reg' )

		# simulation directory
		self.sim_dir = self.config['sim_dir'] if self.config.has_key( 'sim_dir' ) else 'sims'

	def load_default_config( self ):
		""" pygfit.load_default_config()
		
		Very basic configuration example for outputting configuration file """

		config = { 'extractor_config': 'extractor.config', 'extractor_params': 'extractor.param', 'extractor_catalog': 'extractor.cat', 'hres_catalog': 'hres.cat', 'hres_rotangle': 0.0, 'hres_pixscale': 1.388889e-05, 'lres_image': 'lres.fits', 'lres_rms': 'lres_rms.fits', 'lres_psf': 'lres_psf.fits', 'lres_magzero': 22.5, 'global_max_shift': 2.0, 'max_shift': 0.2, 'min_mag': 16, 'max_mag': 22, 'align_min_mag': 18, 'align_max_mag': 20, 'pad_length': 25}

		config['check_plots'] = self.check_plot_types
		files = []
		for t in config['check_plots']: files.append( '%s.fits' % t )
		config['check_plot_files'] = files

		return config

	#########
	## Go! ##
	#########
	def go( self, skip_extractor=None, residuals_file=None, output_catalog=None ):
		""" pygfit.go( skip_extractor=None, residuals_file=None, output_catalog=None )
		
		Runs pygfit with the parameters in the given parameter file.
		Set skip_extractor = True/False to override configuration.
		Set residuals_file='filename' to override configuration.
		Set output_catalog='filename' to override configuration. """

		# track total time for run
		self.start_timer( 'full_run' )

		# first, run source extractor
		self.run_extractor( skip=skip_extractor )

		# load the low resolution catalog/images
		self.load_low_resolution_data()

		# load the high resolution catalog
		self.load_high_resolution_data()

		# align catalogs
		self.align_catalogs()

		# prepare the gblend objects
		self.prepare_gblends()

		# fit!
		self.fit()

		# generate output images
		self.generate_output_images( residuals_file=residuals_file )

		# and output the final catalog
		self.output_catalog( output_catalog=output_catalog )

		#log.log( '\nTime spent keeping track of time spent %f seconds' % np.random.normal( 300.0, 100.0 ) )

		# record total time
		log.log( '\npygfit finished in %f seconds' % self.get_timer( 'full_run' ) )

	##########################
	## run source extractor ##
	##########################
	def run_extractor( self, skip=None, force=False ):
		""" pygfit.run_extractor( skip=False )
		
		Runs source extractor on low resolution image.
		If skip=True and all the necessary source extractor outputs already
		exist, then source extractor will not be run again (but will be logged) """

		# nothing to do if we've already run source extractor 
		if self.extractor_done and not force: return True

		# log stuff
		self.start_timer()
		log.log( '\nStarting source extractor' )

		# should we skip running source extractor if the output files already exist?
		if skip or ( self.config['skip_extractor'] and skip is None ):
			if os.path.isfile( self.lres_back_file ) and os.path.isfile( self.lres_seg_file ) and os.path.isfile( self.lres_cat_file ): skip = True

		# first output the parameter file
		log.log( 'Generating source extractor parameters file %s' % self.config['extractor_params'] )
		# generate string for parameters file
		params = ''
		for col in self.extractor_columns: params += "%s\n" % col.upper()
		# update MAG_APER parameter line if it was specified and there is more than one aperture mag
		if self.number_aperture_mags > 1:
			params = params.replace( 'MAG_APER', 'MAG_APER(%d)' % self.number_aperture_mags )
			params = params.replace( 'MAGERR_APER', 'MAGERR_APER(%d)' % self.number_aperture_mags )

		# open parameters file and write it out
		fp = open( self.config['extractor_params'], 'wb' )
		# write out all columns
		fp.write( params )
		fp.close()

		# now run and log source extractor
		cmd = "%s %s -c %s -CATALOG_NAME %s -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE %s -PARAMETERS_NAME %s -CATALOG_TYPE FITS_LDAC -CHECKIMAGE_TYPE 'BACKGROUND,SEGMENTATION' -CHECKIMAGE_NAME '%s,%s' -MAG_ZEROPOINT %f" % (self.config['extractor_cmd'], self.lres_img_file, self.config['extractor_config'], self.lres_cat_file, self.lres_rms_file, self.config['extractor_params'], self.lres_back_file, self.lres_seg_file, self.config['lres_magzero'])
		if not skip:
			os.system( cmd )
		else:
			log.log( 'Skipping actual source extractor run' )
		log.log( cmd )

		# now generate a region file for the source extractor detections
		if not skip:
			log.log( 'Generating region file %s from source extractor output' % self.lres_reg )
			fits = pyfits.open( self.lres_cat_file )
			cat = fits[2].data
			lines = ['# Region file format: DS9 version 4.0','image']
			for i in range( cat.size ): lines.append( "circle(%d,%d,5) # text={%d}" % (cat.field('xpeak_image')[i],cat.field('ypeak_image')[i],cat.field('number')[i]) )
			fp = open( self.lres_reg, 'wb' )
			fp.write( "\n".join( lines ) )

		# record that source extractor has executed
		self.extractor_done = True

		log.log( 'Source Extractor done in %f seconds' % self.get_timer() )

	#################################
	## load low resolution catalog ##
	#################################
	def load_low_resolution_data( self, force=False ):
		""" pygfit.load_low_resolution_data()
		
		Loads the low resolution images/catalog. """

		# first make sure source extractor has run
		self.run_extractor()

		# nothing to do if we have alreadly loaded everything
		if self.lres_loaded and not force: return True

		self.start_timer()
		log.log( '\nLoading low resolution images/data' )

		# load in the various low resolution images
		(self.lres_img,self.lres_img_header) = utils.load_image( self.lres_img_file, header=True )
		self.lres_rms = utils.load_image( self.lres_rms_file )
		self.lres_seg = utils.load_image( self.lres_seg_file )
		self.lres_back = utils.load_image( self.lres_back_file )

		# load psf - normalize and make odd-sized and square
		self.lres_psf = utils.load_psf( self.lres_psf_file )

		# retrieve the wcs info out of the header
		self.load_wcs()

		# subtract off background
		self.lres_img -= self.lres_back

		# copy the low resolution image to the residuals image - this will be updated as fits finish
		self.lres_res = self.lres_img.copy()

		# set rms to a very low value where zero
		self.lres_rms = np.where( self.lres_rms == 0, 1e-20, self.lres_rms )

		# now load in the actual catalog, filter according to min/max magnitude, and sort by decreasing magnitude
		fits = pyfits.open( self.lres_cat_file )
		# store the column information, which will be used later when outputting catalog
		self.lres_columns = fits[2].columns
		log.log( 'Found %d objects in the low resolution catalog' % fits[2].data.size )
		sinds = fits[2].data.field('mag_auto').argsort()

		# store the low resolution catalog
		m = (fits[2].data.field('mag_auto')[sinds] < self.config['max_mag']) & (fits[2].data.field('mag_auto')[sinds] > self.config['min_mag'])
		self.nlres = m.sum()
		log.log( 'Keeping %d objects found inside the low resolution catalog in magnitude range (%f - %f)' % (self.nlres,self.config['min_mag'],self.config['max_mag']) )
		# store the sorted catalog and keep the objects in the specified mag range
		self.lres_cat = fits[2].data[sinds[m]].copy()

		if self.nlres == 0:
			log.log( 'No objects were found in the low resolution catalog within the specified mag range!  Quitting.' )
			raise ValueError( 'No objects were found in the low resolution catalog within the specified mag range!' )

		# now store the alignment catalog
		m = (fits[2].data.field('mag_auto')[sinds] < self.config['align_max_mag']) & (fits[2].data.field('mag_auto')[sinds] > self.config['align_min_mag'])
		self.nlres_align = m.sum()
		log.log( 'Keeping %d objects found inside the low resolution catalog alignment magnitude range (%f - %f)' % (m.sum(),self.config['align_min_mag'],self.config['align_max_mag']) )
		# copy the fields we want into the object catalog, which will be a dictionary (for consistency with self.hres_cat)
		self.lres_align_cat = fits[2].data[sinds[m]].copy()

		if self.nlres_align == 0:
			log.log( 'No objects were found in the low resolution catalog within the specified alignment mag range!  Quitting.' )
			raise ValueError( 'No objects were found in the low resolution catalog within the specified alignment mag range!' )

		# now measure min/max ra/dec for the low resolution catalog (both regular and alignment)
		self.lres_border = {	'minra':np.min( [self.lres_cat['xwin_world'].min(),self.lres_align_cat['xwin_world'].min()] ),
					'maxra':np.max( [self.lres_cat['xwin_world'].max(),self.lres_align_cat['xwin_world'].max()] ),
					'mindec':np.min( [self.lres_cat['ywin_world'].min(),self.lres_align_cat['ywin_world'].min()] ),
					'maxdec':np.max( [self.lres_cat['ywin_world'].max(),self.lres_align_cat['ywin_world'].max()] ) }
		# add a little bit of a border on either side
		self.lres_border['minra'] -= self.border
		self.lres_border['maxra'] += self.border
		self.lres_border['mindec'] -= self.border
		self.lres_border['maxdec'] += self.border

		# record that the low resolution data has been loaded
		self.lres_loaded=True

		log.log( 'Low resolution images/data loaded in %f seconds' % self.get_timer() )

	##################################
	## load high resolution catalog ##
	##################################
	def load_high_resolution_data( self, force=False ):
		""" pygfit.load_high_resolution_data()
		
		Loads the high resolution catalog. """

		# first make sure that the low resolution data has been loaded
		self.load_low_resolution_data()

		# nothing to do if we have already run once
		if self.hres_loaded and not force: return True

		self.start_timer()
		log.log( '\nLoading high resolution catalog' )

		# load catalog
		( catalog, valid, unknown ) = utils.load_data_catalog( self.config['hres_catalog'], self.config )

		# limit high resolution catalog to what is inside the ra/dec range
		w = np.where( (catalog['ra'] >= self.lres_border['minra']) & (catalog['ra'] <= self.lres_border['maxra']) & (catalog['dec'] >= self.lres_border['mindec']) & (catalog['dec'] < self.lres_border['maxdec']) )[0]
		outside = valid - len( w )

		# trim down things outside the range
		if outside:
			for field in catalog.keys(): catalog[field] = catalog[field][w]

		# and store
		self.hres_cat = catalog

		log.log( 'Processed %d data lines in the high resolution catalog' % valid )
		log.log( '%d object(s) were outside the ra/dec range of the low resolution catalog' % outside )
		if unknown > 0: log.log( '%d object(s) were skipped because they had an unrecognized model type' % unknown )
		log.log( 'Kept %d objects from the high resolution catalog' % len( catalog['id'] ) )

		if len( catalog['id'] ) == 0:
			log.log( 'No useable data was found in the high redshift catalog!  Quitting.' )
			raise ValueError( 'No useable data was found in the high redshift catalog! Check log file for details' )

		# convert the high resolution ra/dec to low resolution x/y
		(self.hres_cat['x'], self.hres_cat['y']) = self.rd2xy( self.hres_cat['ra'], self.hres_cat['dec'] )

		# convert the sizes from pixels to arcseconds
		self.hres_cat['re'] *= self.config['hres_pixscale']*3600.0
		if self.hres_cat.has_key( 're_err' ): self.hres_cat['re_err'] *= self.config['hres_pixscale']*3600.0
		# and to pixels of the low resolution image
		self.hres_cat['re_pix'] = self.hres_cat['re']/self.wcs['scale']
		if self.hres_cat.has_key( 're_err' ): self.hres_cat['re_pix_err'] = self.hres_cat['re_err']/self.wcs['scale']

		# now calculate the position angle difference between the high resolution image and low resolution image
		self.rot_angle_diff = np.degrees( self.wcs['rot_angle'] ) - self.config['hres_rotangle']
		# and correct the position angles of all the high resolution objects accordingly
		self.hres_cat['pa'] += self.rot_angle_diff
		# keep -180 < pa < 180
		m = self.hres_cat['pa'] > 180
		if m.any(): self.hres_cat['pa'][m] -= 360
		m = self.hres_cat['pa'] < -180
		if m.any(): self.hres_cat['pa'][m] += 360

		# region containing high resolution objects
		self.hres_border = {	'minra': self.hres_cat['ra'].min() - self.border,
					'maxra': self.hres_cat['ra'].max() + self.border,
					'mindec': self.hres_cat['dec'].min() - self.border,
					'maxdec': self.hres_cat['dec'].max() + self.border }

		# generate a region file for the high resolution catalog
		utils.write_region( self.config['hres_catalog_reg'], self.hres_cat['ra'], self.hres_cat['dec'], self.hres_cat['re']/3600 )

		self.hres_loaded = True

		log.log( 'High resolution catalog loaded in %f seconds' % self.get_timer() )

	####################
	## align catalogs ##
	####################
	def align_catalogs( self, force=False ):
		""" pygfit.align_catalogs()
		
		Run pygfit on a limited subset of low resolution objects with only one high resolution match
		to calculate offset between high resolution and low resolution catalogs.
		"""

		# first load the high resolution data
		self.load_high_resolution_data()

		if self.cats_aligned and not force: return True

		self.start_timer()
		log.log( '\nAligning high and low resolution catalogs' )

		# if alignment info was included in the config then apply it and we are done
		if self.config.has_key( 'x_offset' ) and self.config.has_key( 'y_offset' ):
			self.x_offset = self.config['x_offset']
			self.y_offset = self.config['y_offset']
			self.hres_cat['x'] += self.x_offset
			self.hres_cat['y'] += self.y_offset
			log.log( 'Using stored alignment (low-high) of (%f,%f) low resolution pixels (x,y)' % (self.x_offset,self.y_offset) )
			log.log( 'Using stored alignment (low-high) of (%f,%f) arcseconds (x,y)' % (self.x_offset*self.wcs['scale'],self.y_offset*self.wcs['scale']) )
			self.cats_aligned = True
			return True

		# now prepare gblend objects for everything in the lres alignment catalog, up to the desired number of kept objects
		kept = 0
		fit = 0
		for i in range( self.nlres_align ):

			# make sure there is actually something near this low resolution object before loading it (since that can take some time)
			if ( self.lres_align_cat['xwin_world'][i] < self.hres_border['minra'] ) or ( self.lres_align_cat['xwin_world'][i] > self.hres_border['maxra'] ) or ( self.lres_align_cat['ywin_world'][i] < self.hres_border['mindec'] ) or ( self.lres_align_cat['ywin_world'][i] > self.hres_border['maxdec'] ): continue

			# fetch gblend
			gblend_obj = self.get_gblend( self.lres_img, self.lres_align_cat[i], self.config['global_max_shift'] )

			# will return false on failure
			if type( gblend_obj ) == type( False ): continue

			# skip this unless there is exactly one matching high resolution object
			if gblend_obj.nhres != 1: continue

			# also make sure it has easy to fit sersic values
			model = gblend_obj.hres_models[0]
			if (model.model_type == 'sersic') and ( (model.n > 7.0) or (model.ba < 0.1) ): continue

			# found a winner!
			kept += 1

			# generate models - this will be run automatically with a call to gblend_obj.fit(),
			# but by calling it manually I can supress outputting fits files
			success = gblend_obj.generate_models( output_fits=False )

			# returns a traceback string if failed.  If so skip
			if type( success ) == type( '' ):
				log.warn( 'Failed to prepare models for lres id number: %d.\nTraceback:\n%s' % (gblend_obj.number,success) )
				continue

			# fit!
			( code, best_fit ) = gblend_obj.fit()

			# returns a traceback string if failed.  If so skip
			if type( best_fit ) == type( False ) and best_fit == False:
				log.warn( 'Failed to fit lres id number: %d.\nTraceback:\n%s' % (gblend_obj.number,code) )
				break
				continue

			# record successful fit
			fit += 1

			# retrieve position offsets
			offsets = gblend_obj.get_position_offsets()

			# and store along with mag/ra/dec
			offsets = np.concatenate( ([self.lres_align_cat['mag_auto'][i]], [self.lres_align_cat['xwin_world'][i]], [self.lres_align_cat['ywin_world'][i]], offsets) )
			if fit == 1:
				all_offsets = np.array( offsets )
			else:
				all_offsets = np.vstack( (all_offsets,offsets) )

			# all done if we have reached n_align objects
			if fit == self.config['n_align']: break

			if kept > fit + 20:
				log.warn( 'Too many failed fits for alignment step - quitting' )
				break

		if fit < 2:
			message = 'Did not find enough suitable alignment objects!  Found %d alignment objects and fit %d.' % (kept,fit)
			if kept < 2: message += '  Try extending alignment magnitude range.'
			message += '  Skipping alignment step.'
			log.warn( message )
			log.log( message )
			self.cats_aligned = True
			self.y_offset = 0
			self.x_offset = 0
			return True

		# calculate median x/y offsets
		self.y_offset = np.median( all_offsets[:,3] )
		self.x_offset = np.median( all_offsets[:,4] )

		# apply offsets
		self.hres_cat['x'] += self.x_offset
		self.hres_cat['y'] += self.y_offset

		# generate offset ra/dec for aligned region file
		( ra, dec ) = self.xy2rd( self.hres_cat['x'], self.hres_cat['y'] )
		utils.write_region( self.config['hres_catalog_reg'].replace( '.reg', '_aligned.reg' ), ra, dec, self.hres_cat['re']/3600 )

		# generate a check plot?
		if self.config['checks'].has_key( 'alignment' ):
			( xs, ys ) = self.rd2xy( all_offsets[:,1], all_offsets[:,2] )
			import matplotlib.pyplot as pyplot
			pyplot.subplot( 2,1,1 )
			pyplot.plot( xs, all_offsets[:,4], 'ko' )
			pyplot.axhline( 0, color='k' )
			pyplot.axhline( self.x_offset, linestyle='--', color='b' )
			pyplot.annotate( 'Offset = %f\nScatter = %f' % (self.x_offset,np.std( all_offsets[:,4] )), (0.05,0.85), (0.05,0.85), xycoords='axes fraction', textcoords='axes fraction' )
			pyplot.xlabel( 'X Position' )
			pyplot.ylabel( '$\Delta$ X Position (new-old)' )
			pyplot.subplot( 2,1,2 )
			pyplot.plot( ys, all_offsets[:,3], 'ko' )
			pyplot.axhline( 0, color='k' )
			pyplot.axhline( self.y_offset, linestyle='--', color='b' )
			pyplot.annotate( 'Offset = %f\nScatter = %f' % (self.y_offset,np.std( all_offsets[:,3] )), (0.05,0.85), (0.05,0.85), xycoords='axes fraction', textcoords='axes fraction' )
			pyplot.xlabel( 'Y Position' )
			pyplot.ylabel( '$\Delta$ Y Position (new-old)' )
			pyplot.gcf().set_size_inches( (8.0,10.5) )
			pyplot.savefig( self.config['checks']['alignment'] )
			pyplot.clf()
		if self.config['checks'].has_key( 'alignment_mag' ):
			import matplotlib.pyplot as pyplot
			pyplot.subplot( 2,1,1 )
			pyplot.plot( all_offsets[:,0], all_offsets[:,4], 'ko' )
			pyplot.axhline( 0, color='k' )
			pyplot.axhline( self.x_offset, linestyle='--', color='b' )
			pyplot.annotate( 'Offset = %f\nScatter = %f' % (self.x_offset,np.std( all_offsets[:,4] )), (0.05,0.85), (0.05,0.85), xycoords='axes fraction', textcoords='axes fraction' )
			pyplot.xlabel( 'Mag Auto' )
			pyplot.ylabel( '$\Delta$ X Position (new-old)' )
			pyplot.subplot( 2,1,2 )
			pyplot.plot( all_offsets[:,0], all_offsets[:,3], 'ko' )
			pyplot.axhline( 0, color='k' )
			pyplot.axhline( self.y_offset, linestyle='--', color='b' )
			pyplot.annotate( 'Offset = %f\nScatter = %f' % (self.y_offset,np.std( all_offsets[:,3] )), (0.05,0.85), (0.05,0.85), xycoords='axes fraction', textcoords='axes fraction' )
			pyplot.xlabel( 'Mag Auto' )
			pyplot.ylabel( '$\Delta$ Y Position (new-old)' )
			pyplot.gcf().set_size_inches( (8.0,10.5) )
			pyplot.savefig( self.config['checks']['alignment_mag'] )
			pyplot.clf()

		log.log( 'Fit %d objects to measure high/low resolution catalog alignment in %f seconds' % (fit,self.get_timer()) )
		log.log( 'Measured offsets (low-high) of (%f,%f) low resolution pixels (x,y)' % (self.x_offset,self.y_offset) )
		log.log( 'Measured offsets (low-high) of (%f,%f) arcseconds (x,y)' % (self.x_offset*self.wcs['scale'],self.y_offset*self.wcs['scale']) )

		self.cats_aligned = True

	#####################
	## prepare gblends ##
	#####################
	def prepare_gblends( self, force=False, shift=None ):
		""" pygfit.prepare_gblends( shift=None )
		
		Generates the gblend objects for the fitting process.
		The shift is the maximum allowed position shift (in arcseconds).
		If shift is None then defaults to MAX_SHIFT (from pygfit.config) """

		self.align_catalogs()

		if self.gblends_loaded and not force: return True

		if shift is None: shift = self.config['max_shift']

		self.start_timer()
		log.log( '\nPreparing images for fitting' )

		# first clear out list of gblend objects
		self.gblends = {}
		self.good = {}
		self.gblend_ids = []

		# record a list of everything that was prepared for fitting
		fit_ra = []
		fit_dec = []

		# prepare a gblend object for everything in the low resolution catalog that has high resolution objects
		# this is a multi-step process
		nlres = 0
		nhres = 0
		for i in range( self.nlres ):

			# make sure this is at least near all the high resolution objects...
			if ( self.lres_cat['xwin_world'][i] < self.hres_border['minra'] ) or ( self.lres_cat['xwin_world'][i] > self.hres_border['maxra'] ) or ( self.lres_cat['ywin_world'][i] < self.hres_border['mindec'] ) or ( self.lres_cat['ywin_world'][i] > self.hres_border['maxdec'] ): continue

			# get the gblend object
			gblend_obj = self.get_gblend( self.lres_img, self.lres_cat[i], shift )

			# will return false on failure
			if type( gblend_obj ) == type( False ): continue

			# and if this found high resolution objects, then keep it around
			if gblend_obj.nhres:
				nlres += 1

				# add to the count of matched high resolution objects
				nhres += gblend_obj.nhres

				# store gblend in pygfit object
				self.gblend_ids.append( gblend_obj.number )
				self.gblends[gblend_obj.number] = gblend_obj
				self.good[gblend_obj.number] = True

				# record ra/dec for this object
				fit_ra.append( self.lres_cat['xwin_world'][i] )
				fit_dec.append( self.lres_cat['ywin_world'][i] )

		# generate region file with everything which was fit (they haven't been fit yet, but they will be)
		utils.write_region( self.lres_reg.replace( '.reg', '_fit.reg' ), fit_ra, fit_dec, [self.wcs['scale']*2/3600]*len( fit_ra ) )

		# now generate the models.  This can be done with or without multiprocessing
		log.log( 'Generating models with %d threads' % self.config['n_threads'] )
		self.start_timer( 'modeling' )
		if self.config['n_threads'] > 1:
			self._generate_models_parallel()
		else:
			self._generate_models_sequential()
		log.log( 'Generated models in %f seconds' % self.get_timer( 'modeling' ) )

		# log progress
		if nlres != self.nlres: log.log( '%d low resolution objects did not match anything in the high resolution catalog' % (self.nlres-nlres) )
		log.log( 'Matched up %d high resolution objects to %d low resolution objects' % (nhres,nlres) )
		log.log( 'Prepared %d low resolution objects in %f seconds' % (nlres,self.get_timer()) )

		if nlres == 0:
			log.log( 'No matches were found between high and low resolution catalogs.  Nothing to fit!' )
			raise ValueError( 'No matches were found between high and low resolution catalogs.  Nothing to fit!  Quitting...' )

		# update object state
		self.gblends_loaded = True
		return True

	################################
	## generate models sequential ##
	################################
	def _generate_models_sequential( self ):
		""" pygfit._generate_models_sequential()
		
		Generate the models sequentially, i.e. without parallelization """

		for (i,gblend_obj) in enumerate(self):

			# generate models
			success = gblend_obj.generate_models()

			# returns a traceback string if failed
			if type( success ) == type( '' ):
				log.warn( 'Failed to prepare models for lres id number: %d.\nTraceback:\n%s' % (gblend_obj.number,success) )
				# mark this gblend as bad so that it is skipped in all future iterations
				self.good[gblend_obj.number] = False

	def _generate_models_parallel( self ):
		""" pygfit._generate_models_parallel()
		
		Generate the models in parallel mode """

		# As the gblend objects generate models, they automatically store the results in themselves and use these results for fitting.
		# However, when run in parallel mode a copy of each gblend object is run in a child process, and so the actual gblend object
		# (which is kept in this pygfit object) is not updated.  For this reason we have to use a multprocessing queue.
		# This gives a way for the child processes to return their results, and then I take that result and update the
		# appropriate gblend object with models generated by its own copy ran in a child process.
		results = multiprocessing.Queue()

		completed = 0		# number of gblends which have been successfully modeled
		started = 0		# number of gblends which have been started
		running = 0		# number of gblends currently being modeled
		is_started = False
		# continue looping until all gblends are finished
		while completed < len( self.gblend_ids ):

			# as long as there are free threads and gblends to be modeled, start more processes
			while (running < self.config['n_threads']) & (started < len( self.gblend_ids )):

				# figure out what the next gblend object is
				number = self.gblend_ids[started]

				# start the process, and pass it the results queue
				thread = multiprocessing.Process( target=self.gblends[number].generate_models, args=(results,) )
				thread.start()

				# and update the statistics
				started += 1
				running += 1

			# start pulling out the results from the results queue, and store in the gblend objects.
			while not results.empty():

				# fetch results
				( number, returned ) = results.get()

				# if returned is a string then there was an error, and the string is the traceback
				if type( returned ) == type( '' ):

					self.good[number] = False
					log.warn( 'Failed to prepare models for lres id number: %d.\nTraceback:\n%s' % (number,returned) )

				else:

					# update the appropriate gblend object
					self.gblends[number].set_models( returned )

				# update statistics
				completed += 1
				running -= 1

		# now loop through all the gblends and make sure that they really are all ready
		for gblend_obj in self:
			if not gblend_obj.ready:
				log.log( 'Failed to load low resolution object number %d.  Quitting.' % gblend_obj.number )
				raise ValueError( 'Failed to prepare low resolution object number %d' % gblend_obj.number )

	#########
	## fit ##
	#########
	def fit( self, force=False ):
		""" pygfit.fit()
		
		Generate the gblend objects and fit """

		# first prepare the gblends which fit each low resolution object
		self.prepare_gblends()

		# nothing to do if it is already done
		if self.is_fit and not force: return True

		self.start_timer()
		log.log( '\nStarting fits with %d threads' % self.config['n_threads'] )

		# do the actual fitting - threaded or not
		if self.config['n_threads'] > 1:
			self._fit_parallel()
		else:
			self._fit_sequential()

		# checkplot?
		if self.config['checks'].has_key( 'isolated' ) and self.config['checks']['isolated']:

			# fetch info for checkplot
			mags = np.array( [] )
			mag_autos = np.array( [] )
			sersics = np.array( [] )
			for gblend_obj in self:

				# only interested in things with just one high resolution match
				if gblend_obj.nhres != 1: continue

				# fetch the fit for this gblend
				cat = gblend_obj.get_catalog()
				# as well as lres data from source extractor
				lres = self.get_lres_object( gblend_obj.number, key=True )
				# now store info
				mags = np.append( mags, cat['mag'] )
				mag_autos = np.append( mag_autos, lres['mag_auto'] )
				model = 1 if cat['model'][0] == 'sersic' else 0
				sersics = np.append( sersics, model )

			# and plot!
			if mags.size > 0:
				import matplotlib.pyplot as pyplot
				ms = sersics == 1
				mp = ~ms
				nsm = ms.sum()
				npm = ms.size - nsm
				if nsm > 0: pyplot.plot( mag_autos[ms], mag_autos[ms]-mags[ms], 'ro' )
				if nsm > 1: pyplot.axhline( np.median( mag_autos[ms]-mags[ms] ), color='red' )
				if npm > 0: pyplot.plot( mag_autos[mp], mag_autos[mp]-mags[mp], 'g^' )
				if npm > 1: pyplot.axhline( np.median( mag_autos[mp]-mags[mp] ), color='green' )
				if mags.size > 1:
					pyplot.annotate( 'Offset = %f\nScatter = %f' % (np.median( mag_autos-mags ),np.std( mag_autos-mags )), (0.05,0.90), (0.05,0.90), xycoords='axes fraction', textcoords='axes fraction' )
					pyplot.axhline( np.median( mag_autos-mags ), color='black' )
				pyplot.xlabel( 'Mag Auto' )
				pyplot.ylabel( 'Mag (Galfit) - Mag Auto' )
				pyplot.gcf().set_size_inches( (6.0,6.0) )
				pyplot.savefig( self.config['checks']['isolated'] )

		# all done!
		log.log( 'Fit %d low resolution objects in %f seconds' % (len( self.gblends ),self.get_timer()) )

		# update object state
		self.is_fit = True
		return True

	####################
	## fit sequential ##
	####################
	def _fit_sequential( self ):
		""" pygfit._fit_sequential()
		
		Fit gblend objects sequentially (i.e. without parallelization) """

		# loop through gblends
		for gblend_obj in self:

			# give the gblend object the current residuals image
			success = gblend_obj.generate_fitting_images( self.lres_res )

			# returns a traceback string if failed
			if type( success ) == type( '' ):
				log.warn( 'Failed to prepare fitting images for lres id number: %d.\nTraceback:\n%s' % (gblend_obj.number,success) )
				# mark this gblend as bad so that it is skipped in all future iterations
				self.good[gblend_obj.number] = False
				continue

			# fit it!
			( code, best_fit ) = gblend_obj.fit()

			# returns a traceback string if failed
			if type( best_fit ) == type( False ) and best_fit == False:
				log.warn( 'Failed to fit lres id number: %d.\nTraceback:\n%s' % (gblend_obj.number,code) )
				# mark this gblend as bad so that it is skipped in all future iterations
				self.good[gblend_obj.number] = False
				continue

			# and subtract the fit from the residuals image, which will be passed to subsequent gblend objects
			self.lres_res = gblend_obj.subtract_model( self.lres_res, full=True )

	##################
	## fit parallel ##
	##################
	def _fit_parallel( self ):
		""" pygfit._fit_parallel()
		
		Use parallelization to speed up fitting """

		# first we have to figure out what the dependencies between gblend objects are.
		# We want bright objects to be fit and subtracted before fainter objects which overlap in background are fit.
		# the gblend objects themselves will do most of this dependency determination.
		# They must be passed lists of number,x,y,radius for objects brighter than themselves,
		# they will return number,x,y, and radius for themselves, plus a list of numbers for those objects they depend on.
		numbers = []
		xs = []
		ys = []
		radii = []

		# dependencies and dependents dictionaries are used for keeping track of dependencies.
		# dependencies[key] is a list of keys for those gblend objects which the given gblend object [key] depends on.
		# dependents[key] is a list of keys for those gblend objects which depend on the current gblend object [key].
		dependencies = {}
		dependents = {}

		# to_process is a list of keys of gblend objects which are ready for fitting
		to_process = []

		for gblend_obj in self:

			# pass current list of object info and retrieve object info + dependencies
			( number, x, y, radius, depends ) = gblend_obj.get_dependencies( numbers, xs, ys, radii )

			# append info to appropriate list
			numbers.append( number )
			xs.append( x )
			ys.append( y )
			radii.append( radius )

			# if there are any dependencies, then record them in both the dependents and dependencies arrays
			if len( depends ):
				dependencies[gblend_obj.number] = depends
				for number in depends:
					dependents.setdefault( number, [] )
					dependents[number].append( gblend_obj.number )

			# if there aren't any dependencies, then this is ready to go and can be added directly to the processing queue.
			else:
				to_process.append( gblend_obj.number )

		# Since multiprocessing runs things in child process, a queue must be used to retrieve information from the fits.
		# This will be passed along to gblend.fit(), which will return results to the queue.
		# Those results will then be stored back in the appropriate gblend object.
		results = multiprocessing.Queue()

		completed = 0		# number of gblends which have been successfully fit
		started = 0		# number of gblends which have been started
		running = 0		# number of gblends currently being fit

		# continue looping until all gblends are finished
		while completed < len( self.gblend_ids ):

			# as long as there are free threads and gblends to be fit, start more processes
			while (running < self.config['n_threads']) & (started < len( self.gblend_ids )) & (len( to_process ) > 0):

				# figure out what the next gblend object is (pull from beginning of list so we start bright stuff first)
				number = to_process[0]

				# remove from processing queue
				to_process.remove( number )

				# give it the current residuals image
				self.gblends[number].generate_fitting_images( self.lres_res )

				# start the fitting process, and pass it the results queue
				thread = multiprocessing.Process( target=self.gblends[number].fit, args=(results,) )
				thread.start()

				# and update the statistics
				started += 1
				running += 1

			# check to see if any processes have finished
			while not results.empty():

				# fetch results
				( number, code, fit ) = results.get()

				# was there an error?
				if type( fit ) == type( False ) and fit == False:

					log.warn( 'Failed to fit lres id number: %d.\nTraceback:\n%s' % (number,code) )
					# mark this gblend as bad so that it is skipped in all future iterations
					self.good[number] = False

				else:

					# update the appropriate gblend object - record the best fit
					self.gblends[number].set_best_fit( code, fit )

					# and now subtract its fit from the residuals image
					self.lres_res = self.gblends[number].subtract_model( self.lres_res, full=True )

				# update statistics
				completed += 1
				running -= 1

				# loop through the list of things that depend on this gblend, assuming it has dependents
				if dependents.has_key( number ):
					for dependent in dependents[number]:

						# and remove it from the dependencies list
						dependencies[dependent].remove( number )

						# if the dependent gblend no longer has any dependencies, then add it to the processing pool
						if len( dependencies[dependent] ) == 0: to_process.append( dependent )

	################
	## get gblend ##
	################
	def get_gblend( self, image, info, max_shift ):
		""" pygfit.get_gblend( image, info, max_shift )
		
		Returns a gblend object with the given low resolution catalog info and maximum position offset """

		# make sure there are no errors - if so, log the warning and return false
		try:

			gblend_obj = gblend.gblend( info, max_shift, self.config )
	
			# pass the images needed
			gblend_obj.set_images( image, self.lres_img_header, self.lres_seg, self.lres_rms, self.lres_psf, wcs=self.wcs )
	
			# and finally the high resolution catalog.  This will return the number of matching objects from the high resolution catalog
			this_nhres = gblend_obj.set_hres_catalog( self.hres_cat )
	
		except KeyboardInterrupt:
			raise
		except Exception, e:

			log.warn( 'Failed to get gblend id number: %d.\nTraceback:\n%s' % (info['number'],traceback.format_exc()) )
			return False

		# return the gblend
		return gblend_obj

	############################
	## generate output images ##
	############################
	def generate_output_images( self, residuals_file=None ):
		""" pygfit.generate_output_images( residuals_file=None )
		
		Generate output images: residuals image, and individual low resolution models/residuals (similar to galfit)
		Set residuals_file='filename' to override configuration. """

		# make sure that the fitting has been done.
		if not self.is_fit: self.fit()

		# log
		log.log( '\nGenerating output' )
		self.start_timer()

		# tell each gblend to output its image/model/residuals file
		for gblend_obj in self: gblend_obj.output_fit()

		# get residuals filename
		if residuals_file is None: residuals_file = self.lres_res_file

		# remove the image if it exists to avoid irritating clobber notice from pyfits
		if os.path.isfile( residuals_file ): os.remove( residuals_file )

		# write out residuals file
		if self.config.has_key( 'subtract_background' ) and self.config['subtract_background']:
			pyfits.writeto( residuals_file, self.lres_res, self.lres_img_header )
		else:
			pyfits.writeto( residuals_file, self.lres_res+self.lres_back, self.lres_img_header )

		log.log( 'Total time for outputting images: %f seconds' % self.get_timer() )

	####################
	## output catalog ##
	####################
	def output_catalog( self, output_catalog=None ):
		""" pygfit.output_catalog( output_catalog=None )
		
		Output the final catalog.  Output to the file specified in the configuration, or output_catalog if passed """

		# make sure the fitting is done.
		if not self.is_fit: self.fit()

		# log
		log.log( '\nOutputting final catalog' )
		self.start_timer()

		# loop through gblends and fetch catalog
		for (i,gblend_obj) in enumerate(self):

			# fetch catalog for this gblend
			this_cat = gblend_obj.get_catalog()

			# add to full catalog
			if i == 0:
				cat = this_cat
			else:
				for key in this_cat.keys(): cat[key].extend( this_cat[key] )

		# get list of columns to output
		columns = list( self.config['output_columns'] )

		# finally extend list of output columns and formats to include source extractor columns.
		( extractor_columns, formats ) = self.get_extractor_column_formats()
		columns.extend( extractor_columns )

		# convert x/y to ra/dec
		( cat['ra'], cat['dec'] ) = self.xy2rd( cat['x'], cat['y'] )

		# finally convert re from lres pixels to hres pixels and arcseconds
		cat['re_arcsecs'] = np.array( cat['re_lres'] )*self.wcs['scale']
		cat['re_hres'] = np.array( cat['re_arcsecs'] )/( 3600.0*self.config['hres_pixscale'] )

		# now we are ready to write this out.  Call the appropriate output method based on requested output type
		if self.config['output_format'].lower() == 'fits':
			utils.output_fits_catalog( cat, self.config['output_catalog'], columns, more_formats=formats )
		else:
			utils.output_ascii_catalog( cat, self.config['output_catalog'], columns, more_formats=formats )

		log.log( 'Total time for outputting final catalog: %f seconds' % self.get_timer() )


	##################################
	## get extractor column formats ##
	##################################
	def get_extractor_column_formats( self ):
		""" pygfit.get_extractor_column_formats()
		
		Returns a dictionary with format strings for the source extractor columns """

		# formatting information is taken from the original fits-ldac file (self.lres_columns)
		formats = {}
		columns = []
		for ( name, format ) in zip( self.lres_columns.names, self.lres_columns.formats ):
			name = name.lower()

			# don't include number since that is already included as lres_id
			if name == 'number': continue

			# need a different format parameter with ascii output
			if self.config['output_format'].lower() == 'ascii':
				# skip things with funky formats if we are using ascii output
				if format[0] != '1' or ( format[1] != 'D' and format[1] != 'E' and format[1] != 'J' and format[1] != 'I' ):
					log.warn( 'Skipping source extractor column %s in output: unknown format.  Pygfit can output all source extractor columns if you set OUTPUT_FORMAT fits in the pygfit configuration.' % name )
					continue
				# get ascii format
				if format[1] == 'J' or format[1] == 'i':
					format = 0
				elif format[1] == 'D':
					format = 10
				else:
					format = 6

			# append to column list
			columns.append( name )
			# and to format list
			formats[name] = format

		return ( columns, formats )

	#####################
	## run simulations ##
	#####################
	def run_simulations( self ):
		""" pygfit.run_simulations()
		
		Run simulations to estimate errors - this can take a while """

		# don't log anything
		log.disable_logging()

		# first fetch a pygsti object
		sti = self.get_pygsti_object()

		# now generate simulated images
		self.prepare_simulations( sti )

		# and run pygfit
		self.execute_simulations( sti )

		# output the simulation results
		cat = self.output_sim_results( sti )

		# and plot them up
		self.do_sim_plots()

		# turn logging back on
		log.enable_logging()

	######################
	## plot simulations ##
	######################
	def plot_simulations( self ):
		""" pygfit.plot_simulations()
		
		Plot the simulation results.  Will generate the results file if it doesn't already exist """

		# build the sim results if they don't already exist
		if not os.path.isfile( '%s/results.cat' % self.sim_dir ):

			# first fetch a pygsti object
			sti = self.get_pygsti_object()

			# output the simulation results
			cat = self.output_sim_results( sti )

		else:
			cat = None

		self.do_sim_plots()

	#######################
	## get pygsti object ##
	#######################
	def get_pygsti_object( self ):
		""" pygfit.get_pygsti_object()
		
		Fetch a pygsti object for generating simulated images from the current configuration. """

		# Align the images.  The alignment will be stored and reused in simulation frames to save execution time
		# also, the simulated galaxies need to be "misaligned" so that they are realigned the same was as the real ones
		self.align_catalogs()

		# it will use a lot of the same configuration names/values, so we will start with the current configuration.
		config = self.config.copy()

		# manually set some things to account for differences between pygsti and pygfit configuration
		config['image'] = config['lres_image']
		config['psf'] = config['lres_psf']
		config['magzero'] = config['lres_magzero']
		config['catalog'] = config['hres_catalog']

		# re is in arcseconds already
		config['re_arcsec'] = True

		# the pygsti simulation configuration for pygfit...

		# limit object generation to high resolution image area
		if not config.has_key( 'position_border' ) or type( config['position_border'] ) != type( [] ):
			# get left/top/right/bottom objects from hres catalog and use those as position border (minus a small border)
			( wl, wr, wt, wb ) = ( self.hres_cat['ra'].argmax(), self.hres_cat['ra'].argmin(), self.hres_cat['dec'].argmax(), self.hres_cat['dec'].argmin() )
			# and build position border config
			config['position_border'] = [ self.hres_cat['ra'][wl]-self.border/2.0, self.hres_cat['dec'][wl]-self.border/2.0, self.hres_cat['ra'][wr]-self.border/2.0, self.hres_cat['dec'][wr]+self.border/2.0, self.hres_cat['ra'][wt]+self.border/2.0, self.hres_cat['dec'][wt]-self.border/2.0, self.hres_cat['ra'][wb]+self.border/2.0, self.hres_cat['dec'][wb]+self.border/2.0 ]

		# always randomize positions
		config['random_positions'] = True
		# sample from hres catalog
		config['as_is'] = False
		config.setdefault( 'mix_parameters', False )
		config.setdefault( 'random_mags', False )

		# how many galaxies to generate?  Increase frame density by some fraction (default 2.5%)
		frac = float( self.config['fraction_per_frame'] )/100 if self.config.has_key( 'fraction_per_frame' ) else 0.025
		ngals = self.hres_cat['ra'].size
		config.setdefault( 'gals_per_frame', max( np.ceil( ngals*frac ), 5 ) )
		# unless otherwise specified, set number of frames equal to the number needed to reproduce fraction * nhres galaxies
		frac = float( self.config['simulated_fraction'] )/100 if self.config.has_key( 'simulated_fraction' ) else 2.5
		config.setdefault( 'number_frames', np.ceil( np.ceil( ngals*frac )/config['gals_per_frame'] ) )
		# output files
		if not config.has_key( 'image_output' ) and not config.has_key( 'image_suffix' ): config['image_suffix'] = '_sim.fits'
		if not config.has_key( 'catalog_output' ) and not config.has_key( 'catalog_suffix' ): config['catalog_suffix'] = '_sim.cat'
		if not config.has_key( 'region_output' ) and not config.has_key( 'region_suffix' ): config['region_suffix'] = '_sim.reg'
		config.setdefault( 'sim_dir', self.sim_dir )

		# always use individual directories
		config['individual_dirs'] = True
		# and output a fits catalog
		config['output_format'] = 'fits'

		# make sure generated models are offset in the same way as the high resolution catalog
		config['x_offset'] = self.x_offset
		config['y_offset'] = self.y_offset

		# various model settings
		config['model_dir'] = 'models'
		config.setdefault( 'output_models', True )
		config.setdefault( 'min_size', 10 )
		config.setdefault( 'clobber', False )

		# now create the pygsti object with this configuration
		sti = pygsti.pygsti( config=config )

		# are we outputting the pygsti configuration?
		if self.config.has_key( 'output_pygsti_config' ) and self.config['output_pygsti_config']: sti.output_config( 'pygsti_auto.config' )

		# copy the high resolution catalog for pygsti
		cat = self.hres_cat.copy()
		# generate new magnitudes from the low resolution LF
		cat['mag'] = self.lres_cat['mag_auto'][np.random.random_integers( 0, self.nlres-1, size=cat['mag'].size )]

		# load the pygsti object data and pass this new catalog
		sti.load_data( catalog=cat )

		# and return the pygsti object
		return sti

	#########################
	## prepare simulations ##
	#########################
	def prepare_simulations( self, sti ):
		""" pygfit.prepare_simulations( sti ):
		
		Generate simulated images and output config files.  Pass a pygsti object. """

		# first things first - make the simulated images
		sti.make_images()

		# now loop through frames and setup a valid pygfit run in each directory
		for output in sti:

			# simulation directory
			sim_dir = output['dir']

			# generate the configuration for the pygfit run
			config = self.config.copy()

			# strip out path information so everything is relative to the sim directory
			to_strip = ['extractor_config','extractor_params','extractor_catalog','hres_catalog','lres_image','lres_rms','lres_psf']
			for strip in to_strip: config[strip] = os.path.basename( config[strip] )

			# same for check files, but they have to be done differently
			if config.has_key( 'check_plot_files' ) and config['check_plot_files']:
				if type( config['check_plot_files'] ) == type( [] ):
					for i in range( len( config['check_plot_files'] ) ): config['check_plot_files'][i] = os.path.basename( config['check_plot_files'][i] )
				else:
					config['check_plot_files'] = os.path.basename( config['check_plot_files'] )

			# update directories similarly
			to_strip = ['image_dir','rms_dir','segmentation_dir','segmentation_mask_dir']
			for strip in to_strip:
				directory = config[strip]
				if directory[-1] == os.sep: directory = directory[:-1]
				config[strip] = os.path.basename( config[strip] )

			# check to see if pygfit has been prepared already and skip if so
			output_cat = '%s%s' % (sim_dir,config['hres_catalog'])
			output_config = '%spygfit.config' % sim_dir
			if os.path.isfile( output_cat ) and os.path.isfile( output_config ): continue

			# store image alignment to save time for simulations
			config['x_offset'] = self.x_offset
			config['y_offset'] = self.y_offset

			# output all columns
			config['output_columns'] = self.output_columns
			# and output the final catalog in fits format
			config['output_format'] = 'fits'

			# copy some of the needed files
			shutil.copy( self.lres_psf_file, '%s%s' % ( sim_dir, config['lres_psf'] ) )
			shutil.copy( self.lres_rms_file, '%s%s' % ( sim_dir, config['lres_rms'] ) )
			shutil.copy( self.config['extractor_config'], '%s%s' % ( sim_dir, config['extractor_config'] ) )

			# update lres image to point to sim image
			config['lres_image'] = os.path.basename( output['img_out'] )
			# and rename the output catalog
			config['output_catalog'] = 'pygfit.cat'

			# finally generate the hres catalog, which is the sum of the current high res catalog and the simulated one
			# first load the simulation catalog
			fits = pyfits.open( output['cat_out'] )
			sim_cat = fits[1].data

			# now combine.  While we're at it, set the catalog field names in the config
			hres_cat = {}
			# copy over the hres cat
			for field in ['model','ra','dec','id','re','n','ba','pa']: hres_cat[field] = self.hres_cat[field].copy()
			# now append the simulated galaxies to the hres cat
			fields = ['model','ra','dec','re','n','pa','ba','id']
			config['id'] = 'id'
			for field in fields:
				hres_cat[field] = np.append( hres_cat[field], sim_cat.field( field ) )
				config[field] = field
			config['model_type'] = 'model'
			# have to convert re back to high resolution pixels from arcseconds
			hres_cat['re'] /= self.config['hres_pixscale']*3600.0
			# copy over the magnitude field manually because we want to randomly assign a magnitude
			# to use the actual magnitude as the first guess into the fit would be cheating...
			hres_cat['mag'] = np.append( self.hres_cat['mag'], self.hres_cat['mag'][np.random.random_integers( 0, self.hres_cat['mag'].size-1, sim_cat.size )] )
			config['point_mag'] = 'mag'
			config['mag'] = 'mag'

			# now we can write it out
			utils.output_fits_catalog( hres_cat, output_cat, hres_cat.keys() )

			# as well as the config file
			self.output_config( output_config, config )

	#####################
	## run simulations ##
	#####################
	def execute_simulations( self, sti ):
		""" pygfit.run_simulations( sti )
		
		Run previously prepared simulations """

		# first fetch the current working directory
		cwd = os.getcwd()
		if cwd[-1] != os.sep: cwd += os.sep

		# loop through simulation frames.  pygsti object will give simulation details
		for output in sti:

			# change to the simulation directory
			os.chdir( '%s%s' % (cwd,output['dir']) )

			# load pygfit object
			pyg = pygfit()

			# if pygfit has already been run, then skip
			if os.path.isfile( pyg.config['output_catalog'] ): continue

			# run pygfit.  Catch any errors so that the simulations are not interupted
			try:
				pyg.go()
			except KeyboardInterrupt:
				raise
			except Exception, e:
				print 'Error raised while running pygfit in %s%s' % (cwd,output['dir'])
				print type( e ), e

		# finally change back to the original working directory
		os.chdir( cwd )

	########################
	## output sim results ##
	########################
	def output_sim_results( self, sti ):
		""" pygfit.output_sim_results( sti )
		
		Output the results of the simulation """

		# final catalog
		cat = {}

		# fields with real and err fields to be copied if they exist
		with_errs = ['n','ba','pa']

		# loop through simulation frames
		c = 0
		for (i,output) in enumerate(sti):

			# make sure sim catalog and pygfit catalog both exist
			if not os.path.isfile( output['cat_out'] ) or not os.path.isfile( output['dir'] + 'pygfit.cat' ): continue

			# load the sim catalog
			fits = pyfits.open( output['cat_out'] )
			real = fits[1].data

			# and the pygfit catalog
			fits = pyfits.open( output['dir'] + 'pygfit.cat' )
			fit = fits[1].data

			# extract the hres id as a list for matching.  This requires a little extra work because .tolist() will leave trailing spaces on some.
			ids = [ id.strip() for id in fit.field('hres_id') ]

			# make a list of matched simulation and fit objects based on id
			mr = []
			mf = []
			for (j,name) in enumerate(real.field('id')):
				if not ids.count( name ): continue
				ind = ids.index( name )
				mr.append( j )
				mf.append( ind )

			# if nothing matched, then we are done with this sim
			if len( mr ) == 0: continue

			# copy the result to the final catalog
			if c == 0:
				# get the list of output columns
				columns = list( self.config['output_columns'] )
				for name in columns: cat[name] = fit.field(name)[mf]
				cat['mag_input'] = real.field('mag_real')[mr] if real.names.count( 'mag_real' ) else real.field('mag')[mr]
				cat['flux_input'] = real.field('flux_real')[mr] if real.names.count( 'flux_real' ) else real.field('flux')[mr]
				cat['frame'] = np.zeros( len( mr ) ) + i
				# copy over errors and real values
				for name in with_errs:
					if real.names.count( '%s_err' % name ): cat['%s_err' % name] = real.field('%s_err' % name)[mr]
					if real.names.count( '%s_real' % name ): cat['%s_real' % name] = real.field('%s_real' % name)[mr]
				# re has to be done by hand because the error fields will be renamed
				if real.names.count( 're_err' ): cat['re_arcsecs_err'] = real.field('re_err')[mr]
				if real.names.count( 're_real' ): cat['re_arcsecs_real'] = real.field('re_real')[mr]
				# finally get list of output columns and formats to include from source extractor columns.
				( extractor_columns, extractor_formats ) = self.get_extractor_column_formats()
				for name in extractor_columns: cat[name] = fit.field(name)[mf]
			else:
				for name in columns: cat[name] = np.append( cat[name], fit.field(name)[mf] )
				if real.names.count( 'mag_real' ):
					cat['mag_input'] = np.append( cat['mag_input'], real.field('mag_real')[mr] )
				else:
					cat['mag_input'] = np.append( cat['mag_input'], real.field('mag')[mr] )
				if real.names.count( 'flux_real' ):
					cat['flux_input'] = np.append( cat['flux_input'], real.field('flux_real')[mr] )
				else:
					cat['flux_input'] = np.append( cat['flux_input'], real.field('flux')[mr] )
				cat['frame'] = np.append( cat['frame'], np.zeros( len( mr ) ) + i )
				# add in source extractor columns
				for name in extractor_columns: cat[name] = np.append( cat[name], fit.field(name)[mf] )
				# copy over errors and real values
				for name in with_errs:
					if real.names.count( '%s_err' % name ): cat['%s_err' % name] = np.append( cat['%s_err' % name], real.field('%s_err' % name)[mr] )
					if real.names.count( '%s_real' % name ): cat['%s_real' % name] = np.append( cat['%s_real' % name], real.field('%s_real' % name)[mr] )
				# re has to be done by hand because the error fields will be renamed
				if real.names.count( 're_err' ): cat['re_arcsecs_err'] = np.append( cat['re_arcsecs_err'], real.field('re_err')[mr] )
				if real.names.count( 're_real' ): cat['re_arcsecs_real'] = np.append( cat['re_arcsecs_real'], real.field('re_real')[mr] )


			c += 1

		if not c: raise ValueError( 'Could not find any completed simulations!' )

		# output the mag_input field right after the mag field
		columns.insert( columns.index( 'mag' )+1, 'mag_input' )
		# and frame right after hres_id
		columns.insert( columns.index( 'hres_id' )+1, 'frame' )

		# just stick the source extractor columns at the very end of the catalog
		columns.extend( extractor_columns )

		# also output the error fields after their original fields
		for name in ['re_arcsecs','n','ba','pa']:
			if not cat.has_key( '%s_real' % name ): continue
			columns.insert( columns.index( name )+1, '%s_real' % name )
			columns.insert( columns.index( name )+2, '%s_err' % name )

		# okay, all done generating catalog.  Now output it, and return it
		if self.config['output_format'] == 'fits':
			utils.output_fits_catalog( cat, '%s/results.cat' % self.sim_dir, columns, more_formats=extractor_formats )
		else:
			utils.output_ascii_catalog( cat, '%s/results.cat' % self.sim_dir, columns, more_formats=extractor_formats )

		return cat

	##################
	## do sim plots ##
	##################
	def do_sim_plots( self, cat=None ):
		""" pygfit.do_sim_plots( cat )
		
		Output some plots showing the results of the simulations """

		import matplotlib.pyplot as pyplot

		if cat is None:
			if not os.path.isfile( '%s/results.cat' % self.sim_dir ): raise ValueError( 'Can not find simulation results!' )
			fits = pyfits.open( '%s/results.cat' % self.sim_dir )
			cat = fits[1].data

		# find the worst 5%
		diffs = cat['mag_input'] - cat['mag']
		good = np.ones( diffs.size, dtype='bool' )
		d = np.abs( diffs - np.median( diffs ) )
		sind = d.argsort()
		good[sind[-1*int( np.ceil(diffs.size*0.05) ):]] = False
		bad = ~good

		# don't keep anything that is more than 5 mags fainter than mag_brightest
		keep = (cat['mag'] - cat['mag_brightest'] < 5) #& (cat['mag'] < 21.5) & (cat['re_arcsecs'] < 2.5) & (cat['n'] < 7.5)
		bad = bad & keep
		good = good & keep

		# use the best 5% to set magnitude limits
		#ymax = diffs[good].max()
		#ymin = diffs[good].min()
		#ymax *= 2
		#ymin = 0 if ymin > 0 else ymin*2
		ymin = -5; ymax = 5

		# use the 95% rule for re to find axis limit
		sind = cat['re_arcsecs'].argsort()
		re_max = cat['re_arcsecs'][sind[-1*int( np.ceil(diffs.size*0.05) )]]

		# same for nearest
		sind = cat['nearest'].argsort()
		near_max = cat['nearest'][sind[-1*int( np.ceil(diffs.size*0.05) )]]

		# generate output plots for all objects, sersics, and points
		# ms = []; files = [ 'all.ps', 'point.ps', 'sersic.ps' ]
		ms = []; files = [ 'all.pdf', 'point.pdf', 'sersic.pdf' ]
		ms.append( np.ones( cat['mag'].size, dtype='bool' ) )
		ms.append( cat['model'] == 'point' )
		ms.append( cat['model'] == 'sersic' )

		for ( m, file ) in zip ( ms, files ):

			g = good & m
			b = bad & m

			# start with the basics - mag versus mag error
			pyplot.subplot( 2, 2, 1 )
			pyplot.plot( cat['mag_input'][g], diffs[g], 'bo' )
			pyplot.plot( cat['mag_input'][b], diffs[b], 'ro' )
			pyplot.axhline( 0.0, color='k' )
			pyplot.xlabel( 'Mag (In)' )
			pyplot.ylabel( 'Mag (In - Out)' )
			#pyplot.axis( ymin=ymin, ymax=ymax )

			# nblend versus mag diff.  This is quantized so go ahead and bin it
			pyplot.subplot( 2, 2, 2 )
			locs = []; medians = []; errs = []; errs_reject = []
			for n in range( cat['nblend'].min(), cat['nblend'].max()+1 ):
				mn = (cat['nblend'] > n - 0.5) & (cat['nblend'] < n + 0.5)
				if not m.any(): continue
				locs.append( n )
				medians.append( np.median( diffs[mn] ) )
				errs.append( np.std( diffs[mn & (b | g)] ) )
				errs_reject.append( np.std( diffs[mn & g] ) )
			pyplot.errorbar( locs, medians, errs, fmt='ro' )
			pyplot.errorbar( locs, medians, errs_reject, fmt='bo' )
			pyplot.axhline( 0.0 )
			pyplot.xlabel( 'Number of Blended Objects' )
			pyplot.ylabel( 'Mag (In - Out)' )
			#pyplot.axis( ymin=ymin, ymax=ymax )

			if file != 'point.pdf':
				# re versus mag diff
				pyplot.subplot( 2, 2, 3 )
				pyplot.plot( cat['re_arcsecs'][g], diffs[g], 'bo' )
				pyplot.plot( cat['re_arcsecs'][b], diffs[b], 'ro' )
				pyplot.axhline( 0.0, color='k' )
				pyplot.xlabel( '$Re$ (")' )
				pyplot.ylabel( 'Mag (In - Out)' )
				#pyplot.axis( ymin=ymin, ymax=ymax, xmin=0, xmax=re_max )

				# n versus mag diff
				pyplot.subplot( 2, 2, 4 )
				pyplot.plot( cat['n'][g], diffs[g], 'bo' )
				pyplot.plot( cat['n'][b], diffs[b], 'ro' )
				pyplot.axhline( 0.0, color='k' )
				pyplot.xlabel( 'Sersic Index' )
				pyplot.ylabel( 'Mag (In - Out)' )
				#pyplot.axis( ymin=ymin, ymax=ymax )

			pyplot.gcf().set_size_inches( (11.0,7.5) )
			pyplot.savefig( '%s/%s' % (self.sim_dir,file), orientation='landscape' )
			pyplot.clf()

		# mag errors versus mag fields
		pyplot.subplot( 2, 2, 1 )
		pyplot.plot( cat['mag_input'], diffs, 'bo' )
		pyplot.axhline( 0.0, color='k' )
		pyplot.xlabel( 'Mag (In)' )
		pyplot.ylabel( 'Mag (In - Out)' )
		pyplot.axis( ymin=ymin, ymax=ymax )

		pyplot.subplot( 2, 2, 2 )
		pyplot.plot( cat['nearest'], diffs, 'bo' )
		pyplot.axhline( 0.0, color='k' )
		pyplot.xlabel( 'Nearest (")' )
		pyplot.ylabel( 'Mag (In - Out)' )
		pyplot.axis( ymin=ymin, ymax=ymax, xmax=near_max )

		pyplot.subplot( 2, 2, 3 )
		pyplot.plot( cat['mag_brightest'], diffs, 'bo' )
		pyplot.axhline( 0.0, color='k' )
		pyplot.xlabel( 'Mag (Brightest)' )
		pyplot.ylabel( 'Mag (In - Out)' )
		pyplot.axis( ymin=ymin, ymax=ymax )

		pyplot.subplot( 2, 2, 4 )
		s = cat['mag_brightest'] != cat['mag']
		pyplot.plot( (cat['mag_brightest']-cat['mag_input'])[s], diffs[s], 'bo' );
		pyplot.axhline( 0.0, color='k' )
		pyplot.xlabel( 'Mag (Brightest) - Mag (In)' )
		pyplot.ylabel( 'Mag (In - Out)' )
		pyplot.axis( ymin=ymin, ymax=ymax )
		pyplot.gcf().set_size_inches( (11.0,7.5) )
		pyplot.savefig( '%s/mag.pdf' % self.sim_dir, orientation='landscape' )
		pyplot.clf()

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
		extractor_order = ['extractor_config','extractor_params','extractor_catalog','extractor_cmd','extractor_columns','skip_extractor']
		extractor = {	'extractor_config':	{'format': '%s', 'comment': 'Name of source extractor configuration file'},
				'extractor_params':	{'format': '%s', 'comment': 'Name of source extractor parameters file (will be generated automatically)'},
				'extractor_catalog':	{'format': '%s', 'comment': 'Output catalog name for source extractor'},
				'extractor_cmd':	{'format': '%s', 'comment': 'Location of source extractor executable'},
				'extractor_columns':	{'format': '%s', 'comment': 'Additional source extractor fields to copy to final catalog'},
				'skip_extractor':	{'format': '%s', 'comment': 'Skip running source extractor if it has already been run (True/False)'} }

		hres_order = ['hres_catalog','hres_rotangle','hres_pixscale']
		hres = {	'hres_catalog':		{'format': '%s', 'comment': 'Filename of high resolution catalog'},
				'hres_rotangle':	{'format': '%12.6e', 'comment': 'Roll angle for high resolution image, West of North in Degrees'},
				'hres_pixscale':	{'format': '%12.6e', 'comment': 'Pixel scale for high resolution image, Degrees per pixel'} }

		hres_cat_order = ['model_type','id','ra','dec','mag','point_mag','re','n','pa','ba']
		#form = '%d' if type( self.config['model_type'] ) == type( 0 ) else '%s'
		form = '%s'
		hres_cat = {	'model_type':		{'format': form, 'comment': "Galaxy model type (either 'sersic' or 'point')"},
				'id':			{'format': form, 'comment': 'Unique id'},
				'ra':			{'format': form, 'comment': 'Right Ascension'},
				'dec':			{'format': form, 'comment': 'Declination'},
				'mag':			{'format': form, 'comment': 'Magnitude for sersic models'},
				'point_mag':		{'format': form, 'comment': 'Magnitude for point models'},
				're':			{'format': form, 'comment': 'Effective radius (sersic only)'},
				'n':			{'format': form, 'comment': 'Sersic index (sersic only)'},
				'pa':			{'format': form, 'comment': 'Position Angle (sersic only)'},
				'ba':			{'format': form, 'comment': 'Axis Ratio (sersic only)'} }

		lres_order = ['lres_image','lres_rms','lres_psf','lres_magzero']
		lres = {	'lres_image':		{'format': '%s', 'comment': 'Filename of low resolution fits image'},
				'lres_rms':		{'format': '%s', 'comment': 'Filename of low resolution rms image'},
				'lres_psf':		{'format': '%s', 'comment': 'Filename of low resolution psf image'},
				'lres_magzero':		{'format': '%7.4f', 'comment': 'Magnitude zeropoint for low resolution image'} }

		fit_order = ['use_integration','gpu','gpu_nthreads','n_threads','global_max_shift','max_shift','min_mag','max_mag','align_min_mag','align_max_mag','n_align','pad_length','x_offset','y_offset']
		fit = {		'use_integration':	{'format': '%s', 'comment': 'Whether or not to use integration to properly calculate hard-to-estimate sersic models'},
				'gpu':			{'format': '%s', 'comment': 'Whether or not to attempt to speed up calculations with a GPU'},
				'gpu_nthreads':		{'format': '%d', 'comment': 'The number of threads per block to execute on the GPU'},
				'n_threads':		{'format': '%d', 'comment': 'Maximum number of cpu threads to use (will never use more than the actual number of cpus)'},
				'global_max_shift':	{'format': '%.4f', 'comment': 'Maximum allowed (global) positional offset between high and low resolution catalog (arcseconds)'},
				'max_shift':		{'format': '%.4f', 'comment': 'Maximum allowed positional shift for a high resolution object during fit (arcseconds)'},
				'min_mag':		{'format': '%.4f', 'comment': 'Minimum magnitude of objects to fit'},
				'max_mag':		{'format': '%.4f', 'comment': 'Maximum magnitude of objects to fit'},
				'align_min_mag':	{'format': '%.4f', 'comment': 'Minimum magnitude of objects to include in alignment'},
				'align_max_mag':	{'format': '%.4f', 'comment': 'Maximum magnitude of objects to include in alignment'},
				'n_align':		{'format': '%d', 'comment': 'Maximum number of objects used in alignment calculation'},
				'pad_length':		{'format': '%d', 'comment': 'Padding region (in pixels) around model to allow room for interpolation/convolution'},
				'x_offset':		{'format': '%.4f', 'comment': 'Offset in x direction to align catalogs'},
				'y_offset':		{'format': '%.4f', 'comment': 'Offset in y direction to align catalogs'}, }

		out_order = ['subtract_background','image_dir','rms_dir','segmentation_dir','segmentation_mask_dir','output_all_models','output_catalog','output_format','output_columns']
		out = {		'subtract_background':	{'format': '%s', 'comment': 'Whether or not to subtract the background from the final residuals image'},
				'image_dir':		{'format': '%s', 'comment': 'Directory for outputting image cutouts (relative to working directory)'},
				'rms_dir':		{'format': '%s', 'comment': 'Directory for outputting rms cutouts'},
				'segmentation_dir':	{'format': '%s', 'comment': 'Directory for outputting segemntation cutouts'},
				'segmentation_mask_dir':{'format': '%s', 'comment': 'Directory for outputting segmentation mask cutouts'},
				'output_all_models':	{'format': '%s', 'comment': 'Whether or not to output an extension for individual models in the _fit.fits files'},
				'output_catalog':	{'format': '%s', 'comment': 'Filename for output catalog'},
				'output_format':	{'format': '%s', 'comment': "'fits' or 'ascii' - output type for final catalog"},
				'output_columns':	{'format': '%s', 'comment': 'Fields to output to final catalog'} }

		checks_order = ['check_plots','check_plot_files']
		checks = {	'check_plots':		{'format': '%s', 'comment': 'Type of check plots to generate'},
				'check_plot_files':	{'format': '%s', 'comment': 'Filenames for check plots'} }

		# put them all in a list
		groups = [ extractor, hres, hres_cat, lres, fit, out, checks ]
		orders = [ extractor_order, hres_order, hres_cat_order, lres_order, fit_order, out_order, checks_order ]
		# also label them
		labels = [ 'Source Extractor', 'High Resolution Catalog', 'High Resolution Catalog Format', 'Low Resolution Images', 'Fitting Settings', 'Output Settings', 'Check Plots' ]

		if config is None: config = self.config

		# okay, now we can write out the config file
		utils.write_config( filename, config, groups, labels, orders )

	###########################
	## convert ra/dec to x/y ##
	###########################
	def rd2xy( self, ra, dec ):
		""" (x,y) = pygfit.rd2xy( ra, dec ) """

		# make sure the low resolution data has been loaded
		self.load_low_resolution_data()

		# make sure the image's wcs info has been loaded
		self.load_wcs()

		return utils.rd2xy( self.wcs, ra, dec )

	###########################
	## convert x/y to ra/dec ##
	###########################
	def xy2rd( self, x, y ):
		""" (ra, dec) = pygfit.xy2rd( x, y ) """

		# make sure the low resolution data has been loaded
		self.load_low_resolution_data()

		# make sure the image's wcs info has been loaded
		self.load_wcs()

		return utils.xy2rd( self.wcs, x, y )

	##############
	## load wcs ##
	##############
	def load_wcs( self ):
		""" pygfit.load_wcs()
		
		Retrieves basic wcs info from the header and calculates some useful values """

		# nothing to do if it is already loaded
		if self.wcs_loaded: return True

		# use utils.get_wcs_info() to fetch the wcs info
		self.wcs = utils.get_wcs_info( self.lres_img_header )

		# if successful, then mark wcs as loaded
		if len( self.wcs.keys() ) > 0: self.wcs_loaded = True

	#####################
	## get lres object ##
	#####################
	def get_lres_object( self, ind, key=False ):
		""" details = pygfit.get_lres_object( ind, key=False )
		
		Returns a dictionary with all the properties of an object in the low resolution catalog.
		Pass along the index (to pygfit.lres_cat) of the object desired.
		If key=True then it pass the object number instead of the index, and it will fetch it accordingly. """

		# first make sure the low resolution catalog is loaded
		if not self.lres_loaded: self.load_low_resolution_data()

		# find the index manually?
		if key:
			ind = np.where( self.lres_cat['number'] == ind )[0]
			if ind.size == 0: raise ValueError( 'Could not fetch lres object - that number was not found!' )

		# now just return that index
		return self.lres_cat[ind]

def usage():
	print ""
	print "pygfit.py [--help --simulate --plot_sims --skip_extractor --residuals_file=file --output_catalog=file config_file log_file warn_file]"
	print ""
	print "all command line flags must come first"
	print "run pygfit with the parameters specified in config_file"
	print "config_file, log_file, and warn_file are all optional and default to"
	print "pygfit.config pygfit.log pygfit.warn"
	print ""

if __name__ == '__main__':
	import sys,getopt

	# check arguments
	try:
		(opts, args) = getopt.getopt( sys.argv[1:], 'pcshero', ['plot_sims','config','simulate','help','skip_extractor','residuals_file','output_catalog'] )
	except getopt.GetoptError, err:
		print str(err)
		usage()
		sys.exit(2)

	# check flags
	skip_extractor = None
	residuals_file = None
	output_catalog = None
	run_sims = False
	plot_sims = False
	output_config = False
	for (o, a) in opts:
		if o == '-h' or o == '--help':
			usage()
			sys.exit(2)
		if o == '-p' or o == '--plot_sims': plot_sims = True
		if o == '-r' or o == '--residuals_file': residuals_file = a
		if o == '-o' or o == '--output_catalog': residuals_file = a
		if o == '-e' or o == '--skip_extractor': skip_extractor = True
		if o == '-s' or o == '--simulate': run_sims = True
		if o == '-c' or o == '--config': output_config = True

	# default files
	files = ['pygfit.config','pygfit.log','pygfit.warn']
	for i,file in enumerate( args ): files[i] = file

	if output_config:
		pyg = pygfit( True )
		pyg.output_config()
		sys.exit(2)

	# check for blank input
	if len( args ) == 0 and not os.path.isfile( files[0] ) and not os.path.isfile( files[1] ) and not os.path.isfile( files[2] ):
		usage()
		sys.exit(2)

	logging = True
	if run_sims or plot_sims: logging = False

	pyg = pygfit( files[0], files[1], files[2], logging=log )

	if run_sims:
		pyg.run_simulations()
	elif plot_sims:
		pyg.plot_simulations()
	else:
		pyg.go( skip_extractor=skip_extractor, residuals_file=residuals_file, output_catalog=output_catalog )
