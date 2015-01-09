import pyfits,os,re,gpu,data_histogram
import scipy.optimize
import numpy as np

def get_wcs_info( header ):
	""" wcs = get_wcs_info( header )
	
	Returns a dictionary with wcs information.
	Pass a pyfit header """

	#has_key deprecated
	#if header.has_key( 'cd1_1' ) and header.has_key( 'cd1_2' ) and header.has_key( 'cd2_1' ) and header.has_key( 'cd2_2' ):
	if 'cd1_1' in header and 'cd1_2' in header and 'cd2_1' in header and 'cd2_2' in header:
		cd = np.array([header['cd1_1'],header['cd1_2'],header['cd2_1'],header['cd2_2']])
	#elif header.has_key( 'cdelt1' ) and header.has_key( 'cdelt2' ):
	elif 'cdelt1' in header and 'cdelt2' in header:
		cd = np.array([header['cdelt1'],0,0,header['cdelt2']])
	else:
		raise ValueError( 'Unable to find position data in image header!' )

	ra0 = np.radians( header['crval1'] )
	dec0 = np.radians( header['crval2'] )

	# compute pixel scale
	scale = ( ( cd[0]**2 + cd[2]**2 ) * ( cd[1]**2 + cd[3]**2 ) )**0.25 * 3600

	# compute determinant
	det = cd[0]*cd[3] - cd[1]*cd[2]
	if det == 0: raise ValueError( 'Singular CD matrix!' )

	# compute inverse
	inv = np.array( [cd[3]/det, -1.*cd[1]/det, -1.*cd[2]/det, cd[0]/det] )

	# compute rotation angle (Calabretta & Greisen, eqn 162, paper III)
	if ( cd[0] > 0 ):
		( rho_a, rho_b ) = ( np.arctan( cd[2]/cd[0] ), np.arctan( cd[1]/(-1.0*cd[3]) ) )
	else:
		( rho_a, rho_b ) = ( np.arctan( cd[2]/cd[0] ), np.arctan( -1.0*cd[1]/cd[3] ) )
	rot_angle = ( rho_a + rho_b )/2.0
	cdelt = np.sqrt( np.abs( cd[0]/np.cos( rot_angle ) ) * np.abs( cd[3]/np.cos( rot_angle ) ) )

	# save in a dictionary
	wcs = {'scale': scale, 'cd': cd, 'ra0': ra0, 'dec0': dec0, 'x0': header['crpix1'], 'y0': header['crpix2'], 'det':det, 'inv':inv, 'rot_angle': rot_angle, 'cdelt': cdelt}

	# and return
	return wcs

def rd2xy( wcs, ra, dec ):
	""" (x,y) = utils.rd2xy( wcs, ra, dec ) """

	# convert coords to radians
	ra_rad = np.radians( ra )
	dec_rad = np.radians( dec )

	# continue on our way
	bottom = np.sin(dec_rad)*np.sin(wcs['dec0']) + np.cos(dec_rad)*np.cos(wcs['dec0'])*np.cos(ra_rad-wcs['ra0'])
	if ( np.alltrue( bottom != 0. ) == False ):
		log.log( 'RA/Dec values yield zero denominator.  Quitting.' )
		raise ValueError( 'RA/Dec values yield zero denominator' )

	xi = np.degrees( np.cos(dec_rad)*np.sin(ra_rad-wcs['ra0'])/bottom )
	eta = np.degrees( ( np.sin(dec_rad)*np.cos(wcs['dec0']) - np.cos(dec_rad)*np.sin(wcs['dec0'])*np.cos(ra_rad-wcs['ra0']) )/bottom )

	return ( wcs['inv'][0]*xi + wcs['inv'][1]*eta + wcs['x0'], wcs['inv'][2]*xi + wcs['inv'][3]*eta + wcs['y0'] )

def xy2rd( wcs, x, y ):
	""" (ra, dec) = pygfit.xy2rd( wcs, x, y ) """

	# and do the conversion...
	xdiff = np.asarray( x ) - wcs['x0']
	ydiff = np.asarray( y ) - wcs['y0']
	xi = np.radians( wcs['cd'][0] * xdiff + wcs['cd'][1] * ydiff )
	eta = np.radians( wcs['cd'][2] * xdiff + wcs['cd'][3] * ydiff )

	ra = np.degrees( np.arctan2( xi, np.cos(wcs['dec0']) - eta*np.sin(wcs['dec0']) ) + wcs['ra0'] )
	dec = np.degrees( np.arctan2( eta*np.cos(wcs['dec0']) + np.sin(wcs['dec0']), np.sqrt( (np.cos(wcs['dec0']) - eta*np.sin(wcs['dec0']) )**2 + xi**2) ) )

	ra = np.mod( ra, 360. )

	return (ra, dec)

def make_rms_map( img_file, exp_file, gain, rms_file, out_ps=None, fluxconv=1 ):
	""" make_rms_map( img_file, exp_file, gain, rms_file, out_ps=None, fluxconv=1 )
	
	Generates an RMS map from an image file and and exposure map.  Uses the
	scatter in the image to estimate the subtracted background and add
	it back into the RMS map.  Plots up the measured background """
	
	# open the pygfit image and exposure time map
	fits = pyfits.open( exp_file )
	exp = fits[0].data
	fits = pyfits.open( img_file )
	img = fits[0].data/fluxconv
	
	# set anything without data to np.nan
	m = exp > 0
	fits[0].data[~m] = np.nan
	
	# now calculate the subtracted background, which is the square of the image RMS
	m = exp > 0
	dn = img[m].ravel()*exp[m].ravel()
	
	# use a 3-sigma rejection to estimate the image RMS
	( sig, mkeep ) = sigrej( dn, return_mask=True )
	back = sig**2.0
	mean = np.mean( dn[mkeep] )
	# and now calculate the rms
	fits[0].data[m] = np.sqrt( img[m]*exp[m] + back )/(exp[m]*np.sqrt(gain))*fluxconv
	fits.writeto( rms_file, clobber=True )
	
	# if desired, plot out a histogram showing the image RMS and the measured RMS.
	# also overplot the fitted gaussian
	if out_ps is not None:
		import matplotlib.pyplot as pyplot
		xmin = mean - 10*np.sqrt(back)
		xmax = mean + 10*np.sqrt(back)
		hist = data_histogram.data_histogram( dn, min=xmin, max=xmax, nbins=40 )
		norm = 1.0/dn.size
		xs = np.arange( 100 )/99.0*(xmax-xmin) + xmin
		ys = hist.hist.max()*norm*np.exp( -1.0*( xs-mean )**2.0 / (2*sig**2.0) )
		pyplot.clf()
		pyplot.bar( hist.mins, hist.hist*norm, width=hist.bin, color='white' )
		pyplot.plot( xs, ys, 'b-' )
		pyplot.axvline( mean )
		pyplot.annotate( '$\mu$ = %.2f\n$\sigma$ = %.2f\n' % (mean,sig), (0.05,0.85), (0.05,0.85), xycoords='axes fraction', textcoords='axes fraction' )
		pyplot.xlabel( 'Image*Exposure Time' )
		pyplot.ylabel( 'Fraction Pixels' )
		pyplot.savefig( out_ps )
		pyplot.clf()

def load_config( filename, to_lower=True ):
	""" dict = load_config( filename, to_lower=True )
	
	Parses the given configuration file and returns a hash with key/value pairs
	Use # as a comment.  Key/values are separated by spaces.  Lists are specified with commas.
	if to_lower=True then returned key names will be all lowercase
	"""

	file = open( filename, 'r' )
	config = {}

	for line in file:
		# check for comments and clean input
		comment = line.find( '#' );
		if comment >= 0: line = line[:comment]
		line = line.strip()
		if line == '': continue

		# split key/value
		vals = line.split( None, 1 )
		if len( vals ) == 1: continue
		key = vals[0]
		value = vals[1]

		# is this a comma separated list?
		is_list = False
		if value.find( ',' ) >= 0:
			is_list = True
			values = [x.strip() for x in value.split( ',' )]
		else:
			values = [value]

		# now loop through all values and convert to the appropriate type
		for i in range( len( values ) ):

			# is this a boolean value?
			if value.lower() == 'true':
				values[i] = True
			elif value.lower() == 'false':
				values[i] = False
			# string?
			elif values[i][0] == "'" and values[i][-1] == "'":
				values[i] = values[i].replace( "'", "" )
			elif values[i][0] == '"' and values[i][-1] == '"':
				values[i] = values[i].replace( '"', '' )
			elif re.match( '(-|\+)?\d+$', values[i] ):
				values[i] = int( values[i] )
			elif re.match( '(-|\+)?\d{0,}\.?\d{0,}([eEdD](-|\+)?\d+)?$', values[i] ):
				values[i] = float( values[i] )

		if to_lower: key = key.lower()

		if is_list:
			config[key.lower()] = values
		else:
			config[key] = values[0]

	file.close()
	return config

def write_config( filename, config, groups, labels, orders ):
	""" utils.write_config( filename, config, groups, labels, orders ) """

	# to properly format everything, we have to loop through and format the configuration values.
	# as we loop we will record the max length of the config values and the length of the config names.
	# this will allow us to output everything so that it all lines up and satisfies my OCD

	# so, descend through group structure and build up dictionary with formated names, values, and comment
	max_value = 1
	max_label = 1
	formatted = {}
	for group in groups:
		for ( key, info ) in group.iteritems():
			if not config.has_key( key ): continue
			# update max label length
			max_label = max( max_label, len( key ) )

			# format configuration value, which is different for lists
			if type( config[key] ) == type( [] ):
				vals = []
				for item in config[key]: vals.append( info['format'] % item )
				val = ', '.join( vals )
			else:
                                #print info['format'],'info'
                                #print config[key],'key'
				val = info['format'] % config[key]
				max_value = max( max_value, len( val ) )

			# store everything
			formatted[key] = ( key.upper(), val, info['comment'] )

	# calculate label and value format given max lengths
	value_format = '%%-%ds' % max_value
	label_format = '%%-%ds' % max_label
	# and now make format string for the whole line
	format = '%s    %s    # %%s' % ( label_format, value_format )
	# different format string for lists
	list_format = '# %%s\n%s    %%s' % ( label_format )

	# okay, now everything is formatted and we can begin to write stuff out.
	# for now just format lines and store in list.  Loop through in order.
	lines = []
	for ( label, field_order ) in zip( labels, orders ):
		# separate groups with new lines
		if label != labels[0]: lines.append( '' )

		# now the label
		lines.append( '### %s ###' % label )

		# loop through groups again and make config lines
		for ( key ) in field_order:
			if not config.has_key( key ): continue
			# generate line
			if type( config[key] ) == type( [] ):
				( name, value, comment ) = formatted[key]
				lines.append( list_format % ( comment, name, value ) )
			else:
				lines.append( format % formatted[key] )

	# now write it all out
	file = '\n'.join( lines )
	if filename is None:
		print file
	else:
		fp = open( filename, 'wb' )
		fp.write( file )
		fp.close()

def load_image( filename, header=False ):
	""" img = load_image( filename, header=False )
	
	returns the first image extension from a fits file.
	returns (img,header) if header=True """

	if not os.path.isfile( filename ): raise ValueError( 'The fits file, %s, does not exist!' % filename )
	fits = pyfits.open( filename )

	# loop through extensions until an image is found
	for i in range( len( fits ) ):

		# if these header keywords don't exist, then it isn't an image
		#has_key is now deprecated
		#if not fits[i].header.has_key( 'naxis' ) or not fits[i].header.has_key( 'naxis1' ) or not fits[i].header.has_key( 'naxis2' ): continue
		if not 'naxis' in fits[i].header or not 'naxis1' in fits[i].header or not 'naxis2' in fits[i].header: continue

		# now check for reasonable values
		if fits[i].header['naxis'] != 2 or fits[i].header['naxis1'] < 2 or fits[i].header['naxis2'] < 2: continue

		# found an image extension!
		if header:
			return (fits[i].data,fits[i].header)
		else:
			return fits[i].data

	# hmm, nothing found...
	raise ValueError( 'No image extension was found in the file %s!' % filename )

def write_region( filename, ras_in, decs_in, sizes_in, titles=None, color='green' ):
	""" write_region( filename, ras, decs, sizes, titles=None, color='green' )

	Outputs a ds9 region file with the objects given
	sizes can be an array or a scalar """

	ras = np.asarray( ras_in ).ravel()
	decs = np.asarray( decs_in ).ravel()
	sizes = np.asarray( sizes_in ).ravel()
	n = ras.size

	if n != decs.size: raise NameError( 'Size mismatch between ras and decs' )
	if sizes.size != n and sizes.size > 1: raise NameError( 'Size mismatch between ras and sizes' )
	if titles is not None and len(titles) != n: raise NameError( 'Size mismatch between ras and titles' )
	if sizes.size == 1: sizes = sizes.repeat( n )

	lines = ['']*n
	title = ''
	for i in range( n ):
		if titles is not None: title = ' # text={%s}' % (titles[i])
		#lines[i] = "circle(%f,%f,%f)%s" % (ras[i],decs[i],sizes[i],title)
		lines[i] = "point(%f,%f)%s" % (ras[i],decs[i],title)
	fh = open( filename, 'w' )
	fh.write( "# Region file format: DS9 version 4.0\nglobal color=%s\nFK5\n" % color )
	fh.write( "\n".join( lines ) )
	fh.close()

def swap_extension( string, find, replace ):
	""" string = swap_extension( input, find, replace )

	Replaces extension `find` for string `input` and replaces it with `replace`.
	Appends replace if extension is not found. """

	# if string is shorter than find, then the extension obviously isn't there
	if len( string ) < len( find ): return '%s%s' % (string,replace)

	# check if extension matches find and replace if so
	if string[-1*len(find):] == find: return '%s%s' % (string[:-1*len(find)],replace)

	# otherwise append
	return '%s%s' % (string,replace)

def load_data_catalog( file, config ):
	""" res = load_data_catalog( file, config )
	
	Used by pygfit and sim_generator to load the users object catalogs.
	Pass the filename, and the configuration (a dictionary) describing data columns.
	This detects a fits or ascii catalog and calls the appropriate catalog reader. """

	# is this a fits catalog or an ascii catalog?
	is_fits = False
	fp = open( file, 'r' )
	if re.search( 'SIMPLE\s+=\s+T', fp.read( 80 ), re.IGNORECASE ): is_fits = True
	fp.close()

	# now read in the catalog
	if is_fits:
		res = load_fits( file, config )
	else:
		res = load_ascii( file, config )

	return res

def sigrej( vals, sigma=3, max_reject=0.3, return_mask=False ):
	""" std = sigrej( values, sigma=3, max_reject=0.3, return_mask=False )
	
	or
	
	( std, keep_mask ) = sigrej( vals, sigma=3, max_reject=0.3, return_mask=True ) """
	
	if max_reject < 0 or max_reject > 1: max_reject = 0.1
	vals = np.asarray( vals )
	n = vals.size
	keep = np.array( [True]*n )
	last = n

	while True:
		std = np.std( vals[keep] )
		mean = np.mean( vals[keep] )
		keep = ( np.abs( vals-mean ) < std*sigma ) & ( keep )

		nleft = keep.sum()
		if ( float( nleft )/n < 1.0-max_reject ) or ( nleft == last ): break

		last = nleft

	std = np.std( vals[keep] )

	if return_mask:
		return (std,keep)
	else:
		return std

def load_fits( file, config ):
	""" utils.load_fits( file, config )

	Loads the users fits object catalog.
	Pass the filename, and the configuration (a dictionary) describing data columns. """

	catalog = {}

	# open up the file
	fits = pyfits.open( file )

	# look for a binary hdu with the appropriate fields
	check = ['ra','dec','mag','point_mag','re','n','pa','ba']

	use = None
	for (i,ext) in enumerate(fits):

		# don't bother checking if it isn't a binary fits table
		if type( ext ) != type( pyfits.BinTableHDU() ): continue

		# does it have all the necessary fields?
		# copy fits column names but in lowercase so they can be searched in a case insensitive way
		names = []
		for field in ext.data.names: names.append( field.lower() )

		missing = ''
		for field in check:
			if not config[field].lower() in names:
				missing = config[field]
				break
		if missing: continue
		# okay, found our field!
		use = i
		break

	# did we find it?
	if use == None:
		if missing:
			raise ValueError( 'Could not load data catalog - could not locate a binary fits table with field "%s"' % missing )
		else:
			raise ValueError( 'Could not load data catalog - could not locate a binary fits table in input catalog' )

	# now we can load the catalog
	cat = fits[use].data
	valid = cat.size

	# and start loading the data!
	# pretend everything is a point source
	# make sure that ids are strings
	ids = []
	for i in range( cat.size ): ids.append( str( cat.field(config['id'])[i] ) )
	catalog['id'] = np.asarray( ids )
	catalog['ra'] = cat.field(config['ra'])
	catalog['dec'] = cat.field(config['dec'])

	# sersic fields to load
	sersics = ['re','n','pa','ba','re_err','n_err','pa_err','ba_err']
	for field in sersics:
		if config.has_key( field ): catalog[field] = np.zeros( valid )

	catalog['model'] = np.asarray( ['point']*valid ).astype( '|S6' )	# numpy string arrays carry a length - expand to make room for 'sersic'
	if config.has_key( 'point_mag' ):
		catalog['mag'] = cat.field(config['point_mag'])
		if config.has_key( 'point_mag_err' ): catalog['mag_err'] = cat.field(config['point_mag_err'])
	else:
		catalog['mag'] = cat.field(config['mag'])
		if config.has_key( 'mag_err' ): catalog['mag_err'] = cat.field(config['mag_err'])

	# figure out what is a sersic and what is a point source
	m = cat.field(config['model_type']) == 'sersic'
	p = cat.field(config['model_type']) == 'point'
	# number of unrecognized model types
	unknown = valid - m.sum() - p.sum()

	# load sersic-specific values
	if m.sum() > 0:
		catalog['mag'][m] = cat.field(config['mag'])[m]
		catalog['model'][m] = ['sersic']
		if config.has_key( 'mag_err' ) and catalog.has_key( 'mag_err' ): catalog['mag_err'][m] = cat.field(config['mag_err'])[m]
		for field in sersics:
			if config.has_key( field ): catalog[field][m] = cat.field(config[field])[m]

	# return catalog and counts
	return ( catalog, valid, unknown )

def load_ascii( file, config ):
	""" utils.load_ascii( file, config )

	Loads the users ascii object catalog.
	Pass the filename, and the configuration (a dictionary) describing data columns. """

	catalog = {'id':[], 'model':[], 'ra':[], 'dec':[], 'mag':[], 're':[], 'n':[], 'pa':[], 'ba':[]}
	valid = 0
	unknown = 0

	# are we including errors?
	for field in ['mag_err','re_err','n_err','pa_err','ba_err']:
		if config.has_key( field ): catalog[field] = []
	if config.has_key( 'point_mag_err' ): catalog.setdefault( 'mag_err', [] )

	# sersic fields, and whether or not to check for their existence before loading
	sersics = {'re': False, 'n':False, 'pa':False, 'ba':False, 're_err':True, 'n_err':True, 'pa_err':True, 'ba_err':True}

	# open up the file and start reading...
	fp = open( file, 'r' )
	for line in fp:
		line = line.strip()
		if line[0] == '#': continue
		parts = line.split()

		valid += 1
		# check ra/dec
		ra = float( get_field( 'ra', parts, config['ra'] ) )
		dec = float( get_field( 'dec', parts, config['dec'] ) )

		# fetch id
		id = get_field( 'id', parts, config['id'] )

		model = get_field( 'model type', parts, config['model_type'] ).lower()
		if model != 'sersic' and model != 'point':
			unknown += 1
			continue

		# now store the data
		catalog['id'].append( id )
		catalog['model'].append( model )
		catalog['ra'].append( ra )
		catalog['dec'].append( dec )
		if model == 'sersic':
			# load mags
			catalog['mag'].append( get_field( 'mag', parts, config['mag'] ) )
			if config.has_key( 'mag_err' ): catalog['mag_err'].append( get_field( 'mag err', parts, config['mag_err'] ) )
			# load the rest of the sersic values
			for (field,check) in sersics.iteritems():
				if not check or config.has_key( field ): catalog[field].append( get_field( field, parts, config[field] ) )
		else:
			# load mags
			if config.has_key( 'point_mag' ):
				catalog['mag'].append( get_field( 'point mag', parts, config['point_mag'] ) )
				if config.has_key( 'mag_err' ): catalog['mag_err'].append( get_field( 'mag err', parts, config['mag_err'] ) )
			else:
				catalog['mag'].append( get_field( 'mag', parts, config['mag'] ) )
				if config.has_key( 'point_mag_err' ): catalog['point_mag_err'].append( get_field( 'point mag err', parts, config['point_mag_err'] ) )

			# default values for sersic fields
			for (field,check) in sersics.iteritems():
				if not check or config.has_key( field ): catalog[field].append( 0 )

	# convert to a numpy array
	catalog['id'] = np.array( catalog['id'] )
	catalog['model'] = np.array( catalog['model'] )
	catalog['ra'] = np.array( catalog['ra'] )
	catalog['dec'] = np.array( catalog['dec'] )
	catalog['mag'] = np.array( catalog['mag'] ).astype( float )
	if catalog.has_key( 'mag_err' ): catalog['mag_err'] = np.array( catalog['mag_err'] ).astype( float )
	# everything else needs to become a float array
	for (field,check) in sersics.iteritems():
		if not check or config.has_key( field ): catalog[field] = np.array( catalog[field] ).astype( float )

	return ( catalog, valid, unknown )

def get_field( name, data, index ):
	index = int( index )
	if index < len( data ): return data[index]
	raise ValueError( 'Specified column index for %s field does not exist in data catalog!' % name )

def output_fits_catalog( cat, filename, output_columns, more_formats=None ):
	""" utils.output_fits_catalog( cat, filename, output_columns, more_formats=None ):
	
	Outputs a catalog in fits format """

	# pyfits formats for the various fields
	formats = {'lres_id': 'K', 'model': 'A6', 'nblend': 'K', 'nearest': 'D', 'nearest_mag': 'D', 'ra': 'D', 'dec': 'D', 'x': 'D', 'y': 'D', 'img_x': 'D', 'img_y': 'D', 'flags': 'D', 'flux': 'D', 'total_flux': 'D', 'total_mag': 'D', 'blend_fraction': 'D', 'flux_real': 'D', 'flux_input': 'D', 'mag': 'D', 'mag_image': 'D', 'mag_initial': 'D', 'mag_hres': 'D', 'mag_input': 'D', 'mag_brightest': 'D', 'mag_auto': 'D', 'mag_real': 'D', 'mag_err': 'D', 'sky': 'D', 're_hres': 'D', 're_lres': 'D', 're_arcsecs': 'D', 're_arcsecs_err': 'D', 're_arcsecs_real': 'D', 're_pix': 'D', 're_pix_err': 'D', 're_pix_real': 'D', 're': 'D', 're_real': 'D', 're_err': 'D', 'n': 'D', 'n_real': 'D', 'n_err': 'D', 'pa': 'D', 'pa_real': 'D', 'pa_err': 'D', 'ba': 'D', 'ba_real': 'D', 'ba_err': 'D', 'chisq_nu': 'D', 'chisq': 'D', 'nf': 'K', 'frame': 'K', 'segmentation_mag': 'D', 'segmentation_residuals': 'D', 'segmentation_fraction': 'D', 'mask_mag': 'D', 'mask_residuals': 'D', 'mask_fraction': 'D', 'mag_warning': 'K'}
	
	# add on any additional fields
	if more_formats is not None:
		if type( more_formats ) != type( {} ): raise ValueError( 'Cannot output fits catalog - additional formats must be a dictionary!' )
		formats = dict( more_formats.items() + formats.items() )

	# Set format for string fields: what is the maximum length of it in the catalog?
	fields = ['hres_id','id']
	for field in fields:
		if output_columns.count( field ):
			formats[field] = 'A'
			max_len = 0
			for i in range( len( cat[field] ) ): max_len = max( [max_len,len( cat[field][i] )] )
			formats[field] += '%d' % max_len

	# generate pyfits columns
	cols = []
	for key in output_columns:
		format_string = formats[key]
		cols.append( pyfits.Column( name=key, format=format_string, array=cat[key] ) )

	# generate pyfits table
	tbhdu = pyfits.new_table( cols )

	# and write it out! (delete if it exists to avoid clobber notice)
	if os.path.isfile( filename ): os.remove( filename )
	tbhdu.writeto( filename )

def output_ascii_catalog( cat, filename, output_columns, more_formats=None ):
	""" utils.output_ascii_catalog( cat, filename, output_columns, more_formats=None ):
	
	Outputs a catalog in ascii format """

	format_list = {'model': '%6s', 'ra': '%14.9f', 'dec': '%14.9f', 'flux': '%12.6e', 'total_flux': '%12.6e', 'flux_real': '%12.6e', 'flux_input': '%12.6e', 'sky': '%12.6e', 'mag': '%7.4f', 'mag_image': '%7.4f', 'mag_initial': '%7.4f', 'mag_hres': '%7.4f', 'mag_input': '%7.4f', 'mag_brightest': '%7.4f', 'mag_real': '%7.4f', 'mag_err': '%6.4f', 'total_mag': '%7.4f', 'nearest_mag': '%7.4f', 'segmentation_mag': '%7.4f', 'mask_mag': '%7.4f', 'mag_warning': '%d'}

	# Set format for string fields: what is the maximum length of it in the catalog?
	fields = ['hres_id','id']
	for field in fields:
		if output_columns.count( field ):
			max_len = 0
			for i in range( len( cat[field] ) ): max_len = max( [max_len,len( cat[field][i] )] )
			format_list[field] = '%%-%ds' % max_len

	# for these fields calculate the formats from the actual data, given the desired digits.
	# the goal is to make the output file look nice
	to_calc = {'nblend': 0, 'nearest': 4, 'x': 3, 'y': 3, 'img_x': 3, 'img_y': 3, 'mag_auto': 4, 'flags': 0, 're_hres': 4, 're_lres': 4, 're_arcsecs': 4, 're_arcsecs_err': 4, 're_arcsecs_real': 4, 're': 4, 're_pix': 4, 're_pix_real': 4, 're_pix_err': 4, 're_real': 4, 're_err': 4, 'n': 4, 'n_real': 4, 'n_err': 4, 'pa': 4, 'pa_real': 4, 'pa_err': 4, 'ba': 4, 'ba_real': 4, 'ba_err': 4, 'chisq': 2, 'chisq_nu': 4, 'nf': 0, 'lres_id': 0, 'frame': 0, 'segmentation_residuals': 4, 'segmentation_fraction': 6, 'mask_residuals': 4, 'mask_fraction': 6, 'blend_fraction': 4}

	# add on any additional fields
	if more_formats is not None:
		if type( more_formats ) != type( {} ): raise ValueError( 'Cannot output ascii catalog - additional formats must be a dictionary!' )
		to_calc = dict( more_formats.items() + to_calc.items() )

	for key in to_calc.keys():

		# don't bother calculating it unless we are outputting it
		if not output_columns.count( key ): continue

		# how long does the field have to be?  Start with the number of digits before the decimal
		maxval = max( np.abs( cat[key] ) )
		length = np.int( np.log10( maxval ) ) + 1 if maxval > 0 else 1

		# add one if we go into the negatives
		if min( cat[key] ) < 0: length += 1

		# now put together the format string
		if to_calc[key] > 0:
			# add one for the decimal, and then one digit for each digit desired after the decimal
			length += 1 + to_calc[key]
			format = '%%%d.%df' % (length,to_calc[key])
		else:
			format = '%%%dd' % length

		# and store
		format_list[key] = format

	fp = open( filename, 'wb' )

	# write the header
	for (i,key) in enumerate( output_columns ): fp.write( '# %02d %s\n' % (i+1,key) )

	# build the file
	nrows = len( cat[output_columns[0]] )
	strings = ['']*nrows
	for i in range( nrows ):
		strings[i] = ' '.join( [ format_list[key] % cat[key][i] for key in output_columns ] )

	# and write it out!
	fp.write( '\n'.join( strings ) )

def load_psf( file ):
	""" utils.load_psf( file ) """

	# load psf
	if type( file ) == type( '' ):
		if os.path.isfile( file ):
			psf = load_image( file )
		else:
			raise ValueError( 'Cannot find psf file: %s' % file )
	elif type( file ) == type( np.array( [] ) ):
		psf = file.copy()
	else:
		raise ValueError( 'Unrecognized psf!' )

	# normalize
	psf /= psf.sum()

	# and now make it square
	(height,width) = psf.shape

	diff = height - width
	if diff > 0:
		psf = np.hstack( (psf,np.zeros( (height,diff) )) )
		width = height
	elif diff < 0:
		psf = np.vstack( (psf,np.zeros( (-1*diff,width) )) )
		height = width

	# make sure it is odd
	if width % 2 == 0: psf = np.vstack( (np.hstack( (psf,np.zeros( (height,1) )) ),np.zeros( (1,width+1) )) )

	# avoid scipy type problems by copying the psf array onto a numpy array
	tmp = np.zeros( psf.shape )
	tmp[:,:] = psf[:,:]

	# also make it a little endian float 32 so that it is compatiable with the GPU
	gpu_obj = gpu.gpu()
	tmp = gpu_obj.force_float32( gpu_obj.force_le( tmp ) )

	# and then return
	return tmp

def fix_extractor_seg_bug( seg, number ):
	""" seg = utils.fix_extractor_seg_bug( seg, number )
	
	Pass a segmentation image and id.  This will check for
	and fix a source extractor bug that joins distant
	pixels in the segmentation image.
	
	This does not actually find connected pixels as that can
	be computationally expensive.  Instead it compresses everything
	to one dimension (radius) and finds any pixels that are more than
	a pixel away from the nearest pixel in radius space.  This
	should be much faster.  This simpler methodology might miss some
	pixels that are disconnected in asymetrical segmentation images,
	but in such cases the segmentation region will encompass such
	disconnected pixels anyway... """
	
	# figure out what pixels belong to this object
	# also use the indexes as the x/y location of each pixel
	( ys, xs ) = np.where( seg == number )
	
	# now calculate the distance of every pixel from
	# the "center" of the segmentation region
	x = np.median( xs )
	y = np.median( ys )
	dists = np.sqrt( (ys-y)**2.0 + (xs-x)**2.0 )
	
	# and make a sorted distance array
	dists_s = dists[dists.argsort()]
	
	# and find anything that is more than 1 diagonal pixel away from the nearest pixel
	m = (dists_s - np.roll( dists_s, 1 )) > np.sqrt( 2 )
	
	# if we didn't find anything then everything is connected,
	# modulo the exception noted in the doctext.  Therefore, return.
	if not m.sum(): return None
	
	# otherwise mask out all segmentation pixels that are
	# farther away than the nearest disconnected pixel
	max_dist = dists_s[m].min()
	
	# this means that we need to figure out how far every pixel is from the
	# segmentation region center
	( height, width ) = seg.shape
	( ys, xs ) = np.ogrid[0:height,0:width]
	dists = np.sqrt( (ys-y)**2.0 + (xs-x)**2.0 )
	
	# find offending pixels
	m = ( seg == number ) & ( dists >= max_dist )
	
	# and replace
	if m.any():
		seg[m] = 0
	
	# all done
	return None

def gauss( params, xs, ys ):
	a = params[0]
	x0 = params[1]
	y0 = params[2]
	sig = params[3]
	dists = np.sqrt( (xs-x0)**2.0 + (ys-y0)**2.0 )
	return a/(sig*np.sqrt(2*np.pi))*np.exp(-1.0*( (dists**2.0/(2*sig)) ) )

def moffat( params, xs, ys ):
	a = params[0]
	x0 = params[1]
	y0 = params[2]
	alpha = params[3]
	beta = params[4]
	rs = np.sqrt( (xs-x0)**2.0 + (ys-y0)**2.0 )
	return a*(1+(rs/alpha)**2.0)**(-1.0*beta)

def moffat_chi( params, xs, ys, fluxes ):
	return moffat( params, xs, ys ) - fluxes

def gauss_chi( params, xs, ys, fluxes ):
	return gauss( params, xs, ys ) - fluxes

def fit_psf_fwhm( psf, ps_out='' ):
	""" pygfit.utils.fit_psf_fwhm( psf, ps_out='' )
	
	Fits a moffat profile to measure the FWHM of the PSF image.
	Pass the filename of the PSF image (in fits format) or a
	numpy array containing the PSF.
	
	If you specify a filename for ps_out it will plot the profile,
	show the fit, and output a postscript file to the specified filename """
	
	
	# load up the psf
	img = load_psf( psf )
	( height, width ) = img.shape
	
	# pixel locations and fluxes
	( ys, xs ) = np.mgrid[0:height,0:width]
	ys = ys.ravel()
	xs = xs.ravel()
	fluxes = img.ravel()
	
	# fit!
	#res = scipy.optimize.leastsq( gauss_chi, [fluxes.max(),width/2.0,height/2.0,height/5.0], args=(xs,ys,fluxes), full_output=True )
	res = scipy.optimize.leastsq( moffat_chi, [fluxes.max(),width/2.0,height/2.0,height/5.0,1.0], args=(xs,ys,fluxes), full_output=True )
	fit = res[0]
	
	# alpha from moffat profile is supposed to be the fwhm.
	# however, I find interpolation to be a bit more accurate for some reason
	rs = np.linspace( 0, width/2.0, 1000 )
	r_fluxes = moffat( fit, fit[1]+rs, fit[2] )
	hwhm = np.interp( 0.5, r_fluxes[::-1]/moffat( fit, fit[1], fit[2] ), rs[::-1] )
	fwhm = 2.0*hwhm
	
	if not ps_out:
		return fwhm
	
	# bin
	import matplotlib.pyplot as pyplot
	dists = np.sqrt( (xs-fit[1])**2.0 + (ys-fit[2])**2.0 )
	pyplot.plot( dists, fluxes, 'ko', ms=3, label='Profile' )
	pyplot.plot( rs, r_fluxes, 'k-', label='Moffat Fit' )
	pyplot.axvline( hwhm, label='Half width at half max', color='red' )
	pyplot.axis( xmin=0, xmax=3*fwhm )
	pyplot.annotate( 'FWHM = %.2f pixels' % fwhm, (0.60,0.95), (0.60,0.95), xycoords='axes fraction', textcoords='axes fraction' )
	pyplot.xlabel( 'Pixels' )
	pyplot.ylabel( 'Flux' )
	#pyplot.legend()
	pyplot.gcf().set_size_inches( (6,6) )
	pyplot.savefig( ps_out )
	
	return fwhm
