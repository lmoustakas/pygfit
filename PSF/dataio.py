import re,pyfits,math
import numpy as np

def rascii( filename, silent=False, integer=False, debug=False ):
	""" Reads in numeric data stored in ascii format
	
	array = rascii( 'filename'[, silent=True/False, integer=True/False, debug=True/False] )
	Skips any lines that have non numeric data
	Any data lines with different number of columns will be skipped
	The number of columns are determined by the first valid data row
	"""
	file = open( filename, 'r' )
	found = False
	nlines = 0
	ngood = 0

	for line in file:

		nlines += 1
		if re.search( '^\s*$', line ) or re.search( '[^\s\d.eEdD\-+]', line ):
			continue

		parts = line.split()
		nparts = len( parts )

		if not found:
			found = True
			allowed = nparts
			ngood += 1
			res = parts
			if debug: print line
			continue

		if nparts != allowed:
			if debug: print line
			continue

		ngood += 1
		res.extend( parts )

	file.close()

	if ngood == 0:
		print "no data found!"
		return

	if not silent:
		print nlines, " lines"
		print ngood, " kept"

	arr = np.array( res )
	arr.shape = (ngood,-1)

	if integer: return arr.astype('int')
	return arr.astype('float')

def wascii( array, filename, formats, blank=False, header=None, names=None ):
	""" Writes out a np array to a well formatted file """

	table = np.asarray( array )
	if table.ndim != 2: raise NameError( 'I was expecting a 2D data table' )
	nrows,ncols = table.shape

	if type( formats ) is str:
		formats = [formats]*ncols

	if ncols != len( formats ): raise NameError( 'Number of supplied formats does not match number of table columns!' )

	# if column names were provided, create a header that list column names/numbers
	if names is not None:
		if len( names ) != ncols: raise NameError( 'Number of supplied column names does not match number of table columns!' )
		if header is None: header = []
		header.append( '# Column Descriptions:' )
		name_format = '# %0' + ('%1d' % (math.ceil(math.log10(ncols)))) + 'd: %s'
		for i in range(ncols):
			header.append( name_format % (i+1,names[i]) )

	if ( header is not None ) & ( type( header ) != type( '' ) ): header = "\n".join( header )

	if ncols == 1:
		file = "\n".join( formats[0] % val for val in table.ravel() )
	else:
		strings = ['']*nrows
		for i in range(nrows):
			strings[i] = ' '.join([format % val for format,val in zip(formats, table[i,:])])
		file = "\n".join( strings )

	fh = open( filename, 'wb' )
	if header is not None: fh.write( header + "\n" )
	fh.write( file )
	if blank: fh.write( "\n" )
	fh.close()

def wascii_mixed( table, filename, formats, blank=False, header=None, names=None ):
	""" Writes out a list of lists (or arrays) to a well formatted file """

	ncols = len( table )
	nrows = len( table[0] )

	# check that the lengths all match
	for i in range( ncols ):
		if len( table[i] ) != nrows: raise ValueError( 'All lists must have the same length!' )

	if type( formats ) == type( '' ):
		formats = [formats]*ncols

	if ncols != len( formats ): raise NameError( 'Number of supplied formats does not match number of table columns!' )

	# if column names were provided, create a header that list column names/numbers
	if names is not None:
		if len( names ) != ncols: raise NameError( 'Number of supplied column names does not match number of table columns!' )
		if header is None: header = []
		header.append( '# Column Descriptions:' )
		name_format = '# %0' + ('%1d' % (math.ceil(math.log10(ncols)))) + 'd: %s'
		for i in range(ncols):
			header.append( name_format % (i+1,names[i]) )

	if ( header is not None ) & ( type( header ) != type( '' ) ): header = "\n".join( header )

	if ncols == 1:
		file = "\n".join( formats[0] % val for val in table.ravel() )
	else:
		strings = ['']*nrows
		for i in range( nrows ):
				strings[i] = ' '.join( [ formats[j] % table[j][i] for j in range( ncols ) ] )
		file = "\n".join( strings )

	fh = open( filename, 'wb' )
	if header is not None: fh.write( header + "\n" )
	fh.write( file )
	if blank: fh.write( "\n" )
	fh.close()

def write_fits_table( tbl, names, filename, formats=False ):

	table = np.asarray( tbl )

	if table.ndim != 2: raise NameError( 'I was expecting a 2D data table' )

	nrows,ncols = table.shape

	if ncols != len( names ): raise NameError( 'Number of supplied names does not match number of table columns' )

	if not formats:
		if table.dtype.kind == 'i': formats = ['K']*ncols
		elif table.dtype.kind == 'f': formats = ['D']*ncols
		elif table.dtype.kind == 'S': formats = ['A']*ncols
		else: raise NameError( "Couldn't figure out what type of data this is - please provide column formats" )

	cols = [0]*ncols

	for i in range( ncols ):
		cols[i] = pyfits.Column( name=names[i], format=formats[i], array=table[:,i] )

	tbhdu = pyfits.new_table( cols )

	tbhdu.writeto(filename, clobber=True)

def get_pyfits_array( data, integer=False ):
	""" Takes a pyfits data table and returns a np array """

	names = data.names
	nfields = len( names )
	nrows, = data.shape

	if integer: arr = np.empty( (nrows,nfields), dtype='int' )
	else: arr = np.empty( (nrows,nfields), dtype='float' )

	for i in range(nfields): arr[:,i] = data.field(names[i])

	return arr

def write_region( filename, ras_in, decs_in, sizes_in, titles=None, color='green' ):
	""" dataio.write_region( filename, ras, decs, sizes, titles=None, color='green' )

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
		lines[i] = "circle(%f,%f,%f)%s" % (ras[i],decs[i],sizes[i],title)
	fh = open( filename, 'w' )
	fh.write( "# Region file format: DS9 version 4.0\nglobal color=%s\nFK5\n" % color )
	fh.write( "\n".join( lines ) )
	fh.close()

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

def shrink_fits_cat( fits_in, fits_out, m, index=1 ):

	if type( fits_in ) == type( '' ):
		fits = pyfits.open( fits_in )
	else:
		fits = fits_in

	cat = fits[index].data
	ncols = len( cat.names )
	# new columns list
	cols = []

	# fits table parameters
	params = {'name': 'TTYPE', 'format': 'TFORM', 'unit': 'TUNIT', 'null': 'TNULL', 'disp': 'TDISP', 'dim': 'TDIM'}

	# loop through columns and regenerate
	for i in range( ncols ):

		# pass all column properties to the new pyfits.Column.  Fetch them from header
		my_params = {}
		for (key,val) in params.iteritems():
			this_name = '%s%d' % (val,i+1)
			my_params[key] = fits[index].header[this_name] if this_name in fits[index].header else None

		# now generate column
		cols.append( pyfits.Column( array=cat.field(my_params['name'])[m], name=my_params['name'], format=my_params['format'], unit=my_params['unit'], null=my_params['null'], disp=my_params['disp'], dim=my_params['dim'] ) )

	# generate fits header
	tbhdu = pyfits.new_table( cols )

	# and write it out
	tbhdu.writeto( fits_out, clobber=True )
