import numpy as np
import pyfits,os

def find_image_extension( fits, with_wcs=False ):

	for i in range( len( fits ) ):
		header = fits[i].header

		# check for image data
		if not 'naxis' in header or header['naxis'] != 2: continue
		if not 'naxis1' in header or not 'naxis2' in header: continue


		# if image data is found and a wcs isn't required, then return true
		if header['naxis1'] > 0 and header['naxis2'] > 0 and not with_wcs: return i

		# check for wcs data
		if 'crval1' in header and 'crval2' in header and 'crpix1' in header and 'crpix2' in header: return i

	raise ValueError( 'Could not locate image extension!' )

def get_wcs_info( header, extension=None ):
	""" wcs = wcs.get_wcs_info( header, extension=None )
	
	Returns a dictionary with wcs information.
	Pass a pyfit header or filename for fits file.
	If filename is passed, then you can specify the fits extension to use. """

	# assume this is a header unless it is a string
	if type( header ) == type( '' ):
		if not os.path.isfile( header ): raise ValueError( 'Pass a fits header or a filename please!' )
		fits = pyfits.open( header )
		extension = find_image_extension( fits, with_wcs=True )
		header = fits[extension].header

	if 'cd1_1' in header and 'cd1_2' in header and 'cd2_1' in header and 'cd2_2' in header:
		cd = np.array([header['cd1_1'],header['cd1_2'],header['cd2_1'],header['cd2_2']])
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
	wcs = {'scale': scale, 'cd': cd, 'ra0': ra0, 'dec0': dec0, 'x0': header['crpix1'], 'y0': header['crpix2'], 'det':det, 'inv':inv, 'rot_angle': rot_angle, 'cdelt': cdelt, 'width': header['naxis1'], 'height': header['naxis2']}

	# and return
	return wcs

def rd2xy( wcs, ra, dec, extension=None ):
	""" (x,y) = wcs.rd2xy( wcs, ra, dec, extension=0 )
	
	Pass either a wcs dictionary returned by get_wcs_info(), a fits header, or filename.
	If filename is passed, then you can specify the fits extension to use.
	"""

	# assume this is a wcs dictionary if it is a dict, otherwise pass to get_wcs_info
	if type( wcs ) != type( {} ): wcs = get_wcs_info( wcs, extension=extension )

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

def rd2xy_ind( wcs, ra, dec, extension=None ):
	""" (x_ind,y_ind) = wcs.rd2xy_ind( wcs, ra, dec, extension=0 )
	
	Same calling sequence as wcs.rd2xy().  Returns the indexes of the pixels that the given ras and decs land on. """

	( x, y ) = rd2xy( wcs, ra, dec, extension=extension )

	return ( np.floor( x - 0.5 ).astype( 'int' ), np.floor( y - 0.5 ).astype( 'int' ) )

def xy2rd( wcs, x, y, extension=None ):
	""" (ra, dec) = wcs.xy2rd( x, y )

	Pass either a wcs dictionary returned by get_wcs_info(), a fits header, or filename.
	If filename is passed, then you can specify the fits extension to use.
	"""

	# assume this is a wcs dictionary if it is a dict, otherwise pass to get_wcs_info
	if type( wcs ) != type( {} ): wcs = get_wcs_info( wcs, extension=extension )

	# and do the conversion...
	xdiff = np.asarray( x ) - wcs['x0']
	ydiff = np.asarray( y ) - wcs['y0']
	xi = np.radians( wcs['cd'][0] * xdiff + wcs['cd'][1] * ydiff )
	eta = np.radians( wcs['cd'][2] * xdiff + wcs['cd'][3] * ydiff )

	ra = np.degrees( np.arctan2( xi, np.cos(wcs['dec0']) - eta*np.sin(wcs['dec0']) ) + wcs['ra0'] )
	dec = np.degrees( np.arctan2( eta*np.cos(wcs['dec0']) + np.sin(wcs['dec0']), np.sqrt( (np.cos(wcs['dec0']) - eta*np.sin(wcs['dec0']) )**2 + xi**2) ) )

	ra = np.mod( ra, 360. )

	return (ra, dec)
