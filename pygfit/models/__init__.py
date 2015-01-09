import sersic,point,mapping

__all__ = ["sersic","point","mapping"]

sersic = sersic.sersic
point = point.point

def get_model( data, zeropoint, max_array_size=1e7, gpu=False ):
	""" model = models.get_model( data, zeropoint, max_array_size=None, gpu=False )
	
	Pass a dictionary containing the properties of a high resolution object, and the low resolution image zeropoint
	Returns the appropriate model object. """

	if ( data['model'].lower() == "point" ):
		return point( data, zeropoint, max_array_size=max_array_size, gpu=gpu )
	elif ( data['model'].lower() == "sersic" ):
		return sersic( data, zeropoint, max_array_size=max_array_size, gpu=gpu )
	else:
		raise ValueError( 'Unrecognized model type: %s' % data['model'] )