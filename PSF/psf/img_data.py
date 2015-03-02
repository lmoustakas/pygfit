import wcs,pyfits
import numpy as np

img_file = ''	# filename for image
img = []	# image array
xs = []		# x coordinates for pixels
ys = []		# y coordinates for pixels
wcs_info = {}	# dictionary with WCS info

def set_img( file, extension ):
	global img,img_file,wcs_info,xs,ys

	# store filename
	img_file = file

	# load image
	fits = pyfits.open( img_file )
	
	# find image extension
	if extension is None:
		extension = wcs.find_image_extension( fits, True )
	
	img = fits[extension].data

	# store wcs info
	wcs_info = wcs.get_wcs_info( fits[extension].header )

	# store pixel coordinates
	( ysize, xsize ) = img.shape
	#( ys, xs ) = np.mgrid[1:ysize+1,1:xsize+1]

def rd2xy( ras, decs ):
	global wcs_info

	( xs, ys ) = wcs.rd2xy( wcs_info, ras, decs )
	return ( xs, ys )

def xy2rd( xs, ys ):
	global wcs_info

	return wcs.xy2rd( wcs_info, xs, ys )