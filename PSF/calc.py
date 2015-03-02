import numpy as np

def sigrej( vals, sigma=3, max_reject=0.3, return_mask=False ):
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