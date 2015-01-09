import numpy as np
import data_histogram
import calc

def match( x1, y1, x2, y2, radius=1, degrees=False ):
	""" ( inds_1, inds_2 ) = match( x1, y1, x2, y2, radius=1, degrees=False )

	Match coordinates in (x1,y1) against (x2,y2) with given matching radius.
	arrays will be flattened
	"""

	x1 = np.asarray( x1 ).copy()
	x2 = np.asarray( x2 ).copy()
	y1 = np.asarray( y1 ).copy()
	y2 = np.asarray( y2 ).copy()

	if x1.ndim > 1: x1 = x1.ravel()
	if x2.ndim > 1: x2 = x2.ravel()
	if y1.ndim > 1: y1 = y1.ravel()
	if y2.ndim > 1: y2 = y2.ravel()

	if x1.size != y1.size: raise NameError('Array Mismatch: x1 & y1')

	if x2.size != y2.size: raise NameError('Array Mismatch: x2 & y2')

	# bin largest array - make that (x2,y2)
	is_swapped = False
	if x1.size > x2.size:
		(x1, y1, x2, y2) = (x2, y2, x1, y1)
		is_swapped = True

	# bin x2
	hist = data_histogram.data_histogram( x2, bin=radius )
	if hist.nbins < 3: raise NameError('The data has been divided into less than 3 bins - choose a smaller matching radius')

	# figure out what bin every x1 is in
	xbin = np.floor( (x1 - hist.min)/hist.bin )

	# store the matches
	match1 = []
	match2 = []

	# now loop through all x1
	for (i,bin) in enumerate( xbin ):

		# skip this if it can't match anything
		if (bin < -1) | (bin > hist.nbins): continue

		# figure out if we need to look in bins left and right of this one
		fetch = bin
		if bin > 0: fetch = np.hstack( (fetch, bin-1) )
		if bin < hist.nbins-2: fetch = np.hstack( (fetch, bin+1) )
		# remove the original "bin" if it is out of bounds
		if (bin < 0) | (bin >= hist.nbins): fetch = fetch[1:]

		# fetch the necessary bins
		fetch = fetch.astype( 'int' )
		inds = hist.fetch_bins( fetch )

		# if we didn't find anything than there can't be a match - continue
		if len(inds) == 0: continue

		# calculate the distance from this point to all the nearby ones
		if degrees:
			dist = np.sqrt( (x2[inds]-x1[i])**2.0 * np.cos(y1[i]/180.0*np.pi)**2.0 + (y2[inds]-y1[i])**2.0 )
		else:
			dist = np.sqrt( (x2[inds]-x1[i])**2.0 + (y2[inds]-y1[i])**2.0 )

		# find the closest point
		minind = dist.argmin()

		# continue if the closest point isn't within our search radius
		if dist[minind] > radius: continue

		# store this result!
		match1.append( i )
		match2.append( inds[minind] )

	# return the result!  If we swapped the lists earlier, swap the results to return them in the proper order
	if is_swapped:
		return ( np.asarray( match2 ), np.asarray( match1 ) )
	else:
		return ( np.asarray( match1 ), np.asarray( match2 ) )

def refine( x1, y1, x2, y2, cut=3, radius=1, degrees=False, with_shifts=False ):
	"""( inds_1, inds_2 ) = refine( x1, y1, x2, y2, radius=1, degrees=False, cut=3, with_shifts=False )
	( inds_1, inds_2, offx, scatx, offy, scaty ) = refine( x1, y1, x2, y2, radius=1, degrees=False, cut=3, with_shifts=True )

	Match coordinates in (x1,y1) against (x2,y2) with given matching radius.
	After initial match, runs a 3-sigma rejection algorithm to find mean offset in x&y and scatter.
	Then rematches everything within cut*scatter of those measured means.
	"""

	( m1, m2 ) = match( x1, y1, x2, y2, radius=radius, degrees=degrees )

	if len( m1 ) == 0:
		if with_shifts: return ( m1, m2, 0, 0, 0, 0 )
		return ( m1, m2 )

	# measure mean differnce/scatter in both directions
	stdx,mx = calc.sigrej( x1[m1]-x2[m2], return_mask=True )
	offx = np.median( x1[m1[mx]]-x2[m2[mx]] )
	stdy,my = calc.sigrej( y1[m1]-y2[m2], return_mask=True )
	offy = np.median( y1[m1[my]]-y2[m2[my]] )
	# which gives a nice guess for the matching radius...
	if degrees:
		cut *= np.sqrt( ( stdx*np.cos( np.mean( y1 )/180*np.pi ) )**2.0 + stdy**2.0 )
	else:
		cut *= np.sqrt( stdx**2.0 + stdy**2.0 )

	# now rematch
	( m1, m2 ) = match( x1-offx, y1-offy, x2, y2, radius=cut, degrees=True )

	if with_shifts: return ( m1, m2, offx, stdx, offy, stdy )
	return( m1, m2 )

def nearest( x1, x2, left=False, right=False ):
	""" inds = match( x1, x2, left=False, right=False )

	Find the element in x1 nearest to each element in x2
	left = True: find the nearest element on the left (i.e. the next lowest)
	right = True: find the nearest element on the right (i.e. the next highest)
	returns the index of the nearest element in x1 for each element in x2

	When searching left it will always return a valid index.
	So if a value in x2 is lower than the lowest value in x1
	it will reutrn the lowest value in x1, even if that is higher than the x2 value.
	The opposite holds true for searching right

	arrays will be flattened
	"""

	if (left & right):
		left = False
		right = False

	x1 = np.asarray( x1 ).copy()
	x2 = np.asarray( x2 ).copy()

	if x1.ndim > 1: x1 = x1.ravel()
	if x2.ndim > 1: x2 = x2.ravel()
	nx1 = x1.size
	nx2 = x2.size
	ntot = nx1+nx2

	# minimum index from x1
	x1_min = x1.argmin()
	# maximum index from x1
	x1_max = x1.argmax()

	# combine everything into one array
	xs = np.append( x1, x2 )

	# get the indexes to sort this array
	sinds = xs.argsort( kind='mergesort' )

	# now sort that to find out where everything went
	lookup = sinds.argsort()

	# now find out where all the x2 elements went
	landing = lookup[nx1:]

	if not right:
		inds = landing - 1

		# find out of bounds elements
		wout = inds<0
		win = ~wout

		# figure out what is to the left
		left_res = np.empty( nx2, dtype=np.int )
		left_res[win] = sinds[inds[win]]
		left_res[wout] = x1_min

		# see if anything is pointing to x2
		omask = (left_res >= nx1)

		# if something is pointing to an element in x2, keep rotating the array until it dissapears
		while omask.sum():
			inds[omask] -= 1
			these_in = (omask & (inds>=0))
			these_out = (omask & (inds<0))
			if these_in.sum(): left_res[these_in] = sinds[inds[these_in]]
			if these_out.sum(): left_res[these_out] = x1_min
			omask = (left_res >= nx1)

		if left: return left_res

	if not left:
		inds = landing + 1

		# find out of bounds elements
		wout = inds>=ntot
		win = ~wout

		# figure out what is to the right
		right_res = np.empty( nx2, dtype=np.int )
		right_res[win] = sinds[inds[win]]
		right_res[wout] = x1_max

		# see if anything is pointing to x2
		omask = (right_res >= nx1)

		# if something is pointing to an element in x2, keep rotating the array until it dissapears
		while omask.sum():
			inds[omask] += 1
			these_in = (omask & (inds<ntot))
			these_out = (omask & (inds>=ntot))
			if these_in.sum(): right_res[these_in] = sinds[inds[these_in]]
			if these_out.sum(): right_res[these_out] = x1_max
			omask = (right_res >= nx1)

		if right: return right_res

	# if we're still executing then the user wants the nearest in either direction
	dist = np.abs( np.vstack( (x1[left_res],x1[right_res]) ) - x2 )
	final = left_res
	from_right = (dist.argmin(axis=0) == 1)
	final[from_right] = right_res[from_right]

	return final

def match_ids( ids1, ids2 ):
	""" pretty slow and crappy """

	w1 = []
	w2 = []

	for (i,val) in enumerate(ids1):
		for (j,val2) in enumerate(ids2):
			if val == val2:
				w1.append( i )
				w2.append( j )
				break

	return ( np.asarray( w1 ), np.asarray( w2 ) )

#def match_ids( ids1, ids2 ):
	""" ( indexes1, indexes2 ) = match_ids( ids1, ids2 )

	given two lists of unique, integer ids, return two lists of indexes for matching ids
	"""

	# make sure we have numpy arrays
	#l1 = np.asarray( ids1 ).ravel()
	#l2 = np.asarray( ids2 ).ravel()

	## make a list which tells you what each index each entry was in its original list
	#inds = np.hstack( (np.arange( l1.size ),np.arange( l2.size )) )

	## now combine lists and sort (preserve order)
	#all = np.hstack( (l1,l2) )
	#sinds = all.argsort( kind='mergesort' )
	#sorted = all[sinds]

	## now subtract the sorted list from itself rolled by 1 - matching things will have a difference of zero
	#diff = sorted - np.roll( sorted, -1 )
	#w = np.where( diff == 0 )[0]

	## and return the indexes of those matching things and the element after them
	#return ( inds[sinds[w]], inds[sinds[w+1]] )

#def match_ids( ids1, ids2 ):
	""" ( indexes1, indexes2 ) = match_ids( ids1, ids2 )

	given two lists of unique, integer ids, return two lists of indexes for matching ids
	"""

	#inds = nearest( ids1, ids2 )

	#w = np.where( ids1[inds] == ids2 )[0]
	#return ( inds[w], w )