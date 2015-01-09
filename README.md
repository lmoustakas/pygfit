pygfit
======
HISTORY
- 2013 Jun  7 - github setup
- 2015 Jan  8 - astropy affiliate package setup

NOTES
- Because of FITS etc examples, increased buffer size was required, http.postBuffer 524288000

Package for utilizing existing quantitative morphology (galfit format) catalogs as positional and structure priors, for measuring accurate photometry in corresponding lower-resolution imaging. 

Input needed:
- Quantitative morphologies catalog (ascii or FITS)
- Low resolution image
- PSF for low resolution image
- RMS map for low resolution image
- Pygfit configuration file

Required python packages:
- astropy
- scipy
- numpy
- matploglib
- pyfits

Also required:
- Source Extractor


