import psf
from astropy.io import ascii
from astropy.io import fits as pyfits

builder=psf.psf('/Users/leonidas/research/projects/pygfittest/M1206/m1206_ch1_mosaic.fits')
builder.inner=40
builder.outer=60

chcat=ascii.read('/Users/leonidas/research/projects/pygfittest/M1206/m1206_ch1.cat')
stix=(chcat['CLASS_STAR']>0.99)
chstars=chcat[stix]

builder.set_stars(chstars['ALPHA_J2000'],chstars['DELTA_J2000'])
psf_img=builder.combine_stars(size=31)

hdu=pyfits.PrimaryHDU(psf_img)
hdu.writeto('testch1psf.fits',clobber=True)
