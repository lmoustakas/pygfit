#!/usr/bin/python

from math import *
import pyraf
from pyraf import iraf
from pyraf.iraf import gemini
from pyraf.iraf import tables
import sys,os,string
import pyfits
from numpy import *

#  Combines pygfit catalogs from multiple passbands with the high resolution catalog for
#   output catalogs in FITS_LDAC format.
#  Output file will have all bands, and -999 for non-detections in the pygfit bands.


def pygfit_match(cat,pygcat,newcat,filter):
   catalog = pyfits.open(cat)
   cat2=catalog[1].data
   pygcat = pyfits.open(pygcat)[1].data
   galapagosid=cat2['NUMBER']
   pygcatid=pygcat['hres_id']

   #1. Establish the output arrays with -999 values
   pygfit_mag=zeros(len(galapagosid))-999
   pygfit_blends=zeros(len(galapagosid))-999
   pygfit_blendfrac=zeros(len(galapagosid))-999
   pygfit_segfrac=zeros(len(galapagosid))-999
   pygfit_flags=zeros(len(galapagosid))-999
   pygfit_chisq=zeros(len(galapagosid))-999
   pygfit_magtotal=zeros(len(galapagosid))-999

   pygcatid2=zeros(len(pygcatid),dtype=int16)
   for i in range(len(pygcatid)): pygcatid2[i]=int(float(pygcatid[i]))

   #2. Figure out the indices corresponding to the objects in the pygfit catalog.
   #   Basically want an array of all the indices that are good
   #3. Substitute in good values where they exist
   for i in range(len(galapagosid)): 
      tmpindex=where(galapagosid[i]==pygcatid2)[0]
      if(len(tmpindex)==1): 
            j=tmpindex[0]
            pygfit_mag[i]=pygcat['mag'][j]
            pygfit_magtotal[i]=pygcat['total_mag'][j]
            pygfit_blends[i]=pygcat['nblend'][j]
            pygfit_blendfrac[i]=pygcat['blend_fraction'][j]
            pygfit_segfrac[i]=pygcat['segmentation_fraction'][j]
            pygfit_flags[i]=pygcat['flags'][j]
            pygfit_chisq[i]=pygcat['chisq_nu'][j]
      
     

   #Add columns
   cols=[]
   for col in catalog[1].columns: cols.append(col)
   cols.append( pyfits.Column( name='PYGMAG_%s'%(filter), format='A6', array=pygfit_mag ) )
   cols.append( pyfits.Column( name='NBLEND_%s'%(filter), format='A6', array=pygfit_blends ) )
   cols.append( pyfits.Column( name='BLENDFRAC_%s'%(filter), format='A6', array=pygfit_blendfrac ) )
   cols.append( pyfits.Column( name='SEGFRAC_%s'%(filter), format='A6', array=pygfit_segfrac ) )
   cols.append( pyfits.Column( name='TOTMAG_%s'%(filter), format='A6', array=pygfit_magtotal) )
   cols.append( pyfits.Column( name='FLAGS_%s'%(filter), format='A6', array=pygfit_flags) )
   cols.append( pyfits.Column( name='CHISQ_NU_%s'%(filter), format='A6', array=pygfit_chisq) )

   outtabhdu=pyfits.new_table(cols)
   outtabhdu.writeto(newcat)  

    
### MAIN PROGRAM ###

###MATCHING F606W, B,I,z photometry from galapagos and pygfit  ##        
 
if len(sys.argv) < 2:
    sys.stderr.write('Usage: ./merged_pygfit.py  fieldname  reference_catalog filter1,filter2,filter3\n ')
    sys.stderr.write('All pygfit catalogs should have names of the form filter_pygfit.cat\n')
    sys.exit(1)

fieldname=str(sys.argv[1])
galapagoscatalog= str(sys.argv[2])
filter=(string.split(str(sys.argv[3]),','))
print galapagoscatalog
cat="%s_MERGED_PYGFIT.cat"%(fieldname)

for i in range(len(filter)):

   tcat='%s.tmp%s.fits'%(fieldname,str(i+1))
   pycat="%s_pygfit.cat"%(filter[i])
   if(os.access(tcat,0)): os.remove(tcat)

   if(i==0): 
      pygfit_match(galapagoscatalog,pycat,tcat,filter[i])
   else:
      tcatold='%s.tmp%s.fits'%(fieldname,str(i))
      pygfit_match(tcatold,pycat,tcat,filter[i])

os.rename(tcat,cat)

for i in range(len(filter)):
   tcat='%s.tmp%s.fits'%(fieldname,str(i+1))
   if(os.access(tcat,0)): os.remove(tcat)
