# lensMeas.py
# this code models optics bench measurements of the field lens, compares
#   transmission with that of alumina flat with same thickness

import math
import numpy
import matplotlib.pyplot as pyplot
import obMCMC


# step outwards from center of lens in thickness decrements of tstep
# compute radius at which this thickness decrement occurs, and fraction of gaussian
#    beam that lies between this radius and previous one
# return table of thicknesses and intercepted amplitudes
# R1,R2 = radii of curvature of the front and back lens surfaces
# w = beamwaist
# rmax = max allowed radius
# deltat = thickness decrement
#
def LensAnnuli(R1,R2,w,rmax,tstep) :
    tdec = []    # list of average thickness decrements
    frac = []    # list of fractional amplitudes
    wsq = w*w
    tdec1 = 0.
    r1 = 0.
    r2 = 0.
    while (r2 < rmax) :
      tdec2 = tdec1 + tstep
      r2 = math.sqrt(tdec2/(.5/R1 + .5/R2) )
      if r2 > rmax :
        r2 = rmax
        tdec2 = rmax*rmax * (.5/R1 + .5/R2)
      tdec.append( (tdec1+tdec2)/2. )
      frac.append( math.pi * wsq*  (math.exp(-r1*r1/wsq) - math.exp(-r2*r2/wsq) ) )
      r1 = r2
      tdec1 = tdec2
    return tdec,frac

R1 = 59.764     # lens radius 9.312, extra thickness at center = .73
R2 = 86.873     # extra thickness 0.5
w = 0.5	        # guess beamwaist at 3mm
rmax = 0.5	# block off radiation outside rmax

tdec,frac = LensAnnuli(R1,R2,w,rmax,.001)
tmax = .882 + .73 + .5
angIdeg = 5.16

fGHzArr = numpy.arange(90.,95.,.01)
transtot = numpy.zeros(len(fGHzArr),dtype=complex)    # weighted vector sum of transmission
fractot = 0.

fig,ax = pyplot.subplots(2,1)

# compute and plot  transmission of lens
for i in range(0,len(tdec) ):
  print ("%8.5f  %.8f" % (tdec[i],frac[i]))
  # params = [ tmax-tdec[i], 3.12, .0003, .028, 1.7, 0. ]
  params = [ tmax-tdec[i], 3.12, .0003 ]
  trans = obMCMC.solveStack4( fGHzArr, angIdeg, params )
  transtot += frac[i] * trans
  fractot += frac[i]
  #ax[0].plot( fGHzArr, numpy.abs(trans), color='black', alpha=5.*frac[i] )
  #ax[1].plot( fGHzArr, numpy.angle(trans,deg=True), color='black', alpha=5.*frac[i] )
trans = transtot/fractot
ax[0].plot( fGHzArr, numpy.abs(trans), label="lens" )
ax[1].plot( fGHzArr, numpy.angle(trans,deg=True) )

# for comparison, plot transmission of a flat with same thickness as lens
# trans = obMCMC.solveStack4( fGHzArr, angIdeg, [tmax, 3.12, .0003, .028, 1.7, 0.] )
trans = obMCMC.solveStack4( fGHzArr, angIdeg, [tmax, 3.12, .0003] )
ax[0].plot( fGHzArr, numpy.abs(trans), label="flat" )
ax[1].plot( fGHzArr, numpy.angle(trans,deg=True) )

ax[0].set_ylabel("amplitude")
ax[1].set_ylabel("phase (deg)")
ax[1].set_xlabel("freq (GHz)" )
ax[0].legend()
fig.suptitle("uncoated field lens, beamwaist 0.5, aperture 1.0", y=.93)
pyplot.savefig("uncoated2.pdf")
pyplot.show()


