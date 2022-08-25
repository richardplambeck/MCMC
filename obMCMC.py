# obMCMC.py 
# converting ob2.py to use emcee MCMC methods
# all dielectrics are assumed to be isotropic!
# electric field is presumed to be perp to plane of incidence
# 

import numpy
import math
import matplotlib.pyplot as pyplot
import h5py
import emcee
import corner
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

clight = 29.9792458 # speed of light, cm/nanosec

# -----------------------------------------------------------------------------------------------#
# modified version of ob2.readFitFile, to read parameters for each fit
# maintains compatibility with old fitFile format (except no mirroring), but returns parameters in different format
# reads fitFile, returns fitParams dictionary
# this is a newer version that handles FIXED parameters 
# correspondence between 3n model parameters and < 3n MCMC fit parameters through fitParams["paramID"]
# use only datafile names that contain one of the "selectBands" keywords
#
def readFitFileMCMC2( fitFile, selectBands=["VNA","3mm","2mm","1mm"] ) :
    print ("\n%s" % fitFile)

    fitParams = {}
    fitParams["title"] = [] 
    fitParams["datafileList"] = [] 
    fitParams["datatype"] = [] 
    fitParams["label"] = []
    fitParams["tinRange"] = [0.,1000.]
    fitParams["paramID"] = []

    tinList = []
    nrList = []
    tanDeltaList = []
    nfit = 0	               

  # --- parse lines in fitFile ---
    with open( fitFile, "r" ) as fin :
      for line in fin :
        if line.find("#") > -1:
          line = line[0:line.find("#")]      # strip off # and everything following it (e.g, trailing comments)
        if len(line) > 0 :                   # ignore blank lines, or those beginning with #

          if "title" in line :
            fitParams["title"] = line[line.find("=")+1:].strip("\n")

          elif "data" in line :
            dataFileName =  line[line.find("=")+1:].strip() 
            appended = False
            for band in selectBands :
              if band in dataFileName :
                fitParams["datafileList"].append( dataFileName )
                fitParams["datatype"].append( "het" )     
	           # currently all datasets must be heterodyne
                appended = True
            if not appended : 
                print ("...skipping datafile %s because band is not in selectBands list" % dataFileName )
           
          elif "totin" in line :
            fitParams["tinRange"] = numpy.fromstring( line[line.find("=")+1:], sep=',')

          elif "tin" in line :
            tinList.append( numpy.fromstring(line[line.find("=")+1:], sep=',') )

          elif "nr" in line :
            nrList.append( numpy.fromstring(line[line.find("=")+1:], sep=',') )

          elif "tanDelta" in line :
            tanDeltaList.append( numpy.fromstring(line[line.find("=")+1:], sep=',') )

  # --- exit if there are no valid data files ---
    if len(fitParams["datafileList"]) < 1 :
      sys.exit ("\n** FATAL ** no valid data files")
       
  # --- step through [tinList, nrList, tanDeltaList], fill out [label, paramID, paramMin, paramMax] ----
    if len(tinList) != len(nrList) or len(tinList) != len(tanDeltaList) :
      sys.exit ("\n** FATAL ** unequal number of thicknesses, nr, and tanDelta given")

    label = []               # 't1','n1,'tanD1',etc
    paramMin = []
    paramMax = []

    print( "----------------------------------------------------------")
    for i in range(0,len(tinList)) :			# go through layers one by one

      label.append("t%d" % (i+1) )
      if len(tinList[i]) > 1 :				# if both min and max are given, fit this parameter
        paramMin.append( tinList[i][0] )
        paramMax.append( tinList[i][1] )
        fitParams["paramID"].append( nfit )
        print ("%6s: [%.4f,%.4f]  nfit = %d" % (label[-1],paramMin[-1],paramMax[-1],nfit) )
        nfit += 1
      else :                                            # single value indicates mirrored or fixed parameter
        if (tinList[i][0] < 0) :                        
          j = round(-1.*tinList[i][0])
          print("%6s:  MIRROR t%d" % (label[-1],j) )
          fitParams["paramID"].append( fitParams["paramID"][3*j-3] )
        else :                                                         # fixed parameter
          fitParams["paramID"].append( -1. * tinList[i][0] )
          print ("%6s: [%.4f]  FIXED  nfit = %.4f"  % (label[-1],tinList[i][0],fitParams["paramID"][-1]) )
       
      label.append( "nr%d" % (i+1) )
      if len(nrList[i]) > 1 :              
        paramMin.append( nrList[i][0] )
        paramMax.append( nrList[i][1] )
        fitParams["paramID"].append( nfit )
        print ("%6s: [%.4f,%.4f]  nfit = %d" % (label[-1],paramMin[-1],paramMax[-1],nfit) )
        nfit += 1
      else :
        if nrList[i][0] < 0. :
          j = round(-1.*nrList[i][0])
          print("%6s:  MIRROR nr%d" % (label[-1],j) )
          fitParams["paramID"].append( fitParams["paramID"][3*j-2] )
        else :
          fitParams["paramID"].append( -1. * nrList[i][0] )
          print ("%6s: [%.4f]  FIXED  nfit = %.4f" % (label[-1],nrList[i][0],fitParams["paramID"][-1]) )

      label.append( "tanD%d" % (i+1) )
      if len(tanDeltaList[i]) > 1 :             
        paramMin.append( tanDeltaList[i][0] )
        paramMax.append( tanDeltaList[i][1] )
        fitParams["paramID"].append( nfit )
        print ("%6s: [%.4f,%.4f]  nfit = %d" % (label[-1],paramMin[-1],paramMax[-1],nfit) )
        nfit += 1
      else :
        if tanDeltaList[i][0] < 0. :
          j = round(-1.*tanDeltaList[i][0])
          print("%6s:  MIRROR tanD%d" % (label[-1],j) )
          fitParams["paramID"].append( fitParams["paramID"][3*j-1] )
        else :
          fitParams["paramID"].append( -1. * tanDeltaList[i][0] )
          print ("%6s: [%.4f]  FIXED  nfit = %.4f" % (label[-1],tanDeltaList[i][0],fitParams["paramID"][-1]) ) 

    print( "----------------------------------------------------------")
    print (label)
    fitParams["label"] = label
    fitParams["paramMin"] = paramMin
    fitParams["paramMax"] = paramMax
    return fitParams

# -----------------------------------------------------------------------------------------------#
# fitParams["paramID"] makes the correspondence between modelParams used by solveStack and
#     "params" that MCMC is optimizing
# for n layers, there are always 3n modelParams, but there can be 0-3n params to optimize
# if x = fitParams["paramID"][i] is negative, then modelParam[i] is FIXED at abs(x),
#     else modelParam[i] = params[int(x)]
# for 2-sided coatings, modelParams for 1st and 2nd sides can be set equal
# -----------------------------------------------------------------------------------------------#
def fillin( params, fitParams ) :
    modelParams = []
    for p in fitParams["paramID"] :
      if p < 0. :
        modelParams.append( -1. * p )             # fixed parameter
      else :
        modelParams.append( params[round(p)] )     # parameter that is optimized by MCMC
    return modelParams

# -----------------------------------------------------------------------------------------------#
# getfitLabel returns labels only of the parameters that were fit (not fixed parameters)
def getfitLabel( label, paramID ):
    fitLabel = []
    for i in range(0,len(label)) :
      if paramID[i] >= 0. :
        fitLabel.append( label[i] )
    return fitLabel
     
# -----------------------------------------------------------------------------------------------#
# read heterodyne data
# 2022-feb-02; change so that angIdeg MUST be given in datafile
#
def readHetData( infile, addphs=0. ):
    angIdeg = -1000.   
    with open(infile,"r") as fin :
      for line in fin :
        if "angIdeg" in line :
          angIdeg = float( line[line.find("=")+1:] )
    if angIdeg == -1000. :
      sys.exit( "FATAL ERROR: angIdeg is not given in data file %s" % infile )
    else :
      print ("%s:  angIdeg = %.2f" % (infile, angIdeg))
    fGHzArr, amp, phs, unc = numpy.loadtxt(infile, usecols=(0,1,2,3), unpack=True )

  # sign choice in model is that phs DECREASES with freq; reverse sign of experimental data if necessary
    dphs = numpy.diff(phs)
    if len(dphs[dphs > 0.]) > len(dphs[dphs < 0.]) :
      print ("reversing phase of %s data to match new sign convention" % infile)
      phs = -1. * phs
    vec = amp * numpy.exp(1j * numpy.radians(phs+addphs))         # THIS WORKS (but phs is reversed, so it's really exp(-1j...)
    #vec = amp * numpy.exp(-1j * numpy.radians(phs))
    return fGHzArr, vec, unc, angIdeg

# theta0 and theta in radians, t in inches
def extraAirPath( theta0, theta, t ):
    A = t/numpy.cos(theta0)
    L = t/numpy.cos(theta)
    delta = A - L*numpy.cos(theta0-theta)
    return delta
    
# solveStack4 has correct sign conventions and was copied from HetFits.ipynb
def solveStack4( fGHzArr, angIdeg, params) :
    theta1 = numpy.radians(angIdeg)
    Y0 = numpy.cos(theta1)
    k0array = 2. * math.pi *fGHzArr / clight
    tcmtotal = 0.

  # this is a tricky way of making an array of identity matrices  
    M = numpy.zeros(4*len(k0array), dtype=complex)
    M[0:None:4] = 1
    M[3:None:4] = 1
    M2 = numpy.reshape(M, (-1,2,2))

  # step through the layers; param[i]=thickness in inches, param[i+1]=nrefrac; param[i+2]=tanDelta
    for i in range(0,len(params),3) :
      theta2 = numpy.arcsin( numpy.sin(theta1)/params[i+1] )         
          # Snell's law, ignoring imaginary part of n_r
#--------------
    # this applies additional correction for alleged extra air path (doesn't work)
    #  tcmtotal += 2.54 * (params[i] - extraAirPath( theta1, theta2, params[i] ))   # used for h5.1, h5.4
    # this is the "nominal" correction
      tcmtotal += 2.54 * params[i] 
          # keep track of total thickness of stack
#-------------
      ncomplex = params[i+1] * (1. - 1j * params[i+2]/2.)
      h = ncomplex * 2.54 * params[i] * numpy.cos(theta2)
      Y1 = ncomplex * numpy.cos(theta2)
      M1 = [[numpy.cos(k0array*h), 1j*numpy.sin(k0array*h)/Y1],[1j*Y1*numpy.sin(k0array*h),numpy.cos(k0array*h)]]
      M2 = numpy.matmul( M2, numpy.moveaxis(M1,2,0) )
               # M1 has shape (2,2,nfreq); moveaxis moves axis 2 to axis 0 to give desired shape, (nfreq,2,2)      

  # multiply terms in array of transfer matrices by appropriate factors of Y0
    M2[:,0,0] = Y0 * M2[:,0,0]
    M2[:,0,1] = Y0 * Y0 * M2[:,0,1]
    M2[:,1,1] = Y0 * M2[:,1,1]
    denom = numpy.sum( numpy.sum(M2,axis=2), axis=1 )
               # array of nfreq denominators, each the sum of the elements of the 2x2 matrix
    #trans = (2.*Y0/denom) / numpy.exp(-1j * k0array * tcmtotal)   # rO.h5.1, rP.h5.1 - close to correct answer  
       # h5.3 - use with no air correction; h5.4 - use this with air correction (works best)
    trans = (2.*Y0/denom) / numpy.exp(-1j * k0array * tcmtotal * math.cos(theta1))    # this is the nominal correction, used for h5.1, h5.2
    #trans = (2.*Y0/denom) / numpy.exp(-1j * k0array * tcmtotal/math.cos(theta1))
               # final divisor corrects for phase change through equivalent path in air
    return trans
# -----------------------------------------------------------------------------------------------#
def log_likelihood(param, fitParams, fGHzArr, vec, unc) :
    modelParams = fillin( param, fitParams)
	# inserts fixed parameter values into modelParams vector
    trans = solveStack4( fGHzArr, fitParams["angIdeg"], modelParams )
    sigma2 = numpy.power(unc,2) 
    return -0.5 * numpy.sum(numpy.power(numpy.abs(trans - vec),2)/sigma2)

# -----------------------------------------------------------------------------------------------#
def log_prior(param, fitParams):
    log_prior = 0.

  # process flat priors first; return -numpy.inf if any are outside allowed range
    for i in range(0,len(param)) :
      if param[i] < fitParams["paramMin"][i] or param[i] > fitParams["paramMax"][i] :
        return -numpy.inf 

  # also check total thickness, return -numpy.inf if it is outside allowed range
    totThickness = 0.
    modelParams = fillin( param, fitParams )
    for i in range(0,len(modelParams),3) :
      totThickness += modelParams[i]
    if totThickness < fitParams["tinRange"][0] or totThickness > fitParams["tinRange"][1] :  
      return -numpy.inf

  # process gaussian priors last, if there are any; NOTE: should take covariance into account
  #  for i in range(0,len(param)) :
  #    if fitParams["paramType"] == "gaussian" :
  #      log_prior -= 0.5*pow( (param[i]-fitParams["mu"][i])/fitParams["sigma"],2 )

    return log_prior

# -----------------------------------------------------------------------------------------------#
def log_probability(param, fitParams, fGHzArr, vec, unc) :
    lp = log_prior( param, fitParams)
    #print (param, lp, log_likelihood(param, fGHzArr, vec, unc))
    if not numpy.isfinite(lp) :
      return -numpy.inf
    return lp + log_likelihood(param, fitParams, fGHzArr, vec, unc)
  
# -----------------------------------------------------------------------------------------------#
def MCMCfit( fitFile, max_iter=5000, selectBands=["VNA","3mm","1mm"], addphs=0., uncMult=1. ):
    print ("entering MCMCfit")
    fitParams = readFitFileMCMC2( fitFile, selectBands=selectBands )                # fitParams is a dictionary 
    fGHzArr = []
    vec = []
    unc = []
    angIlist = []
    for datafile in fitParams["datafileList"] :    
      fArr, vc, un, angI = readHetData(datafile, addphs=addphs)		 
      fGHzArr = numpy.append(fGHzArr, fArr)
      vec = numpy.append(vec, vc)
      unc = numpy.append(unc, un)
      angIlist.append( angI )
    if len(angIlist) > 1 :
      if numpy.ndarray.any( numpy.array(angIlist)-angIlist[0] != 0. ) :
        sys.exit("\n** FATAL: angIdeg differs among datasets **\n")
    fitParams["angIdeg"] = angIlist[0]  
         
    print (fGHzArr)
    unc = uncMult*unc

  # save metadata as attributes, data as datasets, in an h5 file
    savefile = "%s.h5" % fitFile[0:fitFile.find(".")]
    nseq = 1
    while os.path.isfile( "%s.%d" % (savefile,nseq)) :
      nseq = nseq + 1
    savefile = savefile+".%d" % nseq 
    print ("creating new h5py file %s" % (savefile))

    with h5py.File( savefile, "w" ) as h5file :
      for k in fitParams.keys() :
        h5file.attrs[k] = fitParams[k]
      h5file.create_dataset("fGHzArr", data=fGHzArr)
      h5file.create_dataset("vec", data=vec)
      h5file.create_dataset("unc", data=unc)

  # emcee backend will append sample chain to this same file
    h5backend = emcee.backends.HDFBackend(savefile)

  # initial guess for parameters is midpoint of allowed range
    paramStart = numpy.average( [fitParams["paramMin"],fitParams["paramMax"]], axis=0 )
    ndim = len(paramStart)
    nwalkers = 32
    param = paramStart * (1. + 1e-4 * numpy.random.randn(nwalkers,ndim)) 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(fitParams,fGHzArr,vec,unc), backend=h5backend)

  # this code follows the example in https://emcee.readthedocs.io/en/stable/tutorials/monitor
    old_tau = numpy.inf
    for sample in sampler.sample(param, iterations=max_iter, progress=True) :
      if sampler.iteration % 100 :
        continue
      tau = sampler.get_autocorr_time(tol=0)
        # tau is an array with dimension = number of fit parameters; it can be all 'nan'
      # print ("tau = ",tau)

      converged = numpy.all(sampler.iteration > 10.*tau)           # tests whether all array elements evaluate to True
      converged &= numpy.all(numpy.abs(old_tau - tau) / tau < 0.05)
      # print ("fractional change : ",numpy.abs(tau - old_tau) / tau)
      if converged:
        break
      old_tau = tau

    return savefile

# -----------------------------------------------------------------------------------------------#
# print and return best fit values and error bars from sample chain stored in h5file
# if csvFile (colon-separated-file) is given, append fit values to that file also
# -----------------------------------------------------------------------------------------------#
def bestFit( h5file, csvFile=None ):

    with h5py.File(h5file,"r") as fin :                  # first read ancillary info from h5 file
      title = fin.attrs["title"]
      datafileList = fin.attrs["datafileList"]
      label = fin.attrs["label"]                        
      angIdeg = fin.attrs["angIdeg"]
      try :
        paramID = fin.attrs["paramID"]
      except :
        paramID = numpy.arange(0,len(label))		# only for historic h5 files
      fGHz = fin["fGHzArr"][0:]                          # convert h5 dataset to numpy array
      trans_meas = fin["vec"][0:]
      unc = fin["unc"][0:]

    reader = emcee.backends.HDFBackend( h5file, read_only=True )
    tau = reader.get_autocorr_time(tol=0)                # tau is an array, one value per param
    burnin = int(2 * numpy.max(tau))
    thin = int(0.5 * numpy.min(tau))
    flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    #ndim = flat_samples.shape[1]
    np = len(paramID)
    param = numpy.zeros( np )
    errminus = numpy.zeros( np )
    errplus = numpy.zeros( np)

  # buf1 is beginning of csvFile line, also will be printed
    buf1 = "%s : %s : %s :" % (h5file,title,datafileList)
    print( buf1 )

  # buf2 holds best fit parameters and uncertainties
    buf2 = " "

  # paramID connects fitted and model parameters (some may be fixed)
    for n in range(0,np) :
      if paramID[n] < 0. :            # FIXED parameter
        param[n] = -1.*paramID[n]
        errminus[n] = 0.
        errplus[n] = 0.
        print ("%8s  %.5f  FIXED" % (label[n],param[n]))
      else :					   # FIT parameter
        i = round(paramID[n])      
        mcmc = numpy.percentile(flat_samples[:, i], [16, 50, 84])
        param[n] = mcmc[1]       
        errminus[n] = mcmc[0]-mcmc[1]
        errplus[n] = mcmc[2]-mcmc[1]
        print ("%8s  %.5f (%.5f,+%.5f)" % (label[n],param[n],errminus[n],errplus[n]))
      buf2 += " %.5f : %.5f : %.5f :" % (param[n],errminus[n],errplus[n])
    trans_model = solveStack4( fGHz, angIdeg, param ) 
    sigma2 = numpy.power(unc,2) 
    chisq = numpy.sum(numpy.power(numpy.abs(trans_model - trans_meas),2)/sigma2)/len(trans_meas)
    print ("reduced chisq: %.3f" % chisq)
    buf1 += " %5.3f :" % chisq

    if (csvFile) :
      with open(csvFile,"a") as fout :
        fout.write(buf1+buf2+"\n")
    return label,chisq,param,errminus,errplus

# -----------------------------------------------------------------------------------------------#
def plotSamplerChain( h5file ) :

  # retrieve necessary metadata from h5 file
    with h5py.File(h5file,"r") as f:
      label = f.attrs["label"]
      title = f.attrs["title"]
      try :
        paramID = f.attrs["paramID"]
        fitLabel = getfitLabel( label, paramID )
      except :
        fitLabel = label   
    reader = emcee.backends.HDFBackend(h5file)
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2 * numpy.max(tau))
    thin = int(0.5 * numpy.min(tau))

  # plot parameters vs time
    samples = reader.get_chain(discard=0, flat=False)
    ndim = samples.shape[2]
    fig, axes = pyplot.subplots(ndim, figsize=(10,7), sharex=True)
    for i in range(ndim):
      ax = axes[i]
      ax.plot(samples[:, :, i], "k", alpha=0.3)
      ax.axvline(x=burnin, color='g', linestyle='--')
      ax.set_ylabel( fitLabel[i] )
      ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    pyplot.savefig("sample_chain.pdf")
    pyplot.show()

# -----------------------------------------------------------------------------------------------#
def makeCornerPlot( h5file ) :
    with h5py.File(h5file,"r") as f:
      label = f.attrs["label"]
      title = f.attrs["title"]
      try :
        paramID = f.attrs["paramID"]
        fitLabel = getfitLabel( label, paramID )              # return labels only of parameters that were fit
      except :
        fitLabel = label
    reader = emcee.backends.HDFBackend(h5file)
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2 * numpy.max(tau))
    thin = int(0.5 * numpy.min(tau))
    flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
    figc = pyplot.figure( figsize=(7,7))
	# defining fig before call to corner to set dimensions; otherwise plot is bigger than my screen
    fig = corner.corner(flat_samples, labels=fitLabel, quantiles=[0.16,0.5,0.84], labelpad=0.3, title_kwargs={'fontsize':6}, \
      show_titles=True, title_fmt=".3f", fig=figc )
    pyplot.suptitle(title, x=.95, ha='right',fontsize=14)
    #fig.subplots_adjust(right=1.5, top=1.5)
    pyplot.savefig("corner.pdf",pad_inches=0.3, bbox_inches='tight')
    pyplot.show()

# -----------------------------------------------------------------------------------------------#
def unwrap( phi ) :
    for n in range(0,len(phi)) :
      while phi[n] < -180. :
        phi[n] = phi[n] + 360.
      while phi[n] > 180. :
        phi[n] = phi[n] - 360.
    return phi

# -----------------------------------------------------------------------------------------------#
def gap( fGHz ):
  gaps = numpy.diff( fGHz )
  if gaps.max() > 0.5 :
    return True
  else :
    return False

# -----------------------------------------------------------------------------------------------#
def minmax( varray, margin=.05 ) :
    vmin = varray.min()
    vmax = varray.max()
    diff = vmax - vmin
    return [ vmin - margin*diff, vmax + margin*diff ] 

# -----------------------------------------------------------------------------------------------#
# overplot fit and data (for heterodyne data)
def plotFit( h5file, errorbars=False, saveplot="fit.pdf" ) :
    FTS = False
    with h5py.File(h5file,"r") as fin :
      title = fin.attrs["title"]
      angIdeg = fin.attrs["angIdeg"]
      datafileList = fin.attrs["datafileList"]
      tinRange = fin.attrs["tinRange"]                        
      fGHz = fin["fGHzArr"][0:]             # convert h5 dataset to numpy array
      trans_meas = fin["vec"][0:]
      unc = fin["unc"][0:]
      angIdeg = fin.attrs["angIdeg"]
      label = fin.attrs["label"]
      try :
        paramID = fin.attrs["paramID"]
      except :
        paramID = list(range(0,len(label)))
    label,chisq,paramBest,errminus,errplus = bestFit( h5file )

    fig =  pyplot.figure( figsize=(11,8) )
    pyplot.suptitle( "%s   %s\n%s" % (title,h5file,datafileList), fontsize=12 )

  # fill in freq gaps with the model
    fGHzmodel = fGHz
    if gap(fGHz) :
      fGHzmodel = numpy.arange( fGHz.min(), fGHz.max()+.05, .05 )
   
  # recalculate the model amps and phases for the best fit parameters
    trans_model = solveStack4( fGHz, angIdeg, paramBest ) 
    trans_model_continuous = solveStack4( fGHzmodel, angIdeg, paramBest ) 
    #sigma2 = numpy.power(unc,2) 
    #chisq = numpy.sum(numpy.power(numpy.abs(trans_model - trans_meas),2)/sigma2)/len(trans_meas)

  # plot phase unless FTS
    if not FTS :
      phs_meas = numpy.angle(trans_meas, deg=True)
      phs_model = numpy.angle(trans_model, deg=True)
      phs_model_continuous = numpy.angle( trans_model_continuous, deg=True )

    # top left panel is phase vs freq, measured and fit
      ax = fig.add_axes( [.1,.6,.38,.32] )
      ax.plot( fGHzmodel, phs_model_continuous, "r-", linewidth=1 )
      ax.plot( fGHz, phs_meas, "bo", ms=2. )
      ax.set_ylim( [-195.,195.] )
      ax.tick_params( labelsize=10 )
      ax.set_ylabel("phase (deg)", fontsize=10)
      ax.grid(True)
  
    # middle left panel shows phase residual vs freq; avg shown as horiz dashed line
      ax = fig.add_axes( [.1,.35,.38,.2]) 
      dphi = unwrap( phs_meas - phs_model )
      ax.plot( fGHz, dphi, "bo-", ms=1. )
      ax.tick_params( labelsize=10 )
      dphiavg = numpy.average(dphi)
      ax.plot( [fGHz[0],fGHz[-1]],[dphiavg,dphiavg],"r--" )
      ax.set_ylabel("residual (deg)", fontsize=10)
      ax.set_xlabel("freq (GHz)", fontsize=10)
      ax.set_ylim( [-4.,4.] )
      ax.grid(True)

  # top right panel is transmission vs freq, measured and theoretical
    amp_meas = numpy.abs(trans_meas)
    amp_model = numpy.abs(trans_model)
    amp_model_continuous = numpy.abs(trans_model_continuous)
    if FTS :
      amp_model = trans_model * numpy.conj(trans_model)
      ax = fig.add_axes( [.1,.6,.85,.32] )
      ax.set_ylabel("transmitted pwr", fontsize=10)
    else :
      ax = fig.add_axes( [.57,.6,.38,.32] )
      ax.set_ylabel("amplitude", fontsize=10)
      ax.plot( fGHzmodel, amp_model_continuous, "r-", linewidth=1 )
      if errorbars :
        ax.errorbar( fGHz, amp_meas, yerr=unc, marker='o', ms=2., elinewidth=1, ls='none', capsize=0 )
      else :
        ax.plot( fGHz, amp_meas, 'bo', ms=2., ls='none' )
      ax.tick_params( labelsize=10 )
      ax.grid(True)

    # middle right panel is measured-theoretical trans
      if FTS :
        ax = fig.add_axes( [.1,.35,.85,.2] )
      else :
        ax = fig.add_axes( [.57,.35,.38,.2])
      ax.tick_params( labelsize=10 )
      damp = amp_meas - amp_model
      ax.plot( fGHz, damp, "bo-", ms=1. )
      ax.set_xlabel("freq (GHz)", fontsize=10)
      ax.set_ylabel("residual", fontsize=10)
      ax.set_ylim( [-.06,.06] )
      ax.grid(True)

    # (fictitious) bottom panel lists search ranges, limits, best values
      ystep = .10
      ax = fig.add_axes( [.1,.065,.85,.22])
      ylab = 1 - ystep
      ax.text( 0.1, ylab, "angle = %.2f deg;  allowed thickness = [%.4f,%.4f]" % \
        ( angIdeg, tinRange[0], tinRange[1] ), transform=ax.transAxes, \
        horizontalalignment='left', color="black", rotation='horizontal' )
      ax.text( 0.9, ylab, "reduced chisq = %.3f " % chisq, transform=ax.transAxes, \
        horizontalalignment='right', color="black", rotation='horizontal' )

      ylab = ylab - ystep - .08
      ax.text( 0.03, ylab, "layer", style='oblique', horizontalalignment='center')
      ax.text( 0.18, ylab, "thickness", style='oblique', horizontalalignment='center')
      ax.text( 0.40, ylab, "index", style='oblique', horizontalalignment='center')
      ax.text( 0.63, ylab, "epsilon", style='oblique', horizontalalignment='center')
      ax.text( 0.86, ylab, "tanDelta", style='oblique', horizontalalignment='center')
      ylab = ylab - 0.02
      
      totThickness = 0.
      for i in range(0,len(paramBest),3) : 

        ylab = ylab - ystep 
        ax.text( 0.03, ylab, "%d" % ((i/3. +1)), horizontalalignment='center')
        if paramID[i] < 0. :
          ax.text( .08, ylab, "%.4f  FIXED" % paramBest[i], horizontalalignment='left' )
        else :
          ax.text( .08, ylab, "%.4f (%.4f,+%.4f)" % (paramBest[i],errminus[i],errplus[i]), horizontalalignment='left' )
        totThickness += paramBest[i]

        epsBest = pow(paramBest[i+1],2)
        epsErrMinus = epsBest - pow(paramBest[i+1]-errminus[i+1],2)
        epsErrPlus = pow(paramBest[i+1]+errplus[i+1],2) - epsBest
        if paramID[i+1] < 0. :
          ax.text( .32, ylab, "%.3f  FIXED" % paramBest[i+1], horizontalalignment='left' )
          ax.text( .55, ylab, "%.3f  FIXED" % epsBest, horizontalalignment='left' )
        else :
          ax.text( .32, ylab, "%.3f (%.3f,+%.3f)" % (paramBest[i+1],errminus[i+1],errplus[i+1]), \
            horizontalalignment='left' )
          ax.text( .55,  ylab, "%.3f (%.3f,+%.3f)" % (epsBest,epsErrMinus,epsErrPlus), horizontalalignment='left' )

        if paramID[i+2] < 0. :
          ax.text( .77, ylab, "%.4f  FIXED" % paramBest[i+2], horizontalalignment='left' )
        else :
          ax.text( .77, ylab, "%.4f (%.4f,+%.4f)" % (paramBest[i+2],errminus[i+2],errplus[i+2]), horizontalalignment='left' )

    # put total modeled thickness on last line if more than 1 layer
      if len(paramBest) > 3 :
        ylab = ylab - ystep 
        ax.text( 0.08, ylab, "%.4f" % totThickness, horizontalalignment='left' ) 
      pyplot.axis('off')
      pyplot.savefig( saveplot )
      pyplot.show()

# ---
# compute power transmission from 20-280 GHz, calculate avg transmission in 3mm and 2mm bands
# use modelParams if given; otherwise try to read parameters from h5file
# set all loss tangents to 0

def calcPwrTrans( h5file, modelParams=None, saveplot="pwr_trans.pdf" ):
    if modelParams :
      paramWarm = modelParams
    else :
      label,chisq,paramWarm,errminus,errplus = bestFit(h5file)
      with h5py.File(h5file,"r") as fin :
        title = fin.attrs["title"]
        fGHz_meas = fin["fGHzArr"][0:]             # convert h5 dataset to numpy array
        trans_meas = fin["vec"][0:]
        unc = fin["unc"][0:]
        angIdeg = fin.attrs["angIdeg"]
        label = fin.attrs["label"]
        pwr_meas = numpy.abs( trans_meas * numpy.conj(trans_meas) )

  # zero the loss tangents for cold transmission calculation
    paramCold = paramWarm.copy()
    for i in range(2,len(paramCold),3) :
      paramCold[i] = 0.

    fGHzArr = numpy.arange(50.,190.,.05) 
    transWarm = solveStack4( fGHzArr, 0.0, paramWarm )
    transCold = solveStack4( fGHzArr, 0.0, paramCold )
    pwrWarm = numpy.abs(transWarm * numpy.conjugate(transWarm))
    pwrCold = numpy.abs(transCold * numpy.conjugate(transCold))

    band1 = [74.,104.]
    band2 = [126.,166.]
    iband1 = numpy.rint( (band1-fGHzArr[0]) / (fGHzArr[1]-fGHzArr[0])).astype(int)
    iband2 = numpy.rint( (band2-fGHzArr[0]) / (fGHzArr[1]-fGHzArr[0])).astype(int)
       # fGHzArr[iband[0]] is lower edge of band, fGHzArr[ iband[1]] is upper edge of band
    band1_avg = numpy.average( pwrCold[iband1[0]:iband1[1]] )
    band2_avg = numpy.average( pwrCold[iband2[0]:iband2[1]] )

    fig,ax = pyplot.subplots(1,1,figsize=(10,7))
    ax.plot( fGHz_meas, pwr_meas, 'or', label="meas data", markersize=3 ) 
    ax.plot( fGHzArr, pwrWarm, color='red', linewidth=0.5, label="warm fit" )
    ax.plot( fGHzArr, pwrCold, color='blue', label="cold fit" )
    ax.vlines( band1, 0., 1., color='black', linestyle='dashed' )
    ax.text( numpy.average(band1), .72, "<band1>=%.3f" % band1_avg, \
      ha='center', color='blue', fontsize=10)
    ax.vlines( band2, 0., 1., color='black', linestyle='dashed' )
    ax.text( numpy.average(band2), .72, "<band2>=%.3f" % band2_avg, \
      ha='center', color='blue', fontsize=10)
    ax.set_ylim(0.6,1.)
    ax.set_ylabel("power transmission")
    ax.set_xlabel("freq (GHz)")
    ax.set_xlim( band1[0]-10.,band2[1]+10. )
    ax.legend()
    pyplot.suptitle("power transmission from %s" % h5file, y=.93)
    ax.grid(True)
    pyplot.savefig( saveplot )
    pyplot.show()

# ---
# compare expected transmission results for different models
def compareModels( h5list ) :
    fig,ax = pyplot.subplots(2,1)
    fGHzArr = numpy.arange(0.,300.,.1) 
    for h5file in h5list :
      label,chisq,param,errminus,errplus = bestFit(h5file)
      trans= solveStack4( fGHzArr, 0.0, param )
      ax[0].plot( fGHzArr, numpy.abs(trans), label="%s" % h5file )
      ax[1].plot( fGHzArr, numpy.angle(trans, deg=True) )
    pyplot.show()
    
# predict FTS power transmission spectrum for a sample from a heterodyne fit
def predictFTS( h5list, suptitle=None ) :
    fig,ax = pyplot.subplots(1,1)
    fGHzArr = numpy.arange(0.,300.,.1) 
    for h5file in h5list :
      label,chisq,param,errminus,errplus = bestFit(h5file)
      trans= solveStack4( fGHzArr, 0.0, param )
      pwr = numpy.power( numpy.abs(trans), 2 )
      ax.plot( fGHzArr, pwr, label=h5file )
      with open("FTSprediction.txt", "w") as fout :
        fout.write("# %s\n" % suptitle)
        fout.write("# parameters from %s\n" % h5file)
        for f,p in zip(fGHzArr,pwr) :
          fout.write("%7.2f  %6.4f\n" % (f,p) )
    ax.grid( True )
    ax.legend( fontsize=8 )
    ax.set_ylabel( "predicted power transmission" )
    ax.set_xlabel( "freq (GHz" )
    if suptitle :
      pyplot.suptitle( suptitle, y=.94)
    pyplot.tight_layout()
    pyplot.savefig("FTSprediction.pdf")
    pyplot.show()

# overplot fixed model on observational data
def plotData( fitFile, params, selectBands=["2mm"] ) :
    fitParams = readFitFileMCMC2( fitFile, selectBands=selectBands )                # fitParams is a dictionary 
    fGHzArr = []
    vec = []
    unc = []
    angIlist = []
    for datafile in fitParams["datafileList"] :    
      fArr, vc, un, angI = readHetData(datafile)		 
      fGHzArr = numpy.append(fGHzArr, fArr)
      vec = numpy.append(vec, vc)
      unc = numpy.append(unc, un)
      angIlist.append( angI )
    if len(angIlist) > 1 :
      if numpy.ndarray.any( numpy.array(angIlist)-angIlist[0] != 0. ) :
        sys.exit("\n** FATAL: angIdeg differs among datasets **\n")
    print (fGHzArr)
    trans= solveStack4( fGHzArr, angIlist[0], params )
    fig,ax = pyplot.subplots(1)
    amp_meas = numpy.abs(trans_meas)
    amp_model = numpy.abs(trans_model)
    ax.plot(fGHzArr,ampmeas,'o')
    ax.plot(fGHzArr,ampmodel,'-')
    pyplot.show()
