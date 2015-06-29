# Latest edit - 20150225
# Ryan Valenza, Alex Ditter

from numpy import *
from scipy import misc
import Image
import sys
import math
import matplotlib.pyplot as plt
import glob
import os
import h5py

# GetArray function:
# Gets a quad image as an array from the hdf5 file.
# Pattern of CsPad chips determined from testing.py and old images on 3/30/15.
# Same function as averager4.py
# Input:
#	run: run number
#	event: event number (starts at 1)
# Outputs:
#	numpy array shape 830 x 825

def getArray(run, event):
	f=h5py.File(hdf5folder+'mecd6714-r%04.i.h5' %run,'r')
	quaddata = f['/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/MecTargetChamber.0:Cspad.0/data']
	output = zeros((830,825))
	corners = [
		[429,421],
		[430,634],
		[420,1],
		[633,0],
		[0,213],
		[0,1],
		[16,424],
		[228,424]
		]
	rotated = [1,1,0,0,3,3,0,0]
	for i, arr in enumerate(quaddata[event-1]):
		a = rot90(insert(arr,(193,193,193,193),0,axis = 1),rotated[i])
		if rotated[i]:
			output[corners[i][0]:corners[i][0]+392, corners[i][1]:corners[i][1]+185] = a
		else:
			output[corners[i][0]:corners[i][0]+185, corners[i][1]:corners[i][1]+392] = a
	return output
	
# getImage function:
# Handles some of the options and returns an array from the image
# Input:
#	imloc: image location in whichever format is required.
#	       either filename.tif or [run, event] or filename.csv
# Output:
#	numpy array of the image and a label. Image is not transposed.
def getImage(imloc):
	if runevent:
		run = imloc[0]
		evt = imloc[1]
		label = "run %s event %s" %(run,evt)
		imarray = getArray(run, evt)
	elif arr:
		imarray = loadtxt(open(imloc),delimiter = ',')
		label = imloc[:-4]
	else:
		imarray = array(Image.open(imloc))
		label = imloc[:-4]
	if masked:
		imarray = imarray * array(Image.open(mask))
	return([label,imarray])

	
	

# translate(phi, x0, y0, alpha, r)
# Produces I vs theta values for imarray. For older versions, see bojangles_old.py
# Inputs:  detector configuration parameters and diffraction image
# Outputs:  lists of intensity and 2theta values (data)
def translate(phi, x0, y0, alpha, r, imarray):

	

	length, width = imarray.shape
	y = vstack(ones(width)*i for i in range(length))
	ecks = vstack([1 for i in range(length)])
	x = hstack(ecks*i for i in range(width))
	x2 = -cos(phi) *(x-x0) + sin(phi) * (y-y0)
	y2 = -sin(phi) * (x-x0) - cos(phi) * (y-y0)
	rho = (r**2 + x2**2 + y2**2)**0.5
	y1 = y2 * cos(alpha) + r * sin(alpha)
	z1 = - y2 * sin(alpha) + r * cos(alpha)
	# beta is the twotheta value for a given (x,y)
	beta = arctan2((y1**2 + x2**2)**0.5, z1) * 180 / pi
	if ellipses:
		for ang in anglist:
			imarray = where(logical_and(beta > ang - ew, beta < ang + ew), 0, imarray)
	
	imarray = imarray * square(rho)
	
	newpoints = vstack((beta.flatten(), imarray.flatten()))
	
	return newpoints.T, imarray

	
# processData()
# Inputs:  a single diffraction image array and filename
# Outputs:  data in bins, intensity vs. theta. Saves data to file
def processData(imarray, filename):

	imarray = imarray.T
	# Manually entered data after 2015/04/01 calibration. (really)
	# See Testing.nb for details.
	# Coordinates in pixels. 0.011cm per pixel.
	(phi, x0, y0, alpha, r) = (0.027763, 322.267, 524.473, 0.787745, 1082.1)
	# pre-Jan 2015 inputs.
	#(phi, x0, y0, alpha, r) = (0, -3.2, -4, 0.746, 12.464)
	data, imarray = translate(phi, x0, y0, alpha, r, imarray)
	
	
	
	thetas = data[:,0]
	intens = data[:,1]

	# algorithm for binning the data

		
	
	ma = max(thetas)
	mi = min(thetas)
	stepsize = (ma - mi)/(nbins)
	binangles = binData(mi, ma, stepsize)
	numPix = [0] * (nbins+1)
	intenValue = [0] * (nbins+1)
	
	if valenza: print "putting data in bins"		
	# find which bin each theta lies in and add it to count
	for j,theta in enumerate(thetas):
		if intens[j] != 0:
			k = int(floor((theta-mi)/stepsize))
			numPix[k]=numPix[k]+1
			intenValue[k]=intenValue[k]+intens[j]
	# form average by dividing total intensity by the number of pixels
	if valenza: print "adjusting intensity"
	adjInten = (array(intenValue)/array(numPix))
	
	f = open(filename, 'w')
	
	for k,inten in enumerate(adjInten[:-1]):
		f.write(str(inten) + " ," + str((binangles[k]+binangles[k+1])/2) + "\n")

	f.close()
	return binangles, adjInten, imarray, filename[:-4]

# binData()
# Input:  a minimum, a maximum, and a stepsize
# Output:  a list of bins

def binData(mi, ma, stepsize):
	
	if valenza: print "creating angle bins"
	binangles = list()
	binangles.append(mi)
	i = mi
	while i < ma-(stepsize/2):
		i += stepsize
		binangles.append(i)

	return binangles

	
# anglePlot()
# Inputs:  intensity list and a bin list
# Outputs:  a plot of intensity vs. theta
def anglePlot(processeddata):
	binangles = processeddata[0]
	adjInten = processeddata[1]
	imarray = processeddata[2]
	
	if valenza: print "generating angle plot"	

	fig = plt.figure("CSPAD QUAD Image", figsize = (10,10) )

	ax1 = plt.subplot2grid((4,3),(0,0), colspan = 3, rowspan = 3)
	ax1.set_title('Calibrated Camera Image')
	im = ax1.imshow(imarray, aspect='auto', interpolation = 'nearest')
	im.set_clim(-10,int(nanmax(adjInten)))
	fig.colorbar(im)
	
	ax2 = plt.subplot2grid((4,3),(3,0), colspan = 3)
	ax2.set_title('Angular Distribution')
	ax2.grid(True)
	ax2.plot(binangles,adjInten,'r')
	plt.xlabel('Angle [degrees]')
	plt.ylabel('ADU/pixel [counts]')

	plt.tight_layout()
	plt.show()

def normalizer(binangles, *intens):
	
	if normmin == None:
		if valenza: print "Plots will not be normalized"
		return intens
	sum1 = 0
	sum = 0
	
		
	for i in range(len(intens)):
		sum = 0
		if i == 0:
			for k in range(len(binangles)):
				if normmin <= binangles[k] and binangles[k] <= normmax:
					sum1 = sum1+ intens[0][k]
					
			if valenza: print sum1
		for k in range(len(binangles)):
			if normmin <= binangles[k] and binangles[k] <= normmax:
				sum = sum + intens[i][k]
		if valenza: print sum
		intens[i] = intens[i]*sum1/sum
	return intens
	
def noImPlot(data):
	binangles = data[0][0]
	intens = array(data)[:,1]
	intens = normalizer(binangles, intens)
	for i in range(len(intens[0])):
		plt.plot(binangles, intens[0][i], label = data[i][3])
	plt.legend()
	plt.xlabel('Angle (Degrees)', size = 'large')
	plt.ylabel('Intensity (counts)', size = 'large')
	plt.show()
	
def differencePlot(data):
	binangles = data[0][0]
	adjInten1 = data[0][1]
	adjInten2 = data[1][1]
	diff = adjInten1 - adjInten2

	fig = plt.figure("CSPAD QUAD Image", figsize = (10,10) )
	
	ax1 = plt.subplot2grid((3,3),(0,0), colspan = 3)
	ax1.set_title('Angular Distribution - Image 1')
	ax1.grid(True)
	ax1.plot(binangles,adjInten1,'r')
	plt.xlabel('Angle [degrees]')
	plt.ylabel('ADU/pixel [counts]')

	ax2 = plt.subplot2grid((3,3),(1,0), colspan = 3)
	ax2.set_title('Angular Distribution - Image 2')
	ax2.grid(True)
	ax2.plot(binangles,adjInten2,'r')
	plt.xlabel('Angle [degrees]')
	plt.ylabel('ADU/pixel [counts]')

	ax3 = plt.subplot2grid((3,3),(2,0), colspan = 3)
	ax3.set_title('Intensity Difference (1-2)')
	ax3.grid(True)
	ax3.plot(binangles,diff,'r')
	plt.xlabel('Angle [degrees]')
	plt.ylabel('ADU/pixel [counts]')
	
	plt.tight_layout()
	plt.show()

def main():


	# set up option variables in their default states
	global masked
	masked = False
	global every
	every = False
	global show
	show = True
	global ellipses
	ellipses = False
	global ew
	ew = 0.05
	global anglist
	anglist = [31.42, 37.03, 45.04, 55.95, 59.67, 65.59]
	global valenza
	valenza = False
	difference = False
	spaghetti = False
	all = False
	global runevent
	runevent = False
	global local
	local = True
	global arr
	arr = False
	global norm
	global normmin
	normmin = None
	global normmax
	norm = False
	args = sys.argv[1:]
	s = 20
	seterr('ignore')
	global mask
	mask = 'masks\\mask'
	global hdf5folder
	hdf5folder = 'E:\\hdf5\\'
	global nbins
	nbins = 1000


	#valenza = verbose
	if '-v' in args:
		valenza = True
		args.remove('-v')
		if valenza:
			print("Verbosity enabled")
	
	#nonlocal will use files on the slac server (no longer supported)
	if '-nl' in args:
		local = False
		args.remove('-loc')
		mask = '/reg/d/psdm/MEC/mecd6714/scratch/quad/masks/mask/'
		hdf5folder = '/reg/d/psdm/MEC/mecd6714/hdf5/'
		if valenza:
			print("Using SLAC files")
	
	#masked is an option which takes the image and removes edge
	#effects or selects a single chip.
	#averager3.py typically does this. (later versions as well)
	#usage: python bojangles.py 
	if '-m' in args:
		masked = True
		i = args.index('-m')
		if valenza:
			print("Masking enabled")
		try:
			if args[i+1] == 'all':
				every = True
				show = False
				masked = False
				args.remove('all')
				if valenza:
					print("    Each mask will be used to create a separate text file")
					print("    Plots disabled")
			elif int(args[i+1]) in range(17):
				s = int(args[i+1])
				args.remove(args[i+1])
				mask = mask + '_%s.tif' %s
				if valenza:
					print("    Mask %s" %s)
		except (ValueError, IndexError):
			mask = mask +'.tif'
			if valenza:
				print("    Default mask")
		args.remove('-m')
		
	#ellipses will plot a set of ellipses with the angles in anglist
	#usage: python bojangles.py image [image2 ...] -e [ang1 ang2 ang3 ...]
	#EVERY float immediately after -e will be assumed to be part of anglist
	#if no angles are given, anglist defaults to the list above
	if '-e' in args:
		ellipses = True
		if valenza:
			print("Ellipse plot enabled")
		i = args.index('-e')
		try:
			float(args[i+1])
			anglist = []
			while True:
				try:
					anglist.append(float(args[i+1]))
					args.pop(i+1)
				except(ValueError, IndexError):
					break
			if valenza:
				print("    Ellipse locations:")
				print("    %s" %anglist)
		except:
			if valenza:
				print("    Default ellipse locations:")
				print("    %s" %anglist)
		args.remove('-e')
	
	#ellipse width changes the width of ellipses generated by -e option
	#usage: python bojangles.py image [image2 ...] -e [ang1 ang2 ang3 ...] -ew [ellipsewidth]
	#if no width is given, default of 0.05 degrees is used
	#in the current calibration, default ellipse width translates to ~ 1-3 pixels.
	if '-ew' in args:
		if not ellipses:
			if valenza:
				print("Ellipses not enabled. Ellipse width option will be ignored.")
		else:
			i = args.index('-ew')
			try:
				ew = float(args[i+1])
				args.pop(i+1)
				if valenza:
					print("    Ellipse width: %s" %ew)
			except (ValueError,IndexError):
				if valenza:
					print("    Default ellipse width: %s" %ew)
		args.remove('-ew')
				
	
	#spaghetti will plot the events in a range(start stop step) for a given run
	#usage python bojangles.py run start stop step -s
	#run, start, stop, and step must be integers. -s can go anywhere.
	#conflicts with difference, all
	if '-s' in args:
		spaghetti = True
		runevent = True
		args.remove('-s')
		if valenza:
			print("Spaghetti plot mode enabled")
		
	#difference will plot two images and the difference between them
	if '-d' in args:
		difference = True
		args.remove('-d')
		if valenza:
			print("Difference plot enabled")
		
	#no show turns off plots
	if '-ns' in args:
		show = False
		args.remove('-ns')
		if valenza:
			print("Plotting disabled")
		
	#all will plot all of the items in a given folder.
	if '-all' in args:
		all = True
		args.remove('-all')
		if valenza:
			print("Will bojangle every image in directory")
	
	#runevent treats the image list as a list of run and event numbers
	#usage: python bojangles.py -re run1 event1 run2 event2 ...
	#could add python bojangles.py -re list.csv functionality
	if '-re' in args:
		runevent = True
		args.remove('-re')
		if valenza:
			print("Run-event image specification enabled")

	#arr will import csv arrays instead of tif images.
	if '-a' in args:
		arr = True
		args.remove('-a')
		if valenza:
			print("Array option enabled")
			
	#norm will normalize each image to have area 1 between the given angles
	#usage: python bojangles.py -n min max
	#min and max are angles in degrees. Must be a float
	if '-n' in args:
		norm = True
		i = args.index('-n')
		try:
			normmin = float(args[i+1])
			normmax = float(args[i+2])
			args.pop(i+1)
			args.pop(i+1)
			if normmin >= normmax:
				print("minimum angle must be less than maximum angle")
				return
		except (ValueError, IndexError):
			print("Invalid use of normalization")
			print("Usage: python bojangles.py image [image2, ...] -n min max")
			print("min and max must be floats and immediately after the -n option")
			return
		args.remove('-n')
		if valenza:
			print("Normalization option enabled")
			print("    Normalization range: %s, %s" %(normmin, normmax))

	
	if valenza:
		print args

	# Set up args to be the format we want for these options
	if spaghetti:
		run = int(args[0])
		start = int(args[1])
		stop = int(args[2])+1
		try:
			step = int(args[3])
		except IndexError,ValueError:
			step = 1
			evtlist = range(start,stop,step)
			runlist = [run]*length(evtlist)
			args = array([runlist,evtlist]).T
	if runevent and not spaghetti:
		args = array(args, dtype = int).reshape(-1,2)
	
	if all:
		if arr:
			args = glob.glob("*.csv")
		else:
			args = glob.glob("*.tif")
	
	# get a list of images from args
	imlist = map(getImage, args)
	data = map(lambda x: processData(x[1],x[0]+'.txt'), imlist)
	if valenza:
		print ("Number of images processed: %s" %len(data))
	if show:
		if len(data) == 1:
			anglePlot(data[0])
		elif difference:
			differencePlot(data)
		else:
			noImPlot(data)
		

main()