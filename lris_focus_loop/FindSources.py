"""
FindSources, class to find point sources in an image.
The image is divided into a grid of cells and a centroiding algorithm is performed for each cell.
The result of the centroiding algorithm is list of (xpos, ypos, fwhm, contrast, index)
If contrast is too low (default min contrast = 0.01), the search is aborted for that cell.

The method findAll returns a matrix corresponding to the grid.
"""

import numpy as np
import math
import queue

class Centroid1D:
    def __init__(self):
        self.removeBackground = self.removeBackgroundGrad
    
    def removeBackgroundFlat (self, arr):
        def stats (arr):
            return np.min(arr), np.std(arr)
        """
        Removes background noise
        """
        size = len(arr)
        tenpc = size // 10 + 1
        m1, s1 = stats(arr[0:tenpc])
        m2, s2 = stats(arr[-tenpc:])
        m = (m1+m2)/2
        s = (s1+s2)/2
        #print ("bkg", m, s)
        bkgd = m + 3*s
        narr = arr - bkgd
        narr[narr<0] = 0
        return narr    

    def removeBackgroundGrad (self, arr):
        size = len(arr)
        tenpc = size // 10 + 1
        m1 = np.mean(arr[0:tenpc])
        m2 = np.mean(arr[-tenpc:])
        bgArr = np.linspace(m1, m2, size)
        self.bgArr = bgArr
        res = np.subtract(arr, bgArr)
        res[res < 0] = 0
        return res
        
    def findCentroid(self, arr):
        """
        One step 1D centroiding algo.
        Returns centroid position
        """
        arr = self.removeBackground(arr)
        l = arr.shape[0]
        ixs = np.arange(l)
        ixs2 = ixs * ixs
        sumarr = arr.sum()
        if sumarr == 0:
            return l/2, 0
        cen = np.dot(arr, ixs)/sumarr
        return cen, max(0, np.dot(arr, ixs2)/sumarr - cen*cen)
    
    def getContrast(self, arr):
        try:
            avg = np.mean(arr)
            minArr = np.compress(arr < avg, arr)
            maxArr = np.compress(arr > avg, arr)
            if len(minArr) <= 0 or len(maxArr) <= 0:
                return 0
            mn = np.mean(minArr)
            mx = np.mean(maxArr)
            denom = mx + mn
            if denom == 0:
                return 0
            return (mx - mn) / denom
        except:
            pass
        return 0
    
class FindSources:
    def __init__(self, imgData, minContrast=0.01):
        self.orgImgData = imgData
        self.imgHeight, self.imgWidth = imgData.shape
        self.iimg = self.integral (imgData)
        self.minContrast = minContrast
        self.cAlgo = Centroid1D()
        self.isCentOKFunc = self.isCentOK
    
    def isCentOK (self, varx, vary, fwhm, half):
        return varx > 0 and vary > 0 and fwhm < half
    
    def integral (self, img):
        return img.cumsum(axis=1).cumsum(axis=0)
    
    def rowSum (self, iimg, x0, y0, x1, y1):
        """
        Returns sum in row direction, ie fixed y        
        """
        v0 = y0-1
        v1 = y1-1
        lt = iimg[v0:v1, x0]
        lb = iimg[y0:y1, x0]        
        rt = iimg[v0:v1, x1]
        rb = iimg[y0:y1, x1]
        return lt + rb - rt - lb
    
    def colSum (self, iimg, x0, y0, x1, y1):
        """
        Returns sum in colum direction, ie fixed x       
        """
        u0 = x0-1
        u1 = x1-1
        lt = iimg[y0, u0:u1]
        lb = iimg[y1, u0:u1]
        rt = iimg[y0, x0:x1]
        rb = iimg[y1, x0:x1]
        return lt + rb - rt - lb
    
    def centroid2D(self, ix0, iy0, ix1, iy1, sfactor=0.9, epsilon=1E-5, bgMode=0, nloop=20):
        """        
        Given region of interest roi, returns the centroid position x,y
        Img: input image
        Iimg: integral of input image
        Uses marginal sum algorithm.
        """
        x0,y0,x1,y1 = ix0,iy0,ix1,iy1
        iimg = self.iimg
        width = x1-x0
        height = y1-y0
        xcen = x0 + width//2
        ycen = y0 + height//2
        size = width if width < height else height
        maxSize = size
        minSize = max(3, size // 5)
        half = size//2
        maxStd = size // 3
        maxHalf = half
        maxx =  self.imgWidth - size
        maxy = self.imgHeight - size
        width1 = self.imgWidth - 1
        height1 = self.imgHeight -1
        fwhm = 0
        std, varx, vary = 0, 0, 0
        ctl = -1
        dist = -1
        findCentroid = self.cAlgo.findCentroid
        getContrast = self.cAlgo.getContrast
        mct = self.minContrast
        centOK = self.isCentOK
        
        for i in range(nloop):            
            x0 = max(x0, 1)            
            y0 = max(y0, 1)
            x1 = x0 + size
            y1 = y0 + size
            x1 = min(x1, width1)            
            y1 = min(y1, height1)
            rSum = self.rowSum(iimg, x0, y0, x1, y1)
            if bgMode > 0:
                rSum = self.bgSubtract (rSum)
            ctl = getContrast(rSum)
            if ctl < mct:
                fwhm = 0
                std = 0
                break
            cSum = self.colSum(iimg, x0, y0, x1, y1)            
            if bgMode > 0:
                cSum = self.bgSubtract (cSum)
                
            #self.rSum, self.cSum = rSum, cSum
            nycen,vary = findCentroid(rSum)
            nxcen,varx = findCentroid(cSum)
            nxcen += x0
            nycen += y0
            dist = math.hypot (nxcen-xcen, nycen-ycen)        
            std = math.sqrt(varx+vary)
            fwhm = std * 2.35 / 1.41
            if not centOK(varx, vary, fwhm, half): #varx == 0 or vary == 0 or fwhm > half:
                fwhm = 0
                break
            if dist < epsilon:
                xcen, ycen = nxcen, nycen
                break                
            xcen, ycen = nxcen, nycen  
            nhalf = int(fwhm * 2 * sfactor + 0.5)            
            half = min(max (nhalf, minSize), maxHalf)
            size = half + half + 1
            x0 = int(xcen - half)
            y0 = int(ycen - half)
        return xcen, ycen, fwhm, ctl, dist, i    
    
    def findAll(self, gridSize, sfactor=0.9, epsilon=1E-5, incr=0, nloop=20):
        """
        Returns a list of centroids.
        A centroid is (xcenter, ycenter, fwhm, contrast, dist)
        dist = distance from last centroid
        """
        
        def isInBox (xc, yc, x0, y0, x1, y1):
            return (x0<=xc) and (xc<x1) and (y0<=yc) and (yc<y1)
        
        height, width = self.orgImgData.shape
        #ny = int((height-1) / gridSize)
        #nx = int((width-1) / gridSize)
        if incr == 0:
            incr = gridSize            
        maxFWHM = gridSize//1
        result = []
        for y in range(gridSize//2,height-gridSize-1, incr):
            y1 = y+gridSize
            for x in range(gridSize//2,width-gridSize-1, incr):
                x1 = x+gridSize
                centroid = self.centroid2D(x, y, x1, y1, sfactor, epsilon, nloop=nloop)
                xc,yc,fwhm,contrast,dist,cnt = centroid
                if fwhm > 0 and fwhm < maxFWHM and isInBox (xc, yc, x, y, x1, y1):
                    result.append(centroid)
        return result
    
    def bgSubtract(self, arr1d, deg=2):
        """
        Subtracts the background gradient
        """
        xs = np.arange(len(arr1d))
        results = np.polyfit (xs, arr1d, deg)
        func = np.poly1d (results)
        return arr1d - func(xs)
        
def centroid1DLoop(arr, fromIdx, toIdx, nLoops=10, epsilon=1E-1):
    """
    Finds the centroid by repeatedly centering and recalculating 
    until the centroid position changes by less than epsilon.
    
    Returns status, centroid position, standard deviation, iterations
    
    status: 0 OK, -1 bad centroid or no signal
    centroid position: position relative to input array, ie. 0 is first pixel
    standard deviation: standard deviation as calculated by the centroid algorithm (assumed Gaussian stats)
    iterations: number of iterations needed until change is less than epsilon
    """
    def limit(x):
        if x < 0: return 0
        if x >= length: return length
        return x
    
    length = len(arr)
    radius = (toIdx - fromIdx)/2
    lastCenPos = -9999
    f1d = Centroid1D()
    for i in range(nLoops):
        fromIdx = int(limit(fromIdx))
        toIdx = int(limit(fromIdx + radius + radius + 0.5))
        pos, cenVar = f1d.findCentroid(arr[fromIdx:toIdx])
        cenStd = math.sqrt(abs(cenVar))
        cenPos = pos + fromIdx
        #print (i, fromIdx, toIdx, cenPos, cenStd, lastCenPos)
        
        if cenPos < fromIdx or toIdx < cenPos:
            return -1, 0, 0, i
        
        if abs(lastCenPos - cenPos) < epsilon:
            return 0, cenPos, cenStd, i
        if cenStd > radius/3:
            return -1, cenPos, cenStd, i
        fromIdx = cenPos - radius
        lastCenPos = cenPos
        
    return -1, cenPos, cenStd, i      
        
