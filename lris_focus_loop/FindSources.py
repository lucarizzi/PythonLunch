"""
FindSources, class to find point sources in an image.
The image is divided into a grid of cells and a centroiding algorithm is performed for each cell.
The result of the centroiding algorithm is list of (xpos, ypos, fwhm, contrast, index)
If contrast is too low (default min contrast = 0.01), the search is aborted for that cell.

The method findAll returns a matrix corresponding to the grid.
"""

import numpy as np
import math

import multiprocessing.pool as mpPool

class Centroid1D:
    def __init__(self):
        pass
    
    def removeBackground (self, arr):
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
        arr = self.removeBackgroundGrad(arr)
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
        minSize = size // 5
        half = size//2
        maxHalf = half
        maxx =  self.imgWidth - size
        maxy = self.imgHeight - size
        width1 = self.imgWidth - 1
        height1 = self.imgHeight -1
        fwhm = 0
        std = 0
        ctl = -1
        dist = -1
        findCentroid = self.cAlgo.findCentroid
        getContrast = self.cAlgo.getContrast
        mct = self.minContrast
        
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
            if std == 0: 
                break
            if dist < epsilon:
                xcen, ycen = nxcen, nycen
                break                
            xcen, ycen = nxcen, nycen  
            nhalf = int(std * sfactor + 0.5)            
            half = min(max (nhalf, minSize), maxSize)
            size = half + half + 1
            size = max(5, size)
            x0 = int(xcen - half)
            y0 = int(ycen - half)
        if std > 0:
            fwhm = std * 2.35 / 1.41  
        return xcen, ycen, fwhm, ctl, dist, i    
    
    def findAll(self, gridSize, sfactor=0.9, epsilon=1E-5):
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
        maxFWHM = gridSize//1
        result = []
        for y in range(gridSize//2,height-gridSize-1, gridSize):
            y1 = y+gridSize
            for x in range(gridSize//2,width-gridSize-1, gridSize):
                x1 = x+gridSize
                centroid = self.centroid2D(x, y, x1, y1, sfactor, epsilon)
                xc,yc,fwhm,contrast,dist,cnt = centroid
                if fwhm > 0 and fwhm < maxFWHM and isInBox (xc, yc, x, y, x1, y1):
                    result.append(centroid)
        return result
    
    def findAllParallel(self, gridSize, sfactor=0.9, epsilon=1E-5, nProcess=4):
        """
        Parallel version of findAll.
        This is slower than findAll because of the GIL.
        """
        def isInBox (xc, yc, x0, y0, x1, y1):
            return (x0<=xc) and (xc<x1) and (y0<=yc) and (yc<y1)
        
        def worker (idx):
            y = gridSize * (idx // gWidth)
            x = gridSize * (idx % gWidth)
            x1, y1 = x + gridSize, y + gridSize
            #print ("idx=%d, %d, %d" % (idx, x, y))
            centroid = self.centroid2D (x, y, x+gridSize, y+gridSize, sfactor, epsilon)
            xc,yc,fwhm,contrast,dist,cnt = centroid
            if fwhm > 0 and fwhm < maxFWHM and isInBox (xc, yc, x, y, x1, y1):
                out[idx] = centroid
                
        maxFWHM = int(gridSize)
        gWidth = self.imgWidth // gridSize
        gHeight = self.imgHeight // gridSize
        gSize = gWidth * gHeight
        #print ("gwidth", gWidth, gHeight, gSize)
        
        out = [None] * gSize
        with mpPool.ThreadPool (processes=nProcess) as pool:
            pool.map(worker, range(gSize))
        return [x for x in out if x != None ]
    
    def bgSubtract(self, arr1d, deg=2):
        """
        Subtracts the background gradient
        """
        xs = np.arange(len(arr1d))
        results = np.polyfit (xs, arr1d, deg)
        func = np.poly1d (results)
        return arr1d - func(xs)
        
        