import numpy as np
import cv2
import copy
from PIL import Image 
import matplotlib.pyplot as plt

def normalize(A, minimum = 0.0, maximum = 1.0):    
    mini = np.min(A)
    maxi = np.max(A)
    B = (A - mini) / (maxi - mini) * (maximum - minimum) + minimum
    return B

def loadImg(path, normalize = True, gray = True):
    if gray:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
    if normalize:
        grayscale = 255.0
        if(img.dtype == np.uint16):
            grayscale = 65535.0
        imgData = np.asarray(img) / grayscale
        return imgData, img.dtype    
    else:
        return img, img.dtype

# Convert a gray image to an RGB image
# image: a single-channel image
def grayToRGB(image):
    H, W = image.shape
    return np.tile(np.reshape(image, [H, W, 1]), [1, 1, 3]) 

def reverseRGB(image):
    imgOut = np.zeros(image.shape)
    imgOut[:, :, 0], imgOut[:, :, 1], imgOut[:, :, 2] = np.array(image[:, :, 2]), np.array(image[:, :, 1]), np.array(image[:, :, 0])
    return imgOut         

def saveImg(filepath, image, dtype = np.uint8, revChannel = False):
    if revChannel and len(image.shape) >= 3 and image.shape[2] >= 3:
        image = reverseRGB(image)    
    if(dtype == np.uint16):
        imgOut = np.uint16(image * 65535.0)
    elif(dtype == np.uint8):    
        imgOut = np.uint8(image * 255.0) 
    else:
        imgOut = image
    cv2.imwrite(filepath, imgOut)

# ...............
def saveGrayImg(path, data, bits = 8):
    grayscale = 1 << 8
    dtype = np.uint8
    if bits == 16:
        dtype = np.uint16
        grayscale = 1 << 16
    elif bits == 32:
        dtype = np.uint32
        grayscale = 1 << 32
    im = np.array(data * grayscale, dtype = dtype)
    return cv2.imwrite(path, im)    

def showImg(image, grayscale = 255.0, size = None, gray = True):
    #imgOutA = Image.fromarray(np.uint8(img * grayscale))
    #plt.imshow(imgOutA, cmap='gray')
    if size is not None:
        image = np.reshape(image, size)
    if gray:
        plt.imshow(image, cmap='gray', vmin = 0, vmax = 1)
    else:
        plt.imshow(image, vmin = 0, vmax = 1)
    plt.show()    
    

def subimage(image, rect):
    return image[rect[0]:rect[1], rect[2]:rect[3]]    

def imagePaste(imageS, imageT, rect):
    imageT[rect[0]:rect[1], rect[2]:rect[3]] = imageS

def imageMSE(imageS, imageT):
    return np.linalg.norm(imageS - imageT)

def imageSize(shape):
    sizeI = copy.deepcopy(shape)
    if len(sizeI) < 3:    
        nH, nW = sizeI
        sizeI = (nH, nW, 1)
        sizeO = (nH, nW)
    elif sizeI[2] == 1:
        nH, nW, nCh = sizeI
        sizeO = (nH, nW)
    else:
        sizeO = copy.deepcopy(sizeI)
    return sizeI, sizeO

def mostIntensity(image, bins = 256):
    rMax = bins - 1
    histo = np.histogram(image * rMax, bins=bins, range = (0.0, bins))
    iMax = np.argmax(histo[0])
    return histo[1][iMax] / rMax

def entropy(A, bins, range = (0.0, 1.0)):
    A = np.array(A)
    H, Bins = np.histogram(A, bins = bins, range = range) 
    H = H / np.size(A)
    E = np.log2(H)
    E = np.where(np.isinf(E), 0.0, E)
    E = H * E
    return -np.sum(E)    

# aX + b
def linear2(X):
    X = np.unique(X)
    n = np.size(X)
    M = np.matrix(np.ones((n, 2)))  # 3 x 2 matrix
    I = np.reshape(np.linspace(0, n-1, n), (n, 1)) / (n - 1)
    M[:, 0] = I
    #print(np.shape(M))
    try:
        R = np.linalg.lstsq(M, X, rcond = None)
        return R[0] # [a, b]
    except:
        #return np.array([1.0, 0.0])
        return np.array([0., np.median(X)])

from pathlib import Path
def parsePath(filepath):  
    p = Path(filepath)  
    directory = p.parent 
    filename = p.stem
    extname = p.suffix
    return directory, filename, extname

def shrinkImg(image, intervalX = 1, intervalY = 1):    
    return image[::intervalY + 1, ::intervalX + 1]

def createPyramid(image, maxD = 5, minW = 0, minH = 0, intervalX = 1, intervalY = 1):
    h, w = image.shape
    lv = 1
    pyd = [image]
    imgP = image
    while lv < maxD and h >= minH and w >= minW:
        imgP = shrinkImg(imgP, intervalX, intervalY)
        pyd.append(imgP)
        h, w = imgP.shape
        lv += 1
    return pyd

def findVLines(image, thres = 0.01):
    h, w = image.shape
    VL = []
    for i in range(1, w - 1):
        I = image[:, i]       
        Ip = image[:, i - 1]            
        In = image[:, i + 1]
        dIp = I - Ip
        dIn = I - In
        tvp = (np.sum(dIp)) / h
        tvn = (np.sum(dIn)) / h
        if(np.sign(tvp) == np.sign(tvn)):
            tvp = abs(tvp)
            tvn = abs(tvn)            
            if(tvp >= thres and tvn >= thres):
                VL.append(i)    
    return VL    

def metric(A):
    return np.min(A), np.max(A), np.mean(A), np.median(A)

# Generate all pixel location in Cartesian space
# size: a 2-element tuple, (height, width)
# return: X, Y, the locations of X-axis and Y-axis, [#pixel]
def cartLoc(size):
    h, w = size
    w2 = w // 2
    h2 = h // 2
    nPx = w * h
    Xi = np.linspace(-w2, w2 - 1, w)
    Yi = np.linspace(h2 - 1, -h2, h)
    X, Y = np.meshgrid(Xi, Yi)
    return X, Y    
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)    

# ...............
# return radius[H, W] and radian[H, W]
def polarLocation(size):
    H, W = size
    Hh = H >> 1
    Wh = W >> 1    
    LH = np.arange(start = 0.0, stop = float(H), step = 1.0) - Hh
    LW = np.arange(start = 0.0, stop = float(W), step = 1.0) - Wh    
    X, Y = np.meshgrid(LW, LH)
    radius = np.sqrt(np.square(X) + np.square(Y))
    radian = np.arctan2(Y, X)
    return radius, radian

# ...............
# width: integer, the width of detector array
# numRadians: the number projection angles
# startRadian: a real number between [-PI PI], start radian.
# endRadian: a real number between [-PI PI], end radian.
# return:
#    Rd: the distance axis
#    Rr: the angle axis
def sinoAxis1(numRadians, width, startRadian = 0.0, endRadian = np.pi):
    Wh = width >> 1    
    Rd = np.arange(start = 0.0, stop = float(width), step = 1.0) - Wh    
    Rr = np.linspace(start = startRadian, stop = endRadian, num = numRadians, endpoint = False)
    return Rd, Rr


# ...................
# radius, radian: arrays of [#pixels], polor coordinates of all pixels of a CT image 
#     radius: an array of [#pixels] and each element is in [0, R].
#     radian: an array of [#pixels] and each element is in [-pi, pi].
# phi: a scalar/array in [-pi, pi], which indiate the projection angles; [#projections].
# s: a scalar/array in [-Wh.0, Wh.0], which indiate the positions of a detector elements; [#projections].
# return: weights [#projections, #pixels]
def sinoWeight(radius, radian, phi, s):
    nPrj = np.size(phi)
    nPx = np.size(radius)
    phi = np.reshape(phi, [nPrj, 1])
    s = np.reshape(s, [nPrj, 1])
    radius = np.reshape(radius, [nPx])
    radian = np.reshape(radian, [nPx])
    Xnml = radius * np.sin(phi + radian)
    weight = np.abs(np.minimum(np.abs(Xnml - s), 1.0) - 1.0)
    return weight

# .....................
# foward projection for CT
# image: array [#pixels], an image.
# weight: array [#projections, #pixels], projection weights.
# rayL: a float numer to indicate the length of a ray in px
# return: array [#projections], projections.
def fwproject(image, weight, rayL):
    WI = weight * image 
    P = np.sum(WI, axis = 1) / rayL
    return P      

def sigmoidExp(x):
    return np.where(x > 0.0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0.0)))    
 
def sigmoid(x):    
    return .5 * (1 + np.tanh(.5 * x))    

def dsigmoid(x):
    s = sigmoid(x)
    return  s * (1.0 - s)    

# cell: [row, column]
# rect: [left, right, top, bottom]
# return: True if cell in rect; other wise, False.
def isCellInRect(cell, rect):
    return (rect[0] <= cell[0] < rect[1]) and (rect[2] <= cell[1] < rect[3])


# rectST, rectT: [left, right, top, bottom], 
#    rectST: the rect of source on the target.
#    rectT: the rect of the target.
# return rectAdjS, rectAdjT: the adjusted rect of source and target.
#        True if rect1 and rect2 have overlap; other wise, False.
def rectOnRect(rectST, rectT): 
    # Target rect
    tS, bS, lS, rS = rectST
    tT, bT, lT, rT = rectT
    height = bS - tS
    width = rS - lS      
    tAS, bAS, lAS, rAS = [0, height, 0, width]
    tAT, bAT, lAT, rAT = rectST
  
    if tS < tT:
        tAS = tT - tS   
        #bAS = tAS + height     
        tAT = tT
        #bAT = tAT + height
    if bS > bT:
        bAS -= bS - bT
        bAT = bT           
    
    if lS < lT:
        lAS = lT - lS    
        #rAS = lAS + width        
        lAT = lT        
        #bAT = lAT + width
    if rS > rT:
        rAS -= rS - rT
        rAT = rT  
    
    rectAdjS = [tAS, bAS, lAS, rAS]
    rectAdjT = [tAT, bAT, lAT, rAT]
    check = rectAdjS[0] < rectAdjS[1] and rectAdjS[2] < rectAdjS[3]
    rectAdjS[0] = np.clip(rectAdjS[0], 0, height - 1)
    rectAdjS[1] = np.clip(rectAdjS[1], 1, height)
    rectAdjS[2] = np.clip(rectAdjS[2], 0, width - 1)
    rectAdjS[3] = np.clip(rectAdjS[3], 1, width)
    return rectAdjS, rectAdjT, check

# rect1, rect2: [left, right, top, bottom]
# return: True if rect1 and rect2 have overlap; other wise, False.
def isRectOnRect(rect1, rect2):
    rectAdjS, rectAdjT, check = rectOnRect(rect1, rect2)
    return check
