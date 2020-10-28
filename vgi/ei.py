import numpy as np
import cv2
from PIL import Image 
import matplotlib.pyplot as plt
import vgi
import json
import time

QARGN = 8
        
# class Ellipse
class Ellipse:
    imageSize = (256, 256)
    arg = np.zeros([QARGN]) # [xc, yc, theta, a, b, gamma, alpha, beta]
    boundary = [0, 0, 0, 0] # [x_left, y_top, w, h] in Cartesion space
    srect = [0, 0, 0, 0] # Source rectangle, [top_row, bottom_row, left_column, right_column] in data space
    trect = [0, 0, 0, 0] # Target rectangle, [top_row, bottom_row, left_column, right_column] in data space
    isInImage = False
    image = None
    pad = 0
    sigmoidMax = 100.0
    
    sint = 0.0
    cost = 0.0    
    aa = 0.0
    bb = 0.0
    gamma = 0.0
    alpha = 0.0
    beta = 0.0
    Xt = None
    Yt = None
    XtXt = None
    YtYt = None
    
    def rotate(self, x, y):
        xt = x * self.cost + y * self.sint
        yt = -x * self.sint + y * self.cost          
        return xt, yt

          
    def updateArg(self):
        self.sint = np.sin(self.arg[2])
        self.cost = np.cos(self.arg[2])
        self.aa = self.arg[3] * self.arg[3]
        self.bb = self.arg[4] * self.arg[4] 
        #self.gamma = self.arg[5] * self.sigmoidMax / (self.arg[3]  * self.arg[3] * self.arg[4] * self.arg[4])
        #self.gamma = self.arg[5] * self.sigmoidMax / (self.imageSize[0] * self.imageSize[1])
        self.gamma = self.arg[5] * self.sigmoidMax
        self.alpha = self.arg[6]
        self.beta = self.arg[7]

    def updateBoundary(self):
         # https://math.stackexchange.com/questions/91132/how-to-get-the-limits-of-rotated-ellipse
        sintsq = self.sint * self.sint
        costsq = self.cost * self.cost        
        at = np.sqrt(self.aa * costsq + self.bb * sintsq) + self.pad
        bt = np.sqrt(self.aa * sintsq + self.bb * costsq) + self.pad
        
        # [x_left, y_top, w, h] in Cartesion space
        self.boundary = [self.arg[0] - at, self.arg[1] + bt, 
                         int(np.ceil(at + at)), int(np.ceil(bt + bt))] 
        bW = self.boundary[2]
        bH = self.boundary[3]
        
        imgH, imgW = self.imageSize
        hH = imgH // 2 
        wH = imgW // 2
        hMax = imgH - 1 
        wMax = imgW - 1 
        
        # Source on Target rect
        topST = int(hH - self.boundary[1])
        bottomST = int(topST + self.boundary[3])        
        leftST = int(self.boundary[0] + wH)        
        rightST = int(leftST + self.boundary[2])        
        rectST = [topST, bottomST, leftST, rightST]
        rectT = [0, imgH, 0, imgW]         
        
        self.srect, self.trect, self.isInImage = vgi.rectOnRect(rectST, rectT)
        
        #print('updateBoundary rectST, rectT', rectST, rectT)
        #print('updateBoundary self.srect, self.trect, self.isInImage', self.srect, self.trect, self.isInImage)
        #if self.isInImage == False:
        #    print('!', rectST, rectT)
        #self.isInImage = (leftS >= 0 and leftS < bW and rightS > 0 and rightS <= bW and topS >= 0 and topS < bH and bottomS > 0 and bottomS <= bH)              
        
    def updateImage(self):
        dataSize = (self.boundary[3], self.boundary[2])
        X, Y = vgi.cartLoc(dataSize)
        self.Xt, self.Yt = self.rotate(X, Y)
        self.XtXt = self.Xt * self.Xt
        self.YtYt = self.Yt * self.Yt    
        g = self.gamma
        q = g * (1.0 - self.XtXt / self.aa - self.YtYt / self.bb) 
        q = np.clip(q, -self.sigmoidMax, self.sigmoidMax)
        self.image = vgi.sigmoid(q)
        
    def update(self, imageSize = None, pad = 0, sigmoidMax = 100):      
        if imageSize is None:
            imageSize = self.imageSize
            pad = self.pad
            sigmoidMax = self.sigmoidMax
        else:
            self.imageSize = imageSize
            self.pad = pad
            self.sigmoidMax = sigmoidMax
        self.updateArg()
        self.updateBoundary()        
        self.updateImage()
        
    def __init__(self, arg = np.zeros([QARGN]), Q = None, imageSize = (256, 256), pad = 10, sigmoidMax = 100):
        if Q is not None:
            self.arg = np.array(Q.arg)
            imageSize = (Q.imageSize[0], Q.imageSize[1])
            pad = int(Q.pad)
        else:
            self.arg = np.array(arg)        
        self.update(imageSize, pad, sigmoidMax)         
        
    def __add__(self, arg): 
        argO = self.arg + arg    
        Qout = Ellipse(arg = argO, imageSize = self.imageSize, pad = self.pad, sigmoidMax = self.sigmoidMax)
        return Qout
    
    def __sub__(self, arg): 
        argO = self.arg - arg    
        Qout = Ellipse(arg = argO, imageSize = self.imageSize, pad = self.pad, sigmoidMax = self.sigmoidMax)
        return Qout  
    
    def __iadd__(self, arg): 
        self.arg += arg    
        self.update(imageSize = self.imageSize, pad = self.pad, sigmoidMax = self.sigmoidMax)
        return self
        #print('Q+=', self.isInImage)
    
    def __isub__(self, arg): 
        self.arg -= arg    
        self.update(imageSize = self.imageSize, pad = self.pad, sigmoidMax = self.sigmoidMax)
        return self

    # flags: an integer that uses each bit to indicate whether each argument to be calculated for derivate; default is 0xFF, which enable all arguments.
    # return: [nArg, height, width]
    def derivative(self, flags = 0xFF):
        nArg = 6
        
        bH = self.boundary[3]
        bW = self.boundary[2]
        n = self.Xt.size
        dev = np.zeros([nArg, bH, bW])  
        g = self.gamma
        g2 = g + g
        a = self.arg[3]
        b = self.arg[4]        
        Xtaa = self.Xt / self.aa #* arg.cost
        Ytbb = self.Yt / self.bb #* arg.sint
        
        dS = self.image * (1. - self.image)
        g2dS = g2 * dS

        #if flags & 1:# dxc
        #    dev[0, :, :] = g2 * (bbXt * self.cost - aaYt * self.sint) * dS
        #if flags & 2:# dyc
        #    dev[1, :, :] = g2 * (bbXt * self.sint + aaYt * self.cost) * dS
        #if flags & 4:# dtheta
        #    dev[2, :, :] = g2 * self.Xt * (self.aa * self.Xt - self.bb * self.Yt ) * dS
        #if flags & 8:# da        
        #    dev[3, :, :] = g2 * a * (self.bb - self.YtYt) * dS
        #if flags & 16:# db
        #    dev[4, :, :] = g2 * b * (self.aa - self.XtXt) * dS
        #if flags & 32:# dGamma
        #    dev[5, :, :] = (self.aa * self.bb - self.bb * self.XtXt - self.aa * self.YtYt) * dS    

        dev[0, :, :] = (Xtaa * self.cost - Ytbb * self.sint) * g2dS
        dev[1, :, :] = (Xtaa * self.sint + Ytbb * self.cost) * g2dS
        dev[2, :, :] = self.Xt * (- self.Yt / self.aa + self.Xt / self.bb ) * g2dS
        dev[3, :, :] = (self.XtXt / (self.aa * self.arg[3])) * g2dS
        dev[4, :, :] = (self.YtYt / (self.bb * self.arg[4])) * g2dS        
        return dev


# Creating ellipses 
# q: the representing ellipse, [xc, yc, theta, a, b, gamma, alpha, beta]
# n: the number of ellipses
# qw: the range 
# method: 'same', all ellipse are the same as q
#         'rand', each ellipse is choosed from q-qw to q+qw by unifom random sampling.
#         'nml', each ellipse is choosed from q-qw to q+qw by normal distribution sampling.
def createEllipseArg(q = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.8, 1.0], n = 1, qw = None, method = 'same'):
    if method == 'same':
        Qarg = np.tile(q,(n,1))
    elif method == 'rand':
        q = np.reshape(q, [QARGN, 1])
        qw = np.reshape(qw, [QARGN, 1])
        qa = q - qw
        qb = q + qw
        Qarg = np.transpose(np.random.uniform(qa, qb, size = (QARGN, n)))        
    elif method == 'nml':
        q = np.reshape(q, [QARGN, 1])
        qw = np.reshape(qw, [QARGN, 1])
        Qarg = np.transpose(np.random.normal(q, qw, size = (QARGN, n)))
    return Qarg

def clipEllipseArg(Qarg, qmin, qmax, denoise = True):
    QargO = np.clip(Qarg, a_min = qmin, a_max = qmax)
    if denoise:
        if Qarg[3] < qmin[3]:
            QargO[4]= qmin[4]
        if Qarg[4] < qmin[4]:
            QargO[3]= qmin[3]
    return QargO

def ellipseList(Qarg, imageSize, pad = 0, sigmoidMax = 100.0):
    return [Ellipse(arg = arg, imageSize = imageSize, pad = pad, sigmoidMax = sigmoidMax) for arg in Qarg]

def ellipseArgArray(Q):
    return np.array([q.arg for q in Q])

def ellipseArgList(Q):
    return [q.arg.tolist() for q in Q]

def copyEllipses(Q):
    return [Ellipse(arg = q.arg, imageSize = q.imageSize, pad = q.pad) for q in Q]

def saveEllipses(Q, path):    
    Qarg = ellipseArgList(Q)
    with open(path, 'w') as jFile:        
        json.dump(Qarg, jFile)
        
def saveEllipseSet(QSet, path):
    with open(path, 'w') as jFile:        
        json.dump(QSet, jFile)

        
def openEllipseSetFile(path):
    with open(path) as jFile:
        jData = jFile.read()
        QSet = json.loads(jData)
        return QSet
    
def drawEllipseSet(QSet, rounds = 0, sizeOut = None, mul = None):
    sizeI = QSet['imageSize']    
    if sizeOut is None:
        sizeOut = sizeI
    sizeArray = (sizeOut[0], sizeOut[1])
    pad = QSet['pad']
    QA = np.array(QSet['arg'])
    if len(QA.shape) == 3:
        # [round, #Q, #arguments]
        bg = np.full(sizeArray, QSet['bg'])
        r, nQ, nArg = QA.shape
        if rounds < 1 or rounds > r:
            rounds = r
        iR = 0    
        while iR < rounds:
            imgOut = np.zeros(sizeArray)            
            if mul is None:
                Q = ellipseList(QA[iR], sizeArray, pad)
            else:   
                QA[iR][:, 0:2] *= mul
                QA[iR][:, 3:5] *= mul
                Q = ellipseList(QA[iR], sizeArray, pad)
            imgOut, accAlpha = blendEllipses(Q, imgOut, bg = bg)        
            bg = imgOut
            iR += 1
        return imgOut
    elif len(QA.shape) == 4:
        # [round, channel, #Q, #arguments]
        r, nCh, nQ, nArg = QA.shape
        if rounds < 1 or rounds > r:
            rounds = r        
        bg = np.full(sizeOut, QSet['bg'])
        iR = 0    
        while iR < rounds:
            imgOut = np.zeros(sizeOut)
            iCh = 0
            while iCh < nCh:
                if mul is None:
                    Q = ellipseList(QA[iR][iCh], sizeArray, pad)
                else:
                    QA[iR][iCh][:, 0:2] *= mul
                    QA[iR][iCh][:, 3:5] *= mul
                    Q = ellipseList(QA[iR][iCh], sizeArray, pad)
                imgOut[:, :, iCh], accAlpha = blendEllipses(Q, imgOut[:, :, iCh], bg = bg[:, :, iCh])   
                iCh += 1      
            bg = imgOut
            iR += 1
        return np.clip(imgOut, 0.0, 1.0)
    
def siftEllipses(keypoints, imgSize, sizeRatio = 0.5, gamma = 1.0, alpha = 1.0, beta = 1.0):
    H, W = imgSize
    Wh = W / 2.0
    Hh = H / 2.0
    nQ = len(keypoints)
    Qarg = np.zeros([nQ, QARGN])
    #[xc, yc, theta, a, b, gamma, alpha, beta]
    i = 0
    for fp in keypoints:
        Qarg[i] = np.array([fp.pt[0] - Wh, H - fp.pt[1] - Hh, -np.deg2rad(fp.angle), fp.size, fp.size * sizeRatio, gamma, alpha, beta])
        i += 1
    return Q    

# Q: the parameters of ellipses, which is an array of [#ellipse, #parameters] = [t0, t1, ..., tn-1], 
#    where ti = [xc, yc, theta, a, b, gamma, alpha, beta].
# return: 
#    Areas: [#ellipse, 1]
def ellipseArea(Qarg):
    return Qarg[3] * Qarg[4] * np.pi

def blendEllipse(q, rectS, rectT, image, accAlpha):
    tT, tB, tL, tR = rectT
    sT, sB, sL, sR = rectS
    alpha = q.alpha
    beta = q.beta
    imgQA = q.image[sT:sB, sL:sR] * alpha
    #print('blendEllipse rectS, rectT', rectS, rectT)
    image[tT:tB, tL:tR] += imgQA * beta * accAlpha[tT:tB, tL:tR]
    accAlpha[tT:tB, tL:tR] *= (1.0 - imgQA)    
    return image, accAlpha

def blendEllipseAlpha(q, rectS, rectT, accAlpha):
    tT, tB, tL, tR = rectT
    sT, sB, sL, sR = rectS
    alpha = q.alpha
    beta = q.beta
    imgQA = q.image[sT:sB, sL:sR] * alpha
    accAlpha[tT:tB, tL:tR] *= (1.0 - imgQA)    
    return accAlpha
    
# Front-to-back blending a set of ellipses.
# x, y: the coordinate of a pixel
# Q: a list of ellipses,
#    q.arg = [ t0, t1, ..., tn-1], where ti = [xc, yc, theta, a, b, gamma, alpha, beta
# image: the previous blended image
# accAlpha: The accumulated of (1-aQ), [#pixels]
# return: 
#    image: [#pixels]
#    accA: The accumulated of (1-aQ), [#pixels]
def blendEllipses(Q, image, accAlpha = None, bg = None):
    imgSize = image.shape
    if accAlpha is None:
        accAlphaO = np.ones(imgSize)
    else:
        accAlphaO = np.array(accAlpha)
    for q in Q:
        #print('blendEllipses q.srect', q.srect, q.trect)
        if q.isInImage:
            image, accAlphaO = blendEllipse(q, q.srect, q.trect, image, accAlphaO)
        #else:
        #    print('#', q.srect, q.trect)
    if not (bg is None):
        image += bg * accAlphaO
    return image, accAlphaO   

# flags: an integer that uses each bit to indicate whether each argument to be calculated for derivate; default is 0xFF, which enable all arguments.
# return: [nArg, height, width]
def dBlendEllipses(Q, k, accAlpha = None, flags = 0xFF, bg = None):
    nQ = len(Q)
    qk = Q[k]
    if qk.isInImage == False:
        return None
    imgSize = qk.imageSize
    rectTk = qk.trect
    rectSk = qk.srect
    tTk, bTk, lTk, rTk = rectTk
    tSk, bSk, lSk, rSk = rectSk
    imgQk = vgi.subimage(qk.image, rectSk)
    a_dqk = qk.derivative(flags) * qk.alpha
    a_dqk = a_dqk[:, rectSk[0]:rectSk[1], rectSk[2]:rectSk[3]]    
    
    if accAlpha is None:
        accAlphaO = np.ones(imgSize)
    else:
        accAlphaO = np.array(accAlpha)    
  
    i = 0  
    while i < k:
        qi = Q[i]
        if qi.isInImage == False:
            continue
        rectTi = qi.trect
        rectASi, rectATi, isQiOnQk = vgi.rectOnRect(rectTi, rectTk)
        if isQiOnQk == True:
            accAlphaO = blendEllipseAlpha(qi, rectASi, rectATi, accAlphaO)        
        i += 1
    b_acck = qk.beta * vgi.subimage(accAlphaO, rectTk) 

    if flags & 128:# beta
        #print('dBlendEllipses qk.image', qk.image.shape)
        #print('dBlendEllipses rectTk', rectTk)  
        #print('dBlendEllipses rectTk', rectSk)  
        dIdb = qk.alpha * imgQk * vgi.subimage(accAlphaO, rectTk)
    else:
        dIdb = np.zeros(imgQk.shape)
    
    sum_ab_qi_acci = np.zeros(imgQk.shape)
    i = k + 1
    while i < nQ:
        qi = Q[i]
        rectTi = qi.trect
        rectASi, rectATi, isQiOnQk = vgi.rectOnRect(rectTi, rectTk)
        if isQiOnQk == True:
            rectASik = (rectATi[0] - rectTk[0], rectATi[1] - rectTk[0], rectATi[2] - rectTk[2], rectATi[3] - rectTk[2])
            #print('dBlendEllipses rectTi, rectTk', rectTi, rectTk)
            #print('dBlendEllipses rectASi, rectATi, rectASik', rectASi, rectATi, rectASik)            
            Iik = np.zeros(imgQk.shape)
            vgi.imagePaste(vgi.subimage(qi.image, rectASi), Iik, rectASik)
            ab_qi_acci = qi.alpha * qi.beta * Iik * vgi.subimage(accAlphaO, rectTk)         
            sum_ab_qi_acci += ab_qi_acci
            accAlphaO = blendEllipseAlpha(qi, rectASi, rectATi, accAlphaO)
        i += 1       
    if not (bg is None):
        sum_ab_qi_acci += vgi.subimage(bg, rectTk) * vgi.subimage(accAlphaO, rectTk)
        
    dqkqi = b_acck - sum_ab_qi_acci
    dIdq = a_dqk * dqkqi
    if flags & 64:# alpha
        dIda = imgQk * dqkqi
    else:
        dIda = np.zeros(imgQk.shape)
    dIdq = np.concatenate((dIdq, [dIda], [dIdb]))
    return dIdq

def updateEllipseSE(Q, k, imgT, imgB, accAlpha = None, flags = 0xFF, bg = None):
    imgSize = imgT.shape
    nQ = len(Q)
    qk = Q[k]
    if qk.isInImage == False:
        return None
    rectTk = qk.trect
    rectSk = qk.srect
    #tTk, bTk, lTk, rTk = rectTk
    #tSk, bSk, lSk, rSk = rectSk
    
    imgTk = vgi.subimage(imgT, rectTk)
    imgBk = vgi.subimage(imgB, rectTk)
    imgDk = imgTk - imgBk
    dIdqk = dBlendEllipses(Q, k, accAlpha, flags, bg = bg)
    A = dIdqk * imgDk
    Qu = -np.sum(A, axis = (1, 2))
    #timeS = time.time()
    #timeE = time.time()
    #print('updateEllipseSE time', timeE - timeS)
    return Qu
# ................................................
def ellipseImageEpoch(Q, Qm, Qup, nQ, epoches, sizeArray, targetImage, bg, qMin, qMax, batchSize = 10,
                        shrinkEpoches = 50, shrinkRate = 0.95, minNorm = 0.05, denoise = True, flags = 0b11011111,  
                        testMode = True, testEpoches = 10, testData = None):
    iE = 0
    while iE < epoches:
        if testMode and iE % 100 == 0:        
            print('--- Epoch: ', iE)
        if iE % shrinkEpoches == 0:            
            Qm *= shrinkRate
        k = 0
        normSum = 0.0
        while k < nQ:
            imgB = np.zeros(sizeArray)
            imgB, accAlpha = blendEllipses(Q, imgB, bg = bg)                
            j = k
            Qu = np.zeros([batchSize, 8])
            iB = 0
            while iB < batchSize:
                Quj = updateEllipseSE(Q, j, targetImage, imgB, flags = flags, bg = bg)
                if not (Quj is None):
                    areaRate = 1.0 / ellipseArea(Q[j].arg)         
                    Quj *= Qm[j] * areaRate
                    Qu[iB] = Quj
                j += 1
                iB += 1
                if j >= nQ:
                    break
            j = k 
            iB = 0
            while iB < batchSize:
                normSum += np.linalg.norm(Qu[iB])
                Q[j].arg -= Qu[iB] 
                Q[j].arg = clipEllipseArg(Q[j].arg, qMin, qMax, denoise)
                Q[j].update() 
                Qup[j] = Qu[iB]            
                j += 1
                iB += 1
                if j >= nQ:
                    break
            k = j 
        norm = normSum / nQ    
        if testMode and (iE % testEpoches == 0 or iE == (epoches - 1)):
            mse = vgi.imageMSE(targetImage, imgB)
            #testData.append([iE, mse])
            testData.append([iE, norm])
        iE += 1
        if norm < minNorm:
            break;
    # @ epoches
    return iE
 
# .................................................
def ellipseImage(targetPath, outputPath = '', gray = False, bg = None,
                 nQ = 50, rounds = 100, epoches = 500, batchSize = 10,
                 shrinkEpoches = 50, shrinkRate = 0.95, reduceRound = 3, reduceAreaRate = 0.9, 
                 minNorm = 0.05, denoise = True,
                 flags = 0b11011111, minLength = 1.0, gamma = 0.5 ,pad = 10, sigmoidMax = 100.0, 
                 initMethod = 'rand', randomSeed = 8051, initArg = None, gstart = 0.5, abstart = 0.5,
                 testMode = False, testEpoches = 10, onlyInitEllipse = False, actEpoches = []):
    timeS = time.time()
    sDir, sName, sExt = vgi.parsePath(targetPath)
    imgT, dtype = vgi.loadImg(targetPath, gray = gray)

    sizeI, sizeOut = vgi.imageSize(imgT.shape)
    nH, nW, nCh = sizeI
    if nCh == 1:
        imgT = np.reshape(imgT, sizeI)
    if not (initArg is None):
        # initArg: [nQ, nArg]
        nQ = initArg.shape[0]

    sizeArray = (nH, nW)
    nWh = nW/2.0
    nHh = nH/2.0
    nPx = nH * nW
    
    if testMode:  
        print('Image size:', sizeI, '; dtype:', dtype, '; value range:', vgi.metric(imgT))
        vgi.showImg(imgT, size = sizeOut)  

    #Q = loadEllipses('head130_64_Q2.json')
    rWn = nW * 0.5 
    rHn = nH * 0.5 
    rMinQ = minLength
    rWQ = max(rMinQ, nW * 0.1)
    rHQ = max(rMinQ, nH * 0.1)
    rWQr = rMinQ - 1.0
    rHQr = rMinQ - 1.0
    #rWQr = max(0.0, rHQ - rMinQ)
    #rHQr = max(0.0, rHQ - rMinQ)

    sGamma = ''
    if gamma < 1.0:
        pad = max(20, pad)
        sGamma = '_g' + str(int(gamma * 1000))
    sInitQ  = ''
    if onlyInitEllipse:
        sInitQ = '_iniQ'

    QName = sName + sInitQ + '_c' + str(nCh) + '_q'+ str(nQ) + '_R' + str(rounds) + '_e' + str(epoches) + '_b' + str(batchSize) + '_ro' + str(int(minNorm * 1000)) + '_se' + str(shrinkEpoches) + '_gs' + str(int(gstart)) + '_abs' + str(int(abstart)) + '_ra' + str(int(reduceAreaRate * 100)) + sGamma            
    if outputPath == '':        
        QJsonPath = QName + '.json'
    else:
        QJsonPath = outputPath        
    
    

    np.random.seed(randomSeed)
    if bg is None :
        if nCh == 1:
            bg = np.full(sizeI, vgi.mostIntensity(imgT, bins = 256))
        else:
            bg = np.zeros(sizeI)
            i = 0
            while i < nCh:
                bg[:, :, i] = np.full(sizeArray, vgi.mostIntensity(imgT[:, :, i], bins = 256))
                i += 1

    #vgi.showImg(bg, size = sizeI) 
    QSet = {'path': QJsonPath, 'bg': bg[0, 0].tolist(), 'imageSize':[int(nH), int(nW), int(nCh)], 'pad': pad, 'sigmoidMax': sigmoidMax, 'shape':[int(rounds), int(nCh), int(nQ), int(QARGN)], 'arg': []}
    #Qpath = 'James_q50_500_g01.json'
    qMin = [-nWh, -nHh, -3*np.pi, 0.5, 0.5, 0.001, 0.0, 0.0]
    qMax = [nWh, nHh, 3*np.pi, nHh, nWh, 1.0, 1.0, 1.0]
    gHw = nH * gstart
    gWw = nW * gstart

    testDataR = []
    iR = 0

    while iR <  rounds:
        print('Round:', iR)    
        if initArg is None:
            if flags & 64: #flags = 0b11011111
                Qarg = createEllipseArg( q = [0.0, 0.0, 0.0, rWQ,  rHQ,  gamma, 0.8, 0.8], 
                                        qw = [rWn, rHn, 0.8, rWQr, rHQr, 0.0,   0.2, 0.2], 
                                        n = nQ, method = initMethod)
            else:
                Qarg = createEllipseArg( q = [0.0, 0.0, 0.0, rWQ,  rHQ,  gamma, 1.0, 0.8], 
                                        qw = [rWn, rHn, 0.8, rWQr, rHQr, 0.0,   0.0, 0.2], 
                                        n = nQ, method = initMethod)               
            i = 0
            while i < nQ:
                Qarg[i] = clipEllipseArg(Qarg[i], qMin, qMax, denoise) #np.clip(Qarg, qMin, qMax)
                i += 1
        else:
            Qarg = initArg

        Q0 = ellipseList(Qarg, sizeArray, pad = pad, sigmoidMax = sigmoidMax)
        Qm0 = np.tile(np.array([gWw, gHw, np.pi / gstart, gWw, gHw, abstart, abstart, abstart]), (nQ, 1))
        Qup0 = np.zeros([nQ, QARGN])   

        if iR > 1 and iR % reduceRound == 0:
            rWQ = max(rMinQ, rWQ * reduceAreaRate)
            rHQ = max(rMinQ, rHQ * reduceAreaRate)
            #print('rWQ, rHQ', rWQ, rHQ)
            #rWQr = max(0.0, rHQ - rMinQ)
            #rHQr = max(0.0, rHQ - rMinQ)  

        QargRound = []
        imgOut = np.zeros(sizeI)
        mseR = 0.0

        testDataAll = []
        actEpochesV = []
        iCh = 0
        #if morphing:
        #    Q = copyEllipses(Q0)        
        while iCh < nCh:
            #Q = ellipseList(Qarg, sizeArray, pad = pad)
            #if morphing:
            #    Q = copyEllipses(Q) # For morphing, use an ellipse to represent a color
            #else:
            Q = copyEllipses(Q0)
            Qm = np.array(Qm0)
            Qup = np.array(Qup0)
            imgTc = imgT[:, :, iCh]
            imgBG = bg[:, :, iCh]
            imgB = np.zeros(sizeArray)
            imgB, accAlpha = blendEllipses(Q, imgB, bg = imgBG)  
            #if testMode:
            #    vgi.showImg(np.clip(imgB, 0.0, 1.0), size = sizeArray)    
            testData = []     
            if onlyInitEllipse == False:
                actE = ellipseImageEpoch(Q = Q, Qm = Qm, Qup = Qup, nQ = nQ, epoches = epoches, sizeArray = sizeArray, targetImage = imgTc, bg = imgBG, qMin = qMin, qMax = qMax, 
                                    batchSize = batchSize, shrinkEpoches = shrinkEpoches, shrinkRate = shrinkRate, 
                                    minNorm = minNorm, denoise = denoise, flags = flags, 
                                    testMode = testMode, testEpoches = testEpoches, testData = testData)
                actEpochesV += [actE]
            # ? onlyInitEllipse
            imgOut[:, :, iCh], accAlpha = blendEllipses(Q, imgOut[:, :, iCh], bg = imgBG) 
            QargRound.append(ellipseArgList(Q))
            if testMode:
                error = vgi.imageMSE(imgTc, imgOut[:, :, iCh])
                mseR += error       
                if not onlyInitEllipse:
                    testDataAll += [np.array(testData)]
                    #testData = np.array(testData)
                    #print('Min MSE:', np.min(testData[:, 1]), ', final:', testData[-1, 1])
                    print('Avg MSE:', mseR / (iCh + 1))
                    #plt.plot(testData[:, 0], testData[:, 1], 'go--', linewidth=1, markersize=3)
                    #print('Ch', iCh, vgi.metric(testData))
                    #plt.show()

            iCh += 1

        actEpoches += [actEpochesV]
        if testMode:
            timeE = time.time()
            print('Round:', iR, 'Time: ', timeE - timeS)
            #print(testDataAll.shape)
            #print(testDataAll)

            plt.plot(testDataAll[0][ :, 0], testDataAll[0][ :, 1], 'go--', linewidth=1, markersize=3)
            if nCh == 3:
                plt.plot(testDataAll[1][ :, 0], testDataAll[1][ :, 1], 'ro--', linewidth=1, markersize=3)
                plt.plot(testDataAll[2][ :, 0], testDataAll[2][ :, 1], 'o--', linewidth=1, markersize=3)
            plt.show()            

        QSet['arg'].append(QargRound)
        bg = imgOut 
        if testMode:
            imgOut = np.reshape(imgOut, sizeOut)        
            imgOut = np.clip(imgOut, 0.0, 1.0)            
            vgi.showImg(imgOut)
            testDataR.append([iR, mseR / nCh])
        iR += 1
    # ----------------
    imgOut = np.reshape(imgOut, sizeOut)        
    imgOut = np.clip(imgOut, 0.0, 1.0) 
    timeE = time.time()
    print('Total time:', timeE - timeS)
    if testMode:           
        vgi.showImg(imgOut)
        print(QName)
        testDataR = np.array(testDataR)
        print('Min MSE:', np.min(testDataR[:, 1]), ', final:', testDataR[-1, 1])
        plt.plot(testDataR[:, 0], testDataR[:, 1], 'bo--', linewidth=1, markersize=3)
        plt.show()  
    #print(QSet)    
    #print(QJsonPath)
    saveEllipseSet(QSet, QJsonPath)
    return QSet, imgOut    

     
# =====================================================================    
