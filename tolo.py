import numpy as np

from cala import *


def matchCls(yL, pL):
    nSam, _ = np.shape(yL)
    pSam, _ = np.shape(pL)
    
    acc = 0
    for i in range(nSam):      
        if yL[i] == pL[i]:
            acc = acc + 1
            
    Accuracy = float(acc) / nSam
    
    
    return Accuracy, acc



def imatchCls(yL, pL):
    nSam, _ = np.shape(yL)
    pSam, _ = np.shape(pL)
    
    yB, yIndex = iMax(yL, axis=1)
    pB, pIndex = iMax(pL, axis=1)
    
    acc = 0
    
    for i in range(nSam):       
        tyl = yIndex[i]
        tpl = pIndex[i]
        
        if tyl == tpl:
            acc = acc + 1
            
    Accuracy = float(acc) / nSam
    
    
    return Accuracy, acc