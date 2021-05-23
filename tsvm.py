# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This python file implements the basic Twin SVM algorithm.
# 
# Reference:
# Y. H. Shao, C. H. Chun, X. B. Wang, N. Y. Deng, Improvements on Twin Support Vector Machines,
# IEEE Trans. Neural Networks, vol. 22, no. 6, pp. 962-968, 2011.
# Jayadeva, R. Khemchandani, S. Chandra, Twin Support Vector Machines for Pattern Classification,
# IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 29, no. 5, pp. 905-910, 2007.
# 
# Coded by Miao Cheng
# Email: miao_cheng@outlook.com
# Created Date: 2020-08-01
# Last Modified Date: 2021-05-06
# All rights reserved
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy import linalg as la

from cala import *
from kernel import *


# ++++++++++ The tsvm class ++++++++++
class tsvm(object):
    
    def __init__(self, X, xL, c, **kwargs):
        xSam, xDim = np.shape(X)
        
        # Default values
        if 'c' not in kwargs:
            c = 0
        
        if 'c1' not in kwargs:
            c1 = 0.1
        
        if 'c2' not in kwargs:
            c2 = 0.1
            
        if 'c3' not in kwargs:
            c3 = 0.1
            
        if 'c4' not in kwargs:
            c4 = 0.1
            
        if 'ktype' not in kwargs:
            ktype = 'lin'
            
        # ++++++++++ Get the A-B partitions of instances ++++++++++
        A = np.zeros((1, xDim))
        B = np.zeros((1, xDim))
        
        for i in range(xSam):
            tmx = X[i, :]
            tml = xL[i, :]
            
            if tml[c] == 1:
                A = np.row_stack((A, tmx))
            elif tml[c] == -1:
                B = np.row_stack((B, tmx))
            else:
                a = 1
                b = -1
                assert a == b, 'The input labels are incorrect !'
                
        
        A = A[1::, :]
        B = B[1::, :]
        
        # ++++++++++ Initialization of Parameters ++++++++++
        self.__X = X
        self.__xL = xL
        
        self.__A = A
        self.__B = B
        
        self.__c1 = c1
        self.__c2 = c2
        self.__c3 = c3
        self.__c4 = c4
        self.__ktype = ktype
        
        #self.learn(Y)
        
        pass
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # This function defines the Nonparallel support vector machine
    # Input:
    #     Q - Hessian Matrix (Require Positive Definite)
    #     t - (0, 2) Parameter to control training
    #     C - Upper bound
    #     smallvalue - Termination condition
    # Output:
    #     bestalpha - Solutions of SVM
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def qpSOR(self, Q, t, C, smallvalue):
        nRow, nCol = np.shape(Q)
        alpha0 = np.zeros((nRow, 1))
        L = np.tril(Q)
        E = np.diag(Q)
        twinalpha = alpha0
        
        # +++++++++++++++ Calculate alpha +++++++++++++++
        for j in range(nCol):
            #tmp = t / E[j, 0]
            tmp = t / E[j]
            tmq = Q[j, :]
            tmr = twinalpha[:, 0]
            tm = np.dot(tmp, tmq)
            tm = np.dot(tm, tmr)
            
            tmp = L[j, :]
            tmq = twinalpha[:, 0] - alpha0
            tn = np.dot(tmp, tmq)
            
            tmn = tm - 1 + tn
            tmn = alpha0[j, 0] - tmn
            #twinalpha[j, 0] = tmn
            twinalpha[j, 0] = tmn[j]
            
            if twinalpha[j, 0] < 0:
                twinalpha[j, 0] = 0
            elif twinalpha[j, 0] > C:
                twinalpha[j, 0] = C
            else:
                print('No condition is met !')
                
                
        alpha = np.column_stack((alpha0, twinalpha))
        
        tmp = alpha[:, 1] - alpha[:, 0]
        obj = norm(tmp, 1)
        while obj > smallvalue:
            for j in range(nCol):
                tmp = t / E[j, 0]
                tmq = Q[j, :]
                tmr = twinalpha[:, 0]
                tm = np.dot(tmp, tmq)
                tm = np.dot(tm, tmr)
                
                tmp = L[j, :]
                tmq = twinalpha[:, 0] - alpha[:, 1]
                tn = np.dot(tmp, tmq)
                
                tmn = tm - 1 + tn
                twinalpha[j, 0] = tmn
                
                if twinalpha[j, 0] < 0:
                    twinalpha[j, 0] = 0
                elif twinalpha[j, 0] > C:
                    twinalpha[j, 1] = C
                else:
                    print('No condition is met !')
                    
                    
            tmp = alpha[:, 1::]
            alpha = tmp
            alpha = np.column_stack(alpha, twinalpha)
            
        bestalpha = alpha[:, 1]
        
        return bestalpha
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # This function defines the learning procedure of Twin SVM
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def learn(self, Y):
        Xpos = self.__A
        Xneg = self.__B
        cpos = self.__c1
        cneg = self.__c2
        eps1 = self.__c3
        eps2 = self.__c4
        
        ktype = self.__ktype
        m1, aDim = np.shape(Xpos)
        m2, bDim = np.shape(Xneg)
        e1 = - np.ones((m1, 1))
        e2 = - np.ones((m2, 1))
        
        # ++++++++++++++++++++++++++++++
        if ktype == 'lin':
            tmp = - e1      # Unnecessary negative
            H = np.column_stack((Xpos, tmp))
            tmq = - e2
            G = np.column_stack((Xneg, tmq))
            
        else:
            X = np.row_stack((self.__A, self.__B))
            
            kwargs = []
            kwargs['ktype'] = 'Gaussian'
            tmp = getKernel(Xpos, X, kwargs)
            tmq = - e1
            H = np.column_stack((tmp, tmq))
            
            tmp = getKernel(Xneg, X, kwargs)
            tmq = - e2
            G = np.column_stack((tmp, tmq))
            
        
        # ++++++++++ DTWSVM1 ++++++++++
        HH = np.dot(np.transpose(H), H)
        nRow, nCol = np.shape(HH)
        assert nRow == nCol, 'The number of rows and columns are not identical !'
        
        tme = np.eye(nRow)
        HH = HH + eps1 * tme
        tmp = la.inv(HH)
        tmq = np.transpose(G)
        HHG = np.dot(tmp, tmq)
        
        tmp = np.dot(G, HHG)
        tm = tmp + np.transpose(tmp)
        kerH1 = tm / 2
        
        alpha = self.qpSOR(kerH1, 0.5, cpos, 0.05)
        tmp = np.dot(HHG, alpha)
        vpos = - tmp
        
        # ++++++++++ DTWSVM2 equ(29) ++++++++++
        GG = np.dot(np.transpose(G), G)
        nRow, nCol = np.shape(GG)
        assert nRow == nCol, 'The number of rows and columns are not identical !'
        
        tme = np.eye(nRow)
        GG = GG + eps2 * tme
        tmp = la.inv(GG)
        tmq = np.transpose(H)
        GGH = np.dot(tmp, tmq)
        
        tmp = np.dot(H, GGH)
        tm = tmp + np.transpose(tmp)
        kerH1 = tm / 2
        gamma = self.qpSOR(kerH1, 0.5, cneg, 0.05)
        vneg = np.dot(GGH, gamma)
        
        del kerH1, H, G, HH, HHG, GG, GGH
        
        m = len(vpos)
        n = len(vneg)
        w1 = vpos[0:m-1]
        b1 = vpos[m-1]
        w2 = vneg[0:n-1]
        b2 = vneg[n-1]
        
        # ++++++++++ Predicate and Output ++++++++++
        yRow, yCol = np.shape(Y)
        
        if ktype == 'lin':
            H = Y
            tmp = np.dot(np.transpose(w1), w1)
            w11 = np.sqrt(tmp)
            tmp = np.dot(np.transpose(w2), w2)
            w22 = np.sqrt(tmp)
            
            tmp = np.dot(H, w1)
            tmq = np.ones((yRow, 1)).reshape(yRow, )
            tmq = b1 * tmq
            y1 = tmp + tmq
            
            tmp = np.dot(H, w2)
            tmq = np.ones((yRow, 1)).reshape(yRow, )
            tmq = b2 * tmq
            y2 = tmp + tmq
            
        else:
            C = np.row_stack((self.__A, self.__B))
            H = getKernel(Y, C, 'Gaussian')
            
            tmk = getKernel(X, C, 'Gaussian')
            tmp = np.dot(np.transpose(w1), tmk)
            tm = np.dot(tmp, w1)
            w11 = np.sqrt(tm)
            
            tmp = np.dot(np.transpose(w2), tmk)
            tm = np.dot(tmp, w2)
            w22 = np.sqrt(tm)
            
            tmp = np.dot(H, w1)
            tmq = np.ones(yRow, 1)
            tmq = b1 * tmq
            y1 = tmp + tmq
            
            tmp = np.dot(H, w2)
            tmq = np.ones(yRow, 1)
            tmq = b2 * tmq
            y2 = tmp + tmq
            
        
        tmp = np.dot(np.transpose(w1), w2)
        tmp = 2 * tmp
        tmq = np.dot(w11, w22)
        
        if abs(tmq) < 1e-6:
            tmq = np.sign(tmq)
            
        if tmq == 0:
            tmq = 1
            
        tm = tmp / tmq
        tm = 2 + tm
        wp = np.sqrt(tm)
        
        tmp = np.dot(np.transpose(w1), w2)
        tmp = 2 * tmp
        tmq = np.dot(w11, w22)
        
        #if abs(tmq) < 1e-6:
            #tmq = np.sign(tmq)
            
        #if tmq == 0:
            #tmq = 1
        
        tm = tmp / tmq
        tm = 2 - tm
        wm = np.sqrt(tm)
        
        del H
        if ktype != 'lin':
            del C
        
        # ++++++++++++++++++++
        if abs(w11) < 1e-6:
            w11 = np.sign(w11)
            
        if w11 == 0:
            w11 = 1
        
        m1 = y1 / w11
        
        #if abs(w22) < 1e-6:
            #w22 = np.sign(w22)
            
        #if w22 == 0:
            #w22 = 1
        
        m2 = y2 / w22
        
        mp = m1 + m2
        mp = mp / wp
        mn = m1 - m2
        mn = mn / wm
        
        tmp = np.abs(mp)
        tmq = np.abs(mn)
        
        mind = vMin(tmp, tmq)
        maxd = vMax(tmp, tmq)
        
        tmp = np.abs(m2)
        tmq = np.abs(m1)
        tm = tmp - tmq
        
        #Predict_Y = np.sign(tm)
        Predict_Y = tm
        
        return Predict_Y
        
        
        