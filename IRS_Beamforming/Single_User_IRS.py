import numpy as np

class Single_User:
    def __init__(self, K, M, N, option, value):
        self.N= N
        self.M= M
        self.K= K
        self.option= option
        if self.option=='power':
          self.power= value
        elif self.option=='SINR':
          self.SINR= value
        self.epsilon= 1e-4
        
    def Alternating_Optimization(self, hd_H, hr_H, G):
      TP_dBm_iter=[]
      iter=0
      w= hd_H.conj().T/np.linalg.norm(hd_H.conj().T)
      #while abs(break_condition)>self.epsilon:
      while iter<10:
        iter+=1
        # Let IRS be in the receiving mode and the AP broadcast with w. The IRS estimates phase0 and computes phase shift using w_k
        phase0= np.angle(hd_H @ w) * np.ones(self.N)

        # Estimate theta (phase shift) at the IRS
        theta= phase0.flatten() - np.angle((np.diag(hr_H.flatten()) @ G @ w).flatten())

        v= np.exp(1j*theta)
        Theta= np.diag(v)

        # Let the IRS be in the reflecting mode with given theta and the user broadcast a pilot signal. The AP estimates the composite channel and computes the new transmit beamforming w
        # Optimal MRT transmit beamforming w_MRT
        h_H= hr_H @ Theta @ G + hd_H
        if self.option=='SINR':
          P= self.SINR/ np.linalg.norm(h_H @ w)**2
          w_MRT= np.sqrt(P) * (np.conj(h_H).T)/ np.linalg.norm(h_H)
        elif self.option=='power':
          w_MRT= np.sqrt(self.power) * (np.conj(h_H).T)/ np.linalg.norm(h_H)
        # To make the phase0 equal to zero
        a= -np.angle(hd_H @ w_MRT)
        w= w_MRT*np.exp(1j*a)

        # if iter>=2:
        #   break_condition= TP_dBm_iter[-2]-TP_dBm_iter[-1]
      return w, np.diag(Theta)