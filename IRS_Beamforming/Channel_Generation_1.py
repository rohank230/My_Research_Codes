import numpy as np
import matplotlib.pyplot as plt

class Generate_Channel:
    def __init__(self, K, M, N):
        self.K= K
        self.M= M
        self.N= N
        self.beta_AI= 10**(3/10)
        
    def Generate_Channels(self):
        Hr= np.zeros((self.N, self.K), dtype= complex)
        Hd= np.zeros((self.M, self.K), dtype= complex)
        alpha_AI= 2.8
        alpha_Iu= 2.8
        alpha_Au= 3.5
        beta_AI= self.beta_AI
        beta_Iu= 10**(3/10)

        d0= 51
        G= np.sqrt(beta_AI/(beta_AI+1)) + np.sqrt(1/(beta_AI+1)) * (np.random.randn(self.N, self.M) + 1j * np.random.randn(self.N, self.M)) / np.sqrt(2)
        G*= np.sqrt(10**(-3) * d0**(-alpha_AI))

        for k in range(1, self.K+1):
            h_r_H = np.sqrt(beta_Iu/(beta_Iu+1)) + np.sqrt(1/(beta_Iu+1)) *(np.random.randn(1, self.N) + 1j * np.random.randn(1, self.N)) / np.sqrt(2)
            h_d_H = (np.random.randn(1, self.M) + 1j * np.random.randn(1, self.M)) / np.sqrt(2)

            if k==1:
                d_v= 5/np.sqrt(2)
                d_Au= ((d0-d_v)**2 + d_v**2)**0.5
                d_Iu= 5
            elif k==2:
                d_v= 3/np.sqrt(2)
                d_Au= ((d0-d_v)**2 + d_v**2)**0.5
                d_Iu= 3
            hr_H= h_r_H * np.sqrt(10**(-3)* d_Iu**-alpha_Iu)
            hd_H= h_d_H * np.sqrt(10**(-3)* d_Au**-alpha_Au)
            hr= np.conj(hr_H).T
            hd= np.conj(hd_H).T
            Hr[:, k-1]= hr[:, 0]
            Hd[:, k-1]= hd[:, 0]
        return Hd, Hr, G