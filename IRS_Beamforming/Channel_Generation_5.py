import numpy as np
class Channel_Generation_5:
  def __init__(self, K, M, N):
    self.K= K
    self.M= M
    self.N= N
    self.epsilon= 10**(3/10)
  
  def calculate_distance(self, loc1, loc2):
    distance= np.sqrt((loc2[0]-loc1[0])**2 + (loc2[1]-loc1[1])**2 + (loc2[2]-loc1[2])**2)
    return distance
  def Generate_Channels(self):
    #Initialize path loss parameters
    alpha_AI= 2.5
    alpha_Iu= 2.8
    alpha_Au= 2.1
    Hd= np.zeros((self.M, self.K), dtype= complex)
    Hr= np.zeros((self.N, self.K), dtype= complex)
    loc_BS= (50, -10, 0)
    loc_IRS= (0, 0, 0)
    #loc_users= [(50, -50, 10), (0, 0, -10), (50, -50, 20), (5, 5, -5), (50, -50, -10), (0, 0, 10), (50, -50, -20), (5, 5, 5)]
    loc_users = [(4,5,0), (4,25,0), (30,25,0),(30,5,0)]
    d_AI= self.calculate_distance(loc_BS, loc_IRS)
    
    steering_vector_BS= np.exp(1j* np.pi *np.array(range(self.M)) * (loc_IRS[0]-loc_BS[0])/d_AI)
    for k in range(self.K):
      d_Iu= self.calculate_distance(loc_IRS, loc_users[k])
      steering_vector_IRS= np.exp(1j*np.pi * ((np.mod(range(self.N), 10) * (loc_users[k][1]-loc_IRS[1])/d_Iu) + (np.floor(np.array(range(self.N))/10) * (loc_users[k][2]-loc_IRS[2])/d_Iu)))
    
    #beta_AI= 30 + 22*np.log(d_AI)

    #Generate channel from AP to IRS
    beta_AI= 10**(-3) * d_AI**(-alpha_AI)
    G_NLOS= (np.random.randn(self.M, self.N) - 1j * np.random.randn(self.M, self.N)) / np.sqrt(2)
    G_LOS= steering_vector_BS.reshape(-1, 1) @ steering_vector_IRS.reshape(1, -1)
    G= np.sqrt(beta_AI)*(np.sqrt(self.epsilon/(1 + self.epsilon)) * G_LOS + np.sqrt(1/(1 + self.epsilon)) * G_NLOS)
    G= G.T

    
    # Generate AP-user and IRS-user channels for each user
    for k in range(self.K):
      d_Au= self.calculate_distance(loc_BS, loc_users[k])
      d_Iu= self.calculate_distance(loc_IRS, loc_users[k])
      #beta_Au= 32.6 + 36.7 * np.log(d_Au)
      beta_Au= 10**(-2) * d_Au**(-alpha_Au)
      #beta_Iu= 30 + 22 * np.log(d_Iu)
      beta_Iu= 10**(-3) * d_Iu**(-alpha_Iu)
          
      Hd[:, k]= np.sqrt(beta_Au) * np.sqrt(1/(1+self.epsilon)) * (np.random.randn(self.M) + 1j * np.random.randn(self.M)) / np.sqrt(2)
      steering_vector_Iu= np.exp(1j*np.pi * ((np.mod(range(self.N), 10) * (loc_users[k][1]-loc_IRS[1])/d_Iu) + (np.floor(np.array(range(self.N))/10) * (loc_users[k][2]-loc_IRS[2])/d_Iu)))
      hr_LOS= steering_vector_Iu
      hr_NLOS= (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2)
      
      Hr[:, k]= np.sqrt(beta_Iu) * ((np.sqrt(self.epsilon/(1 + self.epsilon)) * hr_LOS) + (np.sqrt(1/(1 + self.epsilon)) * hr_NLOS))
    return Hd, Hr, G  
    