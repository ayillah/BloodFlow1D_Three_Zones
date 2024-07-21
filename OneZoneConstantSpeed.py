import numpy as np
import scipy as sp

class ConstantSpeedTest:
  """
  Exact solution of constant speed problem. BCs are inflow 
  Q_{in} \cos(\Omega t) and terminal resistance R.
  """
  def __init__(self, Omega):
    self.c = 1.0
    self.L = 1.0
    self.A_r = 1.0
    self.rho = 1.0
    self.R = 3.0
    self.Q_in = 1.0
    self.Omega = Omega
    self.k = self.Omega/self.c
    self.initializeCoeffs()

  def psi(self, x):
    i = 1.0j
    psi1 = np.exp(i*self.k*x)
    psi2 = np.exp(-i*self.k*x)
    return np.array([psi1,psi2])



  def initializeCoeffs(self):

    # We'll need this
    i = 1.0j

    # RHS for the inflow condition
    rhs1 = np.complex128(0.5 * self.c * self.rho * self.Q_in / self.A_r)
    
    rhs = np.array([rhs1, 0.0])

    psi = self.psi(self.L)
    k = self.k
    dPsi = np.array([i*k*psi[0], -i*k*psi[1]])

    g = i*self.A_r * self.R / (self.rho * self.Omega)
    sr1 = psi[0] - g * dPsi[0]
    sr2 = psi[1] - g * dPsi[1]

    S_R = np.array([sr1, sr2])

    M = np.array([[0,1],S_R])
    self.C = np.linalg.solve(M, rhs)

  def uSoln(self,x,t):

    f = np.exp(1.0j*self.Omega*t)
    g = self.A_r/self.c/self.rho
    psi = self.psi(x)

    p = (self.C[0] * psi[0] + self.C[1] * psi[1])*f
    q = g*(-self.C[0]*psi[0] + self.C[1]*psi[1])*f
    return np.array([2.0*p.real, 2.0*q.real])



if __name__ == '__main__':
  model = ConstantSpeedTest(4.0*np.pi)

  C = model.C  
  print('A=', C[0])
  print('B=', C[1])

  x = np.linspace(0.0, 1.0, 11)

  t = 0.0

  u = model.uSoln(x,t)

  for i,x_i in enumerate(x):
    print('x={} p={} q={}'.format(x_i,u[0,i], u[1,i]))


  