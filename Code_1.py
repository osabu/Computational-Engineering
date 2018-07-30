# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:53:05 2018

@author: Haiyan
"""

from dolfin import *
xlength=1200.0
ylength=700.0
zlength=700.0

mesh = BoxMesh (Point(0,0,0), Point(xlength,ylength,zlength),1200,700,700)
V = VectorFunctionSpace(mesh, 'P', 1)
cells = MeshFunction('size_t', mesh, 3, 0)
facets = MeshFunction('size_t', mesh, 2, 0)
dA = Measure('ds', domain=mesh, subdomain_data=facets, metdata={'quadreature_degree': 2})
dV = Measure('dx', domain=mesh, subdomain_data=cells, metadata={'quadrature_degree': 2})
left = CompiledSubDomain('near(x[0],0) && on_boundry')
right = CompiledSubDomain('near(x[0],l) && on_boundry', l=xlength)
top = CompiledSubDomain( 'pow(x[0]−X,2) + pow(x[1]−Y,2) < pow(120.0,2) && near(x[2],l) && on_boundary' ,X=xlength / 4 . , Y=ylength / 3 . , l=zlength)
facets.set_all (0)
top.mark(facets, 1)
tr = Constant ((0.0,0.0,1000.0))
null = Constant ((0.0,0.0,0.0))
bc1=DirichletBC(V, null, left)
bc2=DirichletBC(V, null, right)
bs=[bc1, bc2]

du= TrialFunction(V)
del_u = TestFunction(V)
u = Function(V)

nu=0.375
E=3000.0
G= E/(2.0*(1.0+nu))

lambada = 2.0*G*nu / (1.0-2.00*nu)
mu = G
delta = Identity(3)
#index notation
i, j, k, l = indices(4)
#deformation gradient
F = as_tensor (u[i].dx(j) + delta[i,j] , (i,j))
J = det(F)
#right Cauchy-Green deformation tensor
C = as_tensor(F[i,k]*F[i,j] , (k,j))
#Grean-Lagrange strain tensor
E = as_tensor(1./2.*( C[k,j] - delta[k,j] ) , (k,j))
#second Piola-Kirchhoff stress tensor
S = as_tensor(lambada*E[l,l]*delta[k,j] + 2.0*mu*E[k,j] , (k,j))
#nominal stress
P = as_tensor(F[i,j]*S[j,k] , (k,i))
Form = P[k,i]*del_u[i].dx(k)*dV - tr[i]*del_u[i]*dA(l)
Gain = derivative(Form, u, du)
solve(Form==0, u, bc, J=Gain, \
      solver_parameters={"newton_solver":{"linear_solver": "lu", "relative_tolerance": 1e-3} }, \
      form_compiler_parameters={"cpp_optimize":True, "representation": "quadrature", "quadratue_degree" : 2} )

\write out
pwd='/calcul/CR02/'
file = File(pwd+'nonlin_elastostatics_deformations.pvd')
file << u
\Cauchy stress tensor
sigma = as_tensor(1./J*F[j,k]*P[k.i] , (j,i))
sigma_dev = as_tensor(sigma[i,j] - 1.0/3.0*sigma[k,k]*delta[i,j] , (i,j))
eqStress = as_tensor((3.0/2.0*sigma_dev[i,j]*sigma_dev[i,j])**0.5 , () )
#calculating the equivalent stress by projecting
eqS = project(eqStress, FunctionSpace(mesh, 'P', 1))
file = File(pwd+'nonlin_elastostatics_eqStress.pvd')
file << eqS





