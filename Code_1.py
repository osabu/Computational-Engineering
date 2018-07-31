# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:53:05 2018

@author: Osman & Haiyan
Last update : 30.07.18,21:37
"""

from fenics import * 
mesh = Mesh('')
V = VectorFunctionSpace(mesh , 'P' , 1)
cells = MeshFunction('size_t' , mesh , '')
plot(cells , interactive = True)
facets = MeshFunction('size_t' , mesh , '')
plot(facets , interactive = True)
dA = Measure('ds' , domain=mesh , subdomain_data = facets)
dV = Measure('dx' , domain=mesh , subdomain_data = cells)
left = CompiledSubDomain('near(x[0], 0) && on_boundary')
right = CompiledSubDomain('near(x[0],l) && on_boundary' , l = #####)
tr = Constant((0.0 , 0.0 , 1000.0))  #MPa
null = Constant((0.0 , 0.0 , 0.0 ))
bc = [DirchletBC)(V , null , facets , )]
du = TrialFunction(V)
del_u = TestFunction(V)
u = Function(V)
nu = 0.3
E = 3000 #in MPa
G = E / (2.0 * (1.0 + nu))
#Lame parameters (lambda has another meaning in python)
lambada = 2.0*G*nu / (1.0 - 2.0*nu)
mu = G 
delta = Identity(3)
i , j , k , l = indices(4)
F = as_tensor(u[i].dx(j) + delta[i,j] , (i,j))
J = det(F)
C = as_tensor(F[i,k] *F[i,j] , (k,j))
E = as_tensor(1./2.*(C[k,j] - delta[k,j]), (k,j))
S = as_tensor(F[i,j] * S[j , k] , (k ,i))
Form = P[k , i]*del_u[i].dx(k)*dV - tr[i]*del_u[i]*dA(1)
Gain = derivative(Form , u , du)
solve(Form == 0 , u , bc , J = Gain, \
	solver_parameters={"newton_solver":{"linear_solver" : "lu"
		, "relative_tolerance" : 1e-3} } ,
	form_compiler_parameters = {"cpp_optimize" : True , "
		representation" : "quadrature" , "quadrature_degree"
		: 2}  )

