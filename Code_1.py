#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:13:33 2018

@author: Osman & haiyan
"""

from fenics import *
#pwd = '/home/haiyan/Documents/Try1/'
mesh = Mesh('Mesh_1.xml')
xlength= 1200.0 #in mm
ylength= 700.0
zlength= 700.0
V = VectorFunctionSpace(mesh , 'P' , 1)
cells = MeshFunction('size_t' , mesh , 'Mesh_1_physical_region.xml')
#plot(cells , interactive = True)
facets = MeshFunction('size_t' , mesh , 'Mesh_1_facet_region.xml')
#plot(facets , interactive = True)
dA = Measure('ds' , domain=mesh , subdomain_data = facets)
dV = Measure('dx' , domain=mesh , subdomain_data = cells)

bottom = CompiledSubDomain('near(x[2],0) && on_boundary')
top = CompiledSubDomain('pow(x[0]-X,2) +pow(x[1]-Y,2) < pow(120.0,2) && near(x[2], l) && on_boundary', X=xlength/2., Y=ylength/2., l=zlength)
facets.set_all(0)
top.mark(facets, 1)
#Tracing vector
tr = Constant(('0.0' , '0.0' , '1000.0'))  #MPa
null = Constant((0.0 , 0.0 , 0.0 ))
bc = [DirichletBC(V , null , bottom )]
# definition for the variational formulation
u = TrialFunction(V)
del_u = TestFunction(V)

# material parameters of strong PVC
nu = 0.325
E = 3000 #in MPa
G = E / (2.0 * (1.0 + nu))
#Lame parameters (lambda has another meaning in python)
lambada = 2.0*G*nu / (1.0 - 2.0*nu)
mu = G
# kronecker delta in 3D
delta = Identity(3)
i , j , k  = indices(3)
# Strain tensor
eps = as_tensor(1./2.*(u[i].dx(j)+u[j].dx(i)), (i,j))
# cauchy stress tensor
sigma = as_tensor(lambada*eps[k,k]*delta[i,j]+2.0*mu*eps[i,j], (i,j))
# variational form
a = sigma[j,i]*del_u[i].dx(j)*dV
L = tr[i]*del_u[i]*dA(1)
disp = Function(V)
solve(a==L, disp, bcs=bc)
#write out
file_ = File('/home/osmanabu/Downloads/Ass/Assignmt_1.pvd')
file_ << disp
