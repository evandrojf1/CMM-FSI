'''
This is a directional code to understand the implementation of Coupled Momentum Method FSI with some examples.
'''

from fenics import *
import time
from mshr import *
import numpy as np
import ufl
import time
import math
import os.path

teste = 1  #Help create a directore to save the simulation files

# Mesh creation or import

P0 = Point(0.0, 0.0, 0.0)  #Center inlet
P1 = Point(4.2, 0.0, 0.0)  #Center outlet
re = 0.3
ri = 0.3  #radius inlet
rf = 0.3  #radius outlet
res = 120  #number of faces on the side when generating a polyhedral approximation.
n_cells = 60 #refiniment level
domain = Cylinder(P0,P1,re,re,res)
mesh = generate_mesh(domain, n_cells)

he = mesh.hmin()   #minimum element size  (to define CFL condition)
n = FacetNormal(mesh)    #mesh normal unit vector
File('cylinder'+str(teste)+'.pvd') << mesh   #saving the mesh


# Function spaces: this is an example to solve directely Navier-Stokes.

V = VectorElement("Lagrange", tetrahedron, 1)  #Velocity vector element
P = FiniteElement("Lagrange", tetrahedron, 1)  #Pressure scalar element
VP = MixedElement([V,P])   #mixed element 
W = FunctionSpace(mesh, VP)   #mixed function space
A = VectorFunctionSpace(mesh, 'Lagrange', 1)   # this function space is equal to subspace W.sub(0) but is used sometimes to avoid bugs



#if mesh is created it is need to define and mark boundaries:

# Defining boundaries:

class Inlet(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and near(x[0],0.0)

class Outlet(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary and near(x[0], 4.2)

class Interface(SubDomain):
	def inside(self, x, on_boundary):
		dy = x[1]
		dz = x[2]
		dr = (dy*dy + dz*dz)**0.5
		bmarg = DOLFIN_EPS + 3.0E-2
		#return on_boundary and dr >( ri - bmarg) and between(x[0], (0.1,4.1))
		return on_boundary and between(x[0], (0.1,4.1))

# Marking boundaries

parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0) 

parts.set_all(0)
DomainBoundary().mark(parts,3)
Inlet().mark(parts,1)
Outlet().mark(parts,2)
Interface().mark(parts,4)
ds = Measure('ds', domain=mesh,subdomain_data=parts)  #used to define an specific boundary for neumman bc


#Boundary Conditions: some examples

#vx = '(1+sin(2*pi*t/T))*Vm*(r*r-x[1]*x[1]-x[2]*x[2])/(r*r)'    #Senoidal 3D Parabolic profile inlet
#vx = '(259.06*pow(t,8) - 1260.9*pow(t,7) + 2518.9*pow(t,6) - 2634.6*pow(t,5) + 1508*pow(t,4) - 431.96*pow(t,3) + 35.383*pow(t,2) + 6.064*t + 0.29954)*Vm*((r*r-x[1]*x[1]-x[2]*x[2])/(r*r))'
vx = 'Vm*((r*r-x[1]*x[1]-x[2]*x[2])/(r*r))'  #3D Parabolic profile inlet
vin = Expression((vx, '0.0', '0.0'), t=0.0, r=ri, pi=3.1415, Vm=12.5, T=0.2, degree=2)  #Vm = velocidade média (center-line).
vwalls = Constant((0.,0.,0.))  #No-slit condition
pout = Constant(1330.0)  #Pressure outside


#if mesh was created:

bcvin = DirichletBC(W.sub(0), vin, parts, 1)
bcvwalls = DirichletBC(W.sub(0), vwalls, parts, 3)
bcvinterface = DirichletBC(W.sub(0), vwalls, parts, 4)
bcpout = DirichletBC(W.sub(1), pout, parts, 2)

bcs_rigid = [bcvin, bcvwalls, bcvinterface, bcpout]


###############SIMULATION PARAMETERS################

dt = 0.8*he/12.5
itera = 100
idt = 1./dt
t_end = 0.4

print('dt: ', dt)
print('\n\nNum iteracoes: ', int(t_end/dt))



###############FLUID PROBLEM################

#Fluid parameters
mu_f = 0.04     		#Fluid dynamic viscosity
rho_f = 1.06    		#Fluid density
nu_f = Constant(mu_f/rho_f)     #Fluid kinematic viscosity
f = Constant((0.,0.,0.))	#Fluid body forces



#Trial and test functions
vp = TrialFunction(W)
v, p = ufl.split(vp)                 #v = fluid velocity, p = pressure : Trial Functions!
w, q = TestFunctions(W)              #w = velocity test function (momentum) , q = pressure test function (continuity)

#Functions
v0 = Constant((0.,0., 0.))           #Initial velocity
vp1 = Function(W)		     #solution function for rigid problem
v1, p1 = vp1.split()		     #solution for velocity (v1) e pressure(p1) separeted rigid problem

vn = interpolate(v0, W.sub(0).collapse())   #Defining initial velocity for previously time step (i=0).



#Stress tensor for fluid
def epsilon_f(v):
	return sym(nabla_grad(v))

def sigma_f(v,p,nu_f):
	return 2*nu_f*epsilon_f(v)-p*Identity(len(v))



#Fluid equation using Standard Galerkin discretization
F = rho_f*(1.0/dt)*inner(v-vn,w)*dx + rho_f*inner(dot(v1,nabla_grad(v)),w)*dx + inner(sigma_f(v,p,nu_f),epsilon_f(w))*dx + q*div(v)*dx - inner(f,w)*dx + inner(p*n,w)*ds - nu_f*inner(w,grad(v).T*n)*ds



##########STABILIZATION FOR USING P1-P1#############
h = 2.0*Circumradius(mesh)		#mesh size
vnorm = sqrt(dot(v1, v1))		#velocity norm	

#Residuo momento
R = (1.0/dt)*(v-vn)+dot(v1,nabla_grad(v))-div(sigma_f(v,p,nu_f)) + nabla_grad(p)

#PSPG
tau_pspg = (h*h)/2.
F_pspg = tau_pspg*inner(R,nabla_grad(q))*dx

#SUPG
tau_supg = ((2.0*idt)**2 + (2.0*v/h)**2 + 9.0*(4.0*nu_f/h**2)**2 )**(-0.5)
F_supg = tau_supg*inner(R,dot(v1,nabla_grad(w)))*dx

#LSIC
tau_lsic = 0.5*vnorm*h
F_lsic = inner(div(v),div(w))*tau_lsic*dx


#Galerking + stabilizations
F_est = F + F_pspg + F_supg + F_lsic



###########SOLID PROBLEM################

#Solid parameters
nu_s = 0.5		#Wall viscosity
rho_s = 1.09		#Wall density
E = 4.07E6		#Wall elasticity
qsi = 0.03		#wall thikness (zeta)
ks = Constant(5.0/6.0)		#parabolic variation of transverse shear stress through the membrane  defined as 5/6
beta = 1.0		#Parameter for Newmark equation for deformation initialization  defined as 1

#ks = Constant(ks)
#nu_s = Constant(nu_s)


#Deformable Dirichlet Boundary Conditions
bcs_coupled = [bcvin, bcpout, bcvwalls]  #In the deformable case the no-slip condition for wall is removed


#Functions
u = TrialFunction(A)		#Trial function for displacement

#equations functions
vp_ = Function(W)				#Coupled problem solution
v_, p_ = vp_.split()				#Solution for velocity (v_) e pressure(p_) separeted for the deformable problem
u_ = Function(A)				#Solution for the displacement(u_) : using FunctionSpace (A) to avoid bugs

#timestep functions
v_n = Function(W.sub(0).collapse())		#Velocity of Previously time step 
v_2n = Function(W.sub(0).collapse())		#Velocity of 2time step behind
u_n = Function(A)				#Previously displacement
p_n = Function(W.sub(1).collapse())		#Previously pressure

u_n = interpolate(v0, A)	#initial displacement



######FINITE ELEMENT MODEL FOR THE VESSEL WALL###### : topic 2.2 from article figueroa2006


norm_n = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2])**(1/2.0)
n1 = 1/norm_n*n

#primeiro vetor tangente
t1 = as_vector((n1[1] - n1[2], n1[2] - n1[0], n1[0] - n1[1]))
norm_t1 = (t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2])**(1/2.0)
t1 = 1/norm_t1*t1

#segundo vetor tangente
t2 = as_vector([n1[1]*t1[2] - n1[2]*t1[1], n1[2]*t1[0] - n1[0]*t1[2], n1[0]*t1[1] - n1[1]*t1[0]])
norm_t2 = (t2[0]*t2[0] + t2[1]*t2[1] + t2[2]*t2[2])**(1/2.0)
t2 = 1/norm_t2*t2


#tensor deslocamento e da tensao para solido
def epsilon_s(u):
	e=grad(u)

	exx = dot(dot(t1,e),t1)
	eyy = dot(dot(t2,e),t2)
	ezz = 0.0*dot(dot(n1,e),n1)
	txy = dot(dot(t1,e),t2) + dot(dot(t2,e),t1)
	txz = dot(dot(n1,e),t1)
	tyz = dot(dot(n1,e),t2)

	ep = as_vector( [ exx,eyy,ezz,txy,txz,tyz] )

	return ep

def sigma_s(u):	

	D_coef = E/(1.0 - nu_s*nu_s)
	
	ttzz = 0.2083
	ttxy = 0.5*(1.0-nu_s)

	D = as_matrix([ [1.0, nu_s, 0.0, 0.0, 0.0, 0.0],
			[nu_s, 1.0, 0.0, 0.0, 0.0, 0.0],
			[0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0,  0.0, 0.0, ttxy,0.0, 0.0],
			[0.0,  0.0, 0.0, 0.0, ttzz,0.0],
			[0.0,  0.0, 0.0, 0.0, 0.0,ttzz]])
	D = D*D_coef

	stress = dot(D,epsilon_s(u))

	return stress



#Deslocamento explicito #Newmark equation (2.3)
def disp(u_n, v, v_n, v_2n, dt):
	dd1 = dt*v_n
	dd2 = ((dt*dt)/2.0)*(((1.0 - 2.0*beta)*((v_n-v_2n)/dt)) + 2.0*beta*(v-v_n)/dt)
	return u_n + dd1 + dd2


##### Inicialization ########

#Initialization of deformable problem from the rigid one

vr, pr = Function(W.sub(0).collapse()), Function(W.sub(1).collapse())


#Read the solution of a rigid problem previously solved
#timeseries_v = TimeSeries('velocity_series_1')
#timeseries_p = TimeSeries('pressure_series_1')


#BC modified to solve contour variational equation
bc1 = DirichletBC(A, vwalls, parts, 1)
bc2 = DirichletBC(A, vwalls, parts, 2)
bc3 = DirichletBC(A, vwalls, parts, 3)
bc_pressure = [bc1, bc2, bc3]

#Functions to solve the pressure traction problem
w1 = TestFunction(A)
u1 = TrialFunction(A)
us = Function(A)


#FSI_equation: adding the solid equation to the stabilized fluid equation
F_ = F_est + qsi*(rho_s/dt)*inner(v-v_n,w)*ds(4) + qsi*inner(sigma_s(disp(u_n,v,v_n,v_2n,dt)),epsilon_s(w))*ds(4)


#solution parameters
tolr = 'relative_tolerance'
tola = 'absolute_tolerance'
ln = 'linear_solver'
ns = 'newton_solver'
prec = 'preconditioner'
ks = 'krylov_solver'
mi = 'maximum_iterations'
enon = 'error_on_nonconvergence'

linear = {tolr: 1.0E-3, tola: 1.0E-3, mi: 2000, enon: False}
nonlinear = {tolr: 1.0E-2, tola: 1.0E-2, ln:'gmres', mi:5, ln:'gmres', prec:'ilu', enon: False, ks: linear}
par = {ns: nonlinear}


# Create files for storing solution
vRfile = File("CMM_teste"+str(teste)+"/vR"+str(n_cells)+".pvd")
pRfile = File("CMM_teste"+str(teste)+"/pR"+str(n_cells)+".pvd")
vCfile = File("CMM_teste"+str(teste)+"/vC"+str(n_cells)+".pvd")
pCfile = File("CMM_teste"+str(teste)+"/pC"+str(n_cells)+".pvd")
ufile = File("CMM_teste"+str(teste)+"/u"+str(n_cells)+".pvd")

xdmffile_v1 = XDMFFile('vR.xdmf')
xdmffile_p1 = XDMFFile('pR.xdmf')
xdmffile_v_ = XDMFFile('vC.xdmf')
xdmffile_p_ = XDMFFile('pC.xdmf')
xdmffile_u_ = XDMFFile('uC.xdmf')


# define Jacobian
F1 = action(F_est,vp1)
J = derivative(F1, vp1,vp)		#Jacobian of fluid problem
F2 = action(F_, vp_)
J_ = derivative(F2, vp_, vp)		#Jacobian of coupled problem


# Time-stepping
t = 0
i = 0

while t < t_end:
	vin.t = t
	print ("t = ", t)

	# Compute Fluid problem (rigid case)
	print('\n\nSolving Fluid Problem')
	solve(F1 == 0, vp1, bcs=bcs_rigid, J=J, solver_parameters = par, form_compiler_parameters = {"quadrature_degree": 3})
	# Extract solutions:
	(v1, p1) = vp1.split(True)
	vRfile << v1
	pRfile << p1
	xdmffile_v1.write(v1, t)
	xdmffile_p1.write(p1, t)

	#Fluid next step
	vn.assign(v1)

	# Coupled problem
	v_2n.assign(v1)
	v_n.assign(v1)
	p_n.assign(p1)
	print('\n\nSolving Coupled Problem')
	solve(F2 == 0, vp_, bcs=bcs_coupled, J=J_, solver_parameters = par, form_compiler_parameters = {"quadrature_degree": 3})
	(v_,p_) = vp_.split(True)
	
	#Displacement
	dsp = disp(u_n,v_,v_n,v_2n,dt)
	u_ = project(dsp, A)

	pavg = (assemble(p1*ds(4)))/(assemble(Constant(1.0)*ds(4)))
	Fhp = 1E-5*inner(nabla_grad(u1),nabla_grad(w1))*dx - inner(pavg*n,w1)*ds(4) + qsi*inner(sigma_s(u1),epsilon_s(w1))*ds(4)

	ahp, Lhp = system(Fhp)
	Ahp, bhp = assemble(ahp), assemble(Lhp)
	[bc.apply(Ahp, bhp) for bc in bc_pressure]
	solve(Ahp, us.vector(), bhp, 'gmres', 'ilu')
	
	
	print('\n\ndeslocamento max: ', np.mean(u_.vector()))

	#Save to file
	vCfile << (v_,t)
	pCfile << (p_,t)
	ufile << (us,t)
	
	xdmffile_v1.write(v1, t)
	xdmffile_p1.write(p1, t)

	# Plot
	#plot(v)
	# Save to file
	#xdmffile_u.write(u1, t)
	#xdmffile_p.write(p1, t)
	#timeseries_u.store(u1.vector(), t)
	#timeseries_p.store(p1.vector(), t)
	#if (i%2 == 0):
	#	ufile << u1
	#	pfile << p1
	# Move to next time step
	v_2n.assign(v_n)
	v_n.assign(v_)
	p_n.assign(p_)
	u_n.assign(u_)
	t += dt
	i = i+1
