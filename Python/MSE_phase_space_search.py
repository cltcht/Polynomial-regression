# C. Cho 
##############
# LIBRAIRIES #
##############
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from copy import deepcopy

#############
# FUNCTIONS #
#############

def plot_regression(f1:plt.figure , ax:plt.axis , X: np.ndarray , Y: np.ndarray, Y_reg: np.ndarray, label='', color = 'r', alpha = 1):
	"""Plot (X, Y, Y_reg) on a same (fig, ax) plot.

	Args:
		f1 (plt.figure): Matplotlib figure
		ax (plt.axis): Matplotlib figure
		X (np.ndarray): n samples of real values
		Y (np.ndarray): n samples of real values
		Y_reg (np.ndarray): n samples of linear regression of (X,Y)
		label (str, optional): Label for figure _description_. Defaults to 'r'.
		alpha (int, optional): Transparency of points. Defaults to 1.
	"""
	
	ax.plot(X, Y, '*', label = 'Y noised')
	ax.plot(X, Y_reg, color+'--', label = 'Y_reg', alpha=alpha)
	for (x_couple, y_couple) in zip(np.array([[x, x] for x in X],), [[y,yt] for (y,yt) in zip(Y,Y_reg)]):
		plt.plot(x_couple, y_couple, 'g--')
	if label != '':
		f1.legend()
	else:
		ax.set_title("Reg. Linéaire : "+label)
	plt.grid()

	
########
# MAIN #
########
if __name__ == "__main__":

	### SAMPLES ###
	np.random.seed(0)
	
	nb = 100 #samples
	X = np.arange(0, nb)
	a_th, b_th = 2, 1
	Y = 20*(np.random.rand(nb)-0.5) + a_th*X + b_th
	Y_mean = np.mean(Y)
	SS_mean = np.sum((Y- Y_mean)**2, axis = 0)

	### LATTICE ###
	# Limits for lattice
	amin, amax = -10, 10
	bmin, bmax = -10, 10
	dx, dy = 0.1, 0.1 #A and B resolution

	#Definition of lattice
	a_range = np.arange(amin, amax, dx)
	b_range = np.arange(bmin, bmax, dy)
	A, B = np.meshgrid(a_range, b_range) #meshgrid : lattice

	### Search for best R**2,  SS_fit ###
	SS_mean = np.sum((Y- np.mean(Y))**2, axis = 0)
	R_2 = np.zeros(shape=A.shape)
	SS_fit = np.zeros(shape=A.shape)
	for (i) in range(len(a_range)):
		for (j) in range(len(b_range)) :
			a, b = A[i, j], B[i, j]
			Y_fit = a*X+b
			SS_fit_coeff = np.sum((Y - Y_fit )**2, axis=0)
			r_2_coeff = (SS_mean- SS_fit_coeff)/SS_mean
			R_2[i, j] = r_2_coeff
			SS_fit[i, j] = SS_fit_coeff
		

	# Get the index of the maximum value for r**2
	max_flat_index = R_2.argmax()
	# Get the index of the maximum value for SS_fit
	min_flat_index_ss_fit = SS_fit.argmin()

	# Convert the index to a tuple of coordinates (row, column) for r**2
	max_indices = np.unravel_index(max_flat_index, R_2.shape)

	# Convert the index to a tuple of coordinates (row, column) for SS_fit
	min_indices = np.unravel_index(min_flat_index_ss_fit, R_2.shape)

	print(f"\n(a, b) = {float(A[max_indices[0], max_indices[1]]), \
				float(B[max_indices[0], max_indices[1]])}\n")


	### R**2 PLOT ###
	fig, ax_3d = plt.subplots(subplot_kw={"projection": "3d"})
	Z = R_2

	ax_3d.scatter(A[max_indices[0], max_indices[1]], 
			B[max_indices[0], max_indices[1]], 
			R_2[max_indices[0], max_indices[1]],marker='^', color='red', s=80)

	#ax.plot_wireframe(A, B, Z, color='C0', linewidth=0.5, alpha=0.7)
	ax_3d.plot_surface(A, B, Z,  vmin=Z.min()*0.9, vmax=Z.max(), cmap='tab20', cstride=2, rstride=2)
	

	ax_3d.set_xlabel("a")
	ax_3d.set_ylabel("b")
	label = f"{R_2[max_indices[0], max_indices[1]]} \n \
			   (a, b) = {float(A[max_indices[0], max_indices[1]]), \
				float(B[max_indices[0], max_indices[1]])}"
	ax_3d.set_title(f"r² max = {label}")

	ax_3d.set(xlim=(amin, amax), ylim=(bmin, bmax))

	### SS_FIT PLOT ###
	fig_ssf, ax_3d_ssf = plt.subplots(subplot_kw={"projection": "3d"})
	Z = SS_fit
	Z_diff = np.gradient(Z, dx, dy)
	# norme du gradient
	Z_diff_norm = np.sqrt(Z_diff[0]**2 + Z_diff[1]**2)

	# Get the index of the maximum value for SS_fit
	min_flat_index_grad = Z_diff_norm.argmin()
	min_indices_grad = np.unravel_index(min_flat_index_grad, Z_diff_norm.shape)
	print(f"Minimum trouvé en (a, b) = {A[min_indices_grad[0], min_indices_grad[1]], \
			B[min_indices_grad[0], min_indices_grad[1]]}")

	ax_3d_ssf.scatter(A[min_indices[0], min_indices[1]], 
			B[min_indices[0], min_indices[1]], 
			SS_fit[min_indices[0], min_indices[1]],marker='^', color='red', s=80)

	#ax.plot_wireframe(A, B, Z, color='C0', linewidth=0.5, alpha=0.7)
	#ax_3d_ssf.plot_surface(A, B, Z,  vmin=Z.min()*0.9, vmax=Z.max(), cmap='tab20', cstride=2, rstride=2)
	ax_3d_ssf.plot_surface(A, B, Z_diff_norm,  vmin=Z.min()*0.9, vmax=Z.max(), cmap='tab20b', cstride=2, rstride=2)
	

	ax_3d_ssf.set_xlabel("a")
	ax_3d_ssf.set_ylabel("b")
	label = f"{SS_fit[min_indices[0], min_indices[1]]} \n \
			   (a, b) = {float(A[min_indices[0], min_indices[1]]), \
				float(B[min_indices[0], min_indices[1]])}"
	ax_3d_ssf.set_title(f"ss_fit min = {label}")

	ax_3d_ssf.set(xlim=(amin, amax), ylim=(bmin, bmax))


	f1, ax_reg = plt.subplots()
	# 10 best fits ?
	N = 10 
	R_2_copy = deepcopy(R_2)
	indexes = []
	for i in range (N):
		# Index : argmax
		max_flat_index = R_2_copy.argmax()

		# index -> (row, column) in the matrix
		max_indices = np.unravel_index(max_flat_index, R_2_copy.shape)

		R_2_copy[max_indices[0], max_indices[1]] = 0
		indexes.append([max_indices[0], max_indices[1]])


	# Best fit (a,b)
	label = f"Fit (a, b) = {max_indices[0], max_indices[1]}"
	plot_regression(f1, ax_reg, X, Y, A[max_indices[0], max_indices[1]]*X + B[max_indices[0], max_indices[1]] ,\
				    alpha=0.7, color = 'r', label=label)
	plt.show()


