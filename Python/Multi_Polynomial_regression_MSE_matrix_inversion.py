# C. Cho 
##############
# LIBRAIRIES #
##############
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from copy import deepcopy


###########
# PROBLEM #
###########
"""
    Let (X, Y) be two vectors in R^n x R^n.
    The coordinates (xi, yi) of X and Y form coordinate couples
    which designate points on a graph.
	We want to fit our points with coefficients (a_i) in R^(d+1) to obtain a polynom of degree d,
    such as Y ~ sum_{i=0}^{d} a_i*X**i
    

    In practice we will always have an error vector eps such that
    Y = sum{a_i*X**i} + eps

    example d = 1 : 
    Y ~ a_1*X + a_0 
    fit with affine function of coefficients (a, b)

    example d = 2 :
    Y ~ a_2*X² +a_1*X + a_0
    fit with square function of coefficients (a, b, c)
    
    The idea is to approximate Y with Ŷ such as Ŷ = a_i*X**i and eps = Y - Ŷ, 0 <= i <= d
    In order to do this, one has to find the best coefficients (a_i), 0 <= i <= d

    Ŷ = a_d*X^d+ a_(d-1)*X^(d-1) +... + a_(1)*X + a_(0) = (ŷ_n, ...., ŷ_1) with X = (x_n, ...., x_1)
    for all j in [| 1 ; n |]: 
           ŷ_j = a_d*x_j^d +  a_(d-1)*x_j^(d-1) + ... + a_1*x_j

    Thus, we can write matrix equation :

        [[ x1**d ...   x1**0 ],
    Ŷ =  [ ..    ...   ...   ],  *  (a_d, ..., a_1)^T
         [ xn**d ...   xn**0 ]]

    Let's define w = (a_d, ..., a_1)^T dans M(d,1)(R) 
                 A = [[ x1**d ...   1 ], [ ..    ...     ],[ xn**d ...   1 ]] dans M(n,d)R

    We obtain Ŷ = A*w (E)

    In the best case we have eps = (0, .., 0)
    Meaning Ŷ = Y. Therefore all points are alligned on a slope.

   But this is rarely the cas, let's discuss on equation (E) a bit

    One want to obtain w = argmin(||eps||) = argmin(||Y - Ŷ||)

    Let's consider col(A) = span space from A columns
    Remark :
    M = (C1, .., Cp) matrix of p columns 
    Let be u = (u1, ..., up) a vector of p coefficients
    => We can write M * u = u1 * C1 + u2 * C2 + ... + up * Cp

    Let's call columns of A (A1, A2, ..., Ad)
    Let be col(A) = {vectors t = (t1, .., tn)^T such that it exists v = (v1, ..., vd)^T t = A*v = v1*A1 +...+ vd*Ad}
    Another way to say this is:, col(A) is the span-space of A columns.

    Being given a vector w, A*w belongs to col(A)

    Discussion on eps = Y - Ŷ :
    (i) If eps is null vector then Y belongs to col(A)

    (ii) If not it is outside of col(A)

    Solution Ŷ = A*w that minimise eps is Ŷ in col(A) that's closest to Y.
    Nevertheless minimal distance between a vector and a space is the projection 
    of the vector on this space

    Therefore we want w such that Ŷ = A*w is the projection of Y on col(A)

    Remark:
    For a given vectorial space E we write E⊥ the set of vectors that are orthogonal to E.
    We can demonstrate that in R^n we can write a vector x
    as a sum of vector u and v, one belonging to E and the other to E⊥:
    Such that x = u + v, u in E and v in E⊥

    Therefore one can write Y = u + v with u in à col(A) and v in col(A)⊥.
    Here we can specifically write : u = Ŷ and v = eps.

    One can demonstrate that  :
    given the span-space of A columns col(A),
    it's orthogonal space col(A)⊥,
    and a vector x :
    if x belongs to ker(A^T) then belongs to col(A)
    and symmetrically col(A) then belongs to ker(A^T)
    Therefore for x in col(A)⊥ then A^T*x = 0.

    Let's apply this : Y = Ŷ + eps
    eps belongs to col(A)⊥ => A^T*eps = 0
    But A^T*eps = A^T(Y - Ŷ) = A^T*Y - A^T*Ŷ = A^T*Y - A^T*A*w = 0
    Therefore : ( A^T*A )* w = A^T*Y => w = (A^T*A)^(-1) * A^T*Y
    
"""

#############
# FUNCTIONS #
#############
def graham_schmidt(B:np.ndarray)->np.ndarray:
    """Return an orthogonal basis from a given B basis

    Args:
        B (np.ndarray): B basis
    """
   
    B_ortho = []
    for i in range(len(B)):
        e_i = B[i]
        for j in range (len(B_ortho)):
            e_j = B_ortho[j]
            p_j = np.linalg.vecdot(e_i, e_j)*e_j/np.linalg.vecdot(e_j, e_j)
            e_i = e_i - p_j
        B_ortho.append(e_i)
    return(B_ortho)


def regression_inversion_matricielle_2D_vectorisée(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, d_x:int, d_y:int):
    """Performs mutli-polynomial(X, Y) regression by projecting Y vector onto the regression subs-space.
        Solves the system by inverting it

    Args:
        X (np.ndarray): n samples of real values
        Y (np.ndarray): m samples of real values
        Z (np.ndarray): n.m matrix lattice of points to fit
        d_x (int): Fitting polynome degree on X.
        d_y (int): Fitting polynome degree on Y.
    """


    # Maths notation :
    # Let be i from  1 to n, j from  1 to m.
    # i := line ; j := column
    # Zij = a*Xj + b + c*Yi + d  nxm matrix
    # with n : lines number and m : columns number 
    # Python notation : 
    # if X = [0, 1] and Y = [0, 1, 2]
    # Z is a numpy.array of Z.shape = 3x2 


    Z_vec = np.ravel(Z.T) #Z_vec = (Z00, ..., Z0m, Z10, ...Z1m ... Zn0... Znm)
                
   
    # d_x+d_y-1 car because 0 degree coefficients a0*x^0 et b0*y^0 are undiscernable : x^0 = y^0 
    n_coeffs = (d_x + 1) + d_y  # a0, a1, ..., a_{d_x}, b1, ..., b_{d_y}
    A = np.zeros(shape=(len(X)*len(Y), n_coeffs))  # +1 for constant term


    # Columns X^0, X^1, ..., X^{d_x}
    for d in range(0, d_x):
        col_index = d
        # print(col_index)
        A[:, col_index] = np.ravel([[x**(d+1) for _ in range(len(Y))] for x in X])
        # print(A)


    # Columns for Y^1, ..., Y^{d_y} (not including Y^0 because we counted it in X^0 coefficient)
    for d in range(0, d_y):
        col_index = d_x + d
        A[:, col_index] = np.ravel([[y**(d+1) for y in Y] for _ in X])
        # print(col_index)
        # print(A)
    
    #print("colonne neutre")
    A[:, -1] = 1

    print(A)

    A_T = A.T

    # print(A.shape)
    # print(A_T.shape)
    # print((A_T@A).shape)
    # print((np.linalg.inv(A_T @ A)@A_T).shape)
    # print(Z_vec.shape)

    w = np.linalg.inv(A_T @ A) @ A_T @ Z_vec

    #convention : [a0, ..., a_dx, b0, ...., b_dy]

    return(w)



########
# MAIN #
########
if __name__ == "__main__":	
    # np.random.seed(0) #Repro

    #### DEMO coefficients : X² Y³ ####
    # Demonstration on (^2, ^3) order (X,Y) multi-polynome
    classic_demo = False
    if classic_demo :
        # Liste des coefficients du polynôme
        coeff_th_X = [1, 0.2, -0.04]

        # Liste des coefficients du polynôme
        coeff_th_Y = [0.0, 0.23, 0.1, 0.05]


    #### DEMO coefficients : ELLIPSE ####
    # Using previous results we can predict fitting functions that derives
    # from (X,Y)multi-polynome such as an ellipsoïd
    # Let us consider the following equation T² + X² + Y² = R
    # We consider and solve forZ = R - X² - Y² = T²
    # Then we solve Z = T^0.5
    ellipsis = True
    if ellipsis :
        R = 1
        a = 3
        b = 2 

        # List of X coefficients for polynome
        coeff_th_X = [R, 0.0, -1.0/a]
        
        # List of Y coefficients for polynome
        coeff_th_Y = [0.0, 0.0, -1.0/b]

    ### DEMO coefficients : GAUSIAN ###
    # Using previous results we can predict fitting functions that derives
    # from (X,Y)multi-polynome such as an ellipsoïd
    # Let us consider T(X, Y) = e⁻(X² +Y²)
    # Let Z = ln(T) => Z = ln(T) = -X² -Y²
    # We solve Z = -X² - Y²
    # And plot T = exp(Z)
    gaussian_demo = False
    if gaussian_demo :
        # Liste des coefficients du polynôme
        coeff_th_X = [0.0, -5.0, -1.3*1E1]
        
        # Liste des coefficients du polynôme
        coeff_th_Y = [0.0, -3.0, -1.1*1E1]


    # Polynome degree
    # convention : a0 to ... ad
    d_x = len(coeff_th_X)-1 # degree of X-polynome
    # convention : b0  ... bd
    d_y = len(coeff_th_Y)-1 # degree of Y-polynome

    print(f"degree of X : {d_x}")
    print(f"degree of Y : {d_y}")

    ### SAMPLES ###

    nb_X = 100 #samples
    X = np.arange(0, nb_X)
    nb_Y = 100 #samples
    Y = np.arange(0, nb_Y)

    print(f" X = {X}")
    print(f" Y = {Y}")
    mg_X, mg_Y = np.meshgrid(X, Y) #meshgrid
    Z = np.zeros(shape=mg_X.shape)


    equation_str = "" #String to print the equation in CLI


    for i_x in range(d_x+1):
                Z +=  coeff_th_X[i_x] * mg_X**i_x #ad.x^d + ... + a0
                equation_str += f" +{coeff_th_X[i_x]}*x^{i_x}"
    for i_y in range(d_y+1):
                Z +=  coeff_th_Y[i_y] * mg_Y**i_y #bd.y^d + ... + b0
                equation_str += f" +{coeff_th_Y[i_y]}*y^{i_y}"
    
    print("Equation = ",equation_str)


    # Add noise to get Y
    Bruit = np.random.normal(0, 0.1, size=Z.shape) # Gaussian noise :mean 0 et std 0.1
    Z = Z + Bruit #noise added

    ### REGRESSION ###
    coeff = regression_inversion_matricielle_2D_vectorisée(X, Y, Z, d_x, d_y)

    print(coeff)
    print(coeff_th_X, coeff_th_Y)
    # convention : de a0 à ... ad


    Z_estimate = np.zeros(shape=mg_X.shape)
    # Z_estimate = coeff[0]*mg_X + coeff[1]*mg_Y + coeff[2]

    for i_x in range(d_x):
                d = i_x + 1
                Z_estimate +=  coeff[i_x] * mg_X**d #ad.x^d + ... + a0
    for i_y in range(d_y):
                d = i_y + 1
                i = d_x + i_y 
                Z_estimate +=  coeff[i] * mg_Y**d #bd.y^d + ... + b0
    
    Z_estimate +=  coeff[-1]

    if ellipsis :
        # ellipsis :
        Z = np.sqrt(Z+R)
        Z_estimate = np.sqrt(Z_estimate+R)
        Z_ = -Z
        Z_estimate_ = -Z_estimate

    if gaussian_demo :
        # gaussian : T = exp(Z) :
        Z = np.exp(Z)
        Z_estimate = np.exp(Z_estimate)


	### R**2 PLOT ###
    fig, ax_3d = plt.subplots(subplot_kw={"projection": "3d"})

  
    ax_3d.scatter(mg_X, mg_Y, Z, marker = '.', color='red', alpha=0.1)
    ax_3d.plot_wireframe(mg_X, mg_Y, Z_estimate, linewidth=0.5)
    if ellipsis :
        # negative ellipsis
        ax_3d.scatter(mg_X, mg_Y, Z_, marker = '.', color='red', alpha=0.1)
        ax_3d.plot_wireframe(mg_X, mg_Y, Z_estimate_, linewidth=0.5)


    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_title(f"z")
    ax_3d.set_aspect('equal')
	
    plt.show()
        

    




