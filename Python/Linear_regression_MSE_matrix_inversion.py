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


def regression_inversion_matricielle(X:np.ndarray, Y:np.ndarray, d:int=1):
    """Performs polynomial regression by projecting Y vector onto the regression subs-space.
        Solves the system by inverting it

    Args:
        X (np.ndarray): n samples of real values
        Y (np.ndarray): n samples of real values
        d (int, optional): Fitting polynome degree. Defaults to 1.
    """
    A = np.array([[xi**(d-i) for i in range(d+1)] for xi in X])

    A_T = A.T

    w = np.linalg.inv(A_T @ A) @ A_T @ Y

    #convention : [a0, ..., ad]
    w = np.array([w[len(w)-1-i] for i in range (len(w))], dtype=float)

    return(w)



########
# MAIN #
########
if __name__ == "__main__":	
    # np.random.seed(0)
    ### SAMPLES ###
    # Polynome coefficients 
    coeff_th = [1.0, -3.2, 0.5, 7.8, -12.4, 4.1, 9.6, -15.3, 6.7, 2.9, -5.4, 1.2]

    nb = 100 #samples
    X = np.arange(0, nb)


    # convention : de a0 à ... ad
    d = len(coeff_th)-1 # degree of polynome


    # Definition of Y_th
    Y_th = np.zeros_like(X, dtype=float)
    for i in range(d + 1):
        Y_th += coeff_th[i] * (X ** i) #ad.x^d + ... + a0.x^0


    # Then add noise to get Y
    np.random.seed(0)  # Pour la reproductibilité
    bruit = np.random.normal(0, 0.5, len(X))  # Bruit gaussien de moyenne 0 et écart-type 0.5
    Y = Y_th + bruit

    ### REGRESSION ###
    coeff = regression_inversion_matricielle(X, Y, d)
    # convention : de a0 à ... ad

    Y_estimate = np.zeros_like(X, dtype=float)
    for i in range(d + 1):
        Y_estimate += coeff[i] * (X ** i)

    SS_mean = np.sum((Y-np.mean(Y))**2)
    SS_fit = np.sum((Y-Y_estimate)**2)
    R_squared = (SS_mean - SS_fit)/SS_mean


    print(f"coeff_th = {coeff_th}")
    print(f"coeff_ = {coeff}")


    ### PLOT ###
    f1, ax = plt.subplots()
    ax.plot(X, Y, '*', label = "Signal bruité")
    ax.plot(X, Y_th, 'r--', label = "Signal théorique")
    ax.plot(X, Y_estimate, label = f"Regression polynomiale")
    ax.set_title(f"Regression polynomiale de degrès {d}")
    ax.text(1 , 1, s=f"Regression polynomiale\nr² = {R_squared}")
    plt.legend()
    plt.grid()
    # plt.xlim(-20, 20)
    # plt.ylim(-5, 15)
    plt.show()


        

    




