# Polynomial-regression
Few scripts using different programming languages and approaches to tackle fitting problems.

##### Workflows :  
![CI CPP](https://github.com/cltcht/Polynomial-regression/actions/workflows/ci_cpp.yml/badge.svg)  
![CI Python](https://github.com/cltcht/Polynomial-regression/actions/workflows/ci_python.yml/badge.svg)  
  


## Summary
As personal project I wanted to play with one "simple" mathematical problem along with differents programming languages.
I choosed polynomial regression in order to fit a set of points.

I've seen online some ML based approaches which I don't like.
It felt like "using a bulldozer to crack an egg" when one can find linear-algebra based solution which at least to me (AND in the case of "easy" fitting problems, eg. low dimension, low data volume) feels more elegant.

I begun with Python, see `/Python` repo for scripts and `readme`§**B)**.
Then I switched to C++, see `/Cpp` repo for scripts and `readme` §**C)**. 

Graphic engine functions are inspired from [kavan010 git](https://github.com/kavan010) for C++. 

For reminders and proofs of algebra theorems used, please see [this website](https://textbooks.math.gatech.edu/ila/1553/index2.html).

------------------------------

**I might push some script using Rust later** 

----------------------------------

## A) Discusion on regression

In this paragraph I will discuss different approaches for regression problems to put all the mathematics in one place.
All the mathematical approaches are linked to scripts, see paragraph **B)** to run them.

### 1 - 2D Linear regression "naive approach" : Least Square maxima search in bounded phase space  

**Note : At first I decided to go from scratch and have zero look online and just give it a try in the morning with what I thought overnight. So this solution isn't optimal at all.
In short the idea is to scan parameters subspace to find the best set of parameters.**

Let $n \in \mathbb{N}$, $(X, Y) \in \mathbb{R}^{n}\times\mathbb{R}^{n}$ be two set of $n$ real values.

```math
\text{Rewrite : } X=(X_{k})_{1 \leq k \leq n} \text{ and }  Y=(Y_{k})_{1 \leq k \leq n}
```

Objective is to find $(a, b) \in \mathbb{R}\times\mathbb{R}$ such as a linear approximation $\hat{Y}$ is as close as possible of $Y$ :
```math
\hat{Y}(a, b, X) = aX + b \times (1, ..., 1)^T
```

Let's write the mean ${\sum_{k=1}^{n} Y_{k} }\over{n}$ as : $\bar{Y}$ .  

One usually maximises $R^2$ defined as ${SS_{mean} - SS_{fit}}\over{SS_{mean}}$ as fitting metrics with :

```math
{SS_{mean}} = {\sum_{k=1}^{n} {({Y}_k - \bar{Y})}^2}
```
```math
{SS_{fit}} = {\sum_{k=1}^{n} {(\hat{Y}_k - \bar{Y})}^2}
```

Realistically, one just want to minimise ${SS_{fit}}$. Therefore our goal is to find a vector $(\hat{a}, \hat{b}) = argmin({SS_{fit}}(a, b, X))$.

Hypothesis, In order not to compute indefinitely :
* One can set limits to the vector space that contains $(a, b)$. Let's call this bounded space $A \times B$
* One can define a lattice subspace of $A \times B$ with definite resolution over $A$ and $B$.
Meaning we will search for all existing $(a, b)$ over a definite lattice subspace of $A \times B$.

Limits of this approach :
* Global maxima might be out of lattice subspace of $A \times B$, we have to "guess" good limits to search in phase space
* Great chance that lattice resolutions over A and B are set such as best $(\hat{a}, \hat{b})$ is not on one point of the lattice.

**Conclusion**
Therefore this "overnight thought" method isn't satisfying : 
If the best parameters exist in the limited subspace, there's a great chance than one can only have an approximation of it. 
For the next approach let's introduce a bit of linear-algebra !


### 2 - 2D Linear regression : Vector projection over linear span space

>**Reminder** $\hat{Y}(a, b, X) = aX + b \times (1, ..., 1)^T$.

Which we can rewrite by vector-matrix multiplication :

```math
\hat{Y}(a, b, X) = aX + b \times (1, ..., 1)^T = A \times w
```

```math
A = \begin{bmatrix}
  X_1 & 1 \\
  X_2 & 1\\
  \vdots \\
  X_n & 1
\end{bmatrix}, \in M_{(n,2)} ({\mathbb{R}}),
```

```math
w = \begin{bmatrix}
  a \\
  b
\end{bmatrix},\in \mathbb{R}^{2}
```
>**Reminder**: $A$ is a nx2 matrix.

Let's write $(A_{i})_{1 \leq i \leq 2}$ the columns of A. Here, $A_1 = X and A_2 = (1, ..., 1)^T$ .
One can define $col(A)$ a subspace of ${\mathbb{R}^n}$ as the set of all finite linear combinations of elements $A$ : $(A1, A2)$ .

Let be a vector $v \in col(A)$, therefore $\exists (t_1, t_2) \in {\mathbb{R}^2} | v = {t_1 \times A_1} + {t_2 \times A_2}$ .

>**Remark** : $\hat{Y}$ can be written as $A \times w \implies \hat{Y} \in col(A)$ .

Now let's write $E \in \mathbb{R}^{n}$ error vector such as $E = Y - \hat{Y}$.

**Discussion about the error vector** 

Two cases exist : *either $E$ is the null vector or not.*

1) *Supposing* $E$ *is the null vector*:

Therefore we have $Y = \hat{Y} = A \times w \in col(A)$.
Meaning all our points $(X_i ; Y_i), i \leq n$ are perfectly alligned on a slope.

It also means $Y$ is also located in col(A).
But generally it isn't the case : $(X, Y)$ might be some measurments with it physical effects (noise, accuracy, precision).


2)  *Supposing* $E$ *isn't the null vector*:

$Y$ is a vector in ${\mathbb{R}^n}$ out of sub-space $col(A)$ and $\hat{Y}$ is in $col(A)$.
The best fit $\hat{Y}$ we can have will be the one minimizing $E$ i.e. one want $E$ as short as possible.

>**Remark** : The shortest path between one point and a space is the orthogonal projection.

Therefore we can infer that $E$ is orthogonal to $col(A)$.

Furthermore, 
```math
E \perp col(A) \Leftrightarrow E \in  col(A)^{\perp} \Leftrightarrow E \in  ker(A^{T})
```
```math
\text{Therefore : } A^{T} \times E = 0_{{\mathbb{R}^2}} \implies A^{T} \times Y = A^{T} \times \hat{Y}
```
```math
  \implies A^{T} \times A \times w = A^{T} \times Y 
```
```math
\implies w = (A^{T} A)^{-1} \times A^{T} \times Y
```

**But ... Projection and Graham-Schmidt algorithm !**

Previous result is also valid for polynomial regression (not only linear) if you add more columns to $A$, see next paragraphs.
Before implementing the previous result, I would like to study further this linear case by using another solution.

One might use the multiplication of $(A^{T} A)^{-1} \times A^{T}$ to find the best parameters.

But as we said, it's all about orthogonal projection of $Y$ over $col(A)$.
Let's discuss a bit about projection and basis orthogonality.

Suppose we got a vectorial space $V$ of finite dimension $m$ .
Let $S$ be a subspace of $V$ of finite dimension $n$ and $(e_1, ..., e_n)$ a orthonormed basis of $S$ ,
Let $x$ be a vector of $V$ ,
Let be $p \in S$ the projection of $X$ on $S$ :
```math
p = \sum_{k=1}^{n} {{< x ; e_i>}e_i} \text{,   (< . ; . > being the scalar product)}
```
Now if we have a basis that's not orthonormal one can use the Graham-Schmidt algorithm to generate one.



>**Remark** : The Graham-Schmidt algorithm behavior generate a basis $(u_1, ... ,u_p)$ from a basis $(a_1, ..., a_p) \in {\mathbb{R}>^n}$
>Let's normalise the generated basis by $(q_1, ..., q_p) = (u_1, ..., u_p)$ ,  $q_j$ =  ${ u_j \over \Vert u_j \Vert }$ , $1 \leq j \leq p $  
>
>Let Q be a matrix whose column are the vectors of the generated basis : $Q = [q_1, ..., q_p]$  
>Let write every vector $(a_1, ..., a_p)$ in the $(q_1, ..., q_p)$ basis with $r_{ij}$ coefficients :  
>For every $1 \leq j \leq p$ : $a_{j}$ = $\sum_{i=1}^{p} {{r_{ij}}.q_{i}}$ ,  
>With :$r_{ij}$ = $< q_{i} ; a_{j} >$  

But as every vector $u_j$ is a linear composition of $(a1, ..., a_j)$ : $(a_{j+1}, ..., a_p)$ aren't involved.  
Therefore, $j \lt i \implies r_{ij}  = 0$ such as matrix $R = (r_{ij})_{1 \leq i,j \leq p}$ is triangular.  
We can therefore write $A = Q.R$  
  
Back to our regression problem, generalizing fitting with a $p-degree$ polynom with $n$-points vectors $(\hat{Y}, Y, X)$:
```math
\hat{Y} = Q.R.w \implies Q^T.\hat{Y} = R.w \text{ as by orthogonality } Q^T.Q = I_{d}.
```
Therefore we obtain an "easy to solve" triangular system $Q^T.\hat{Y} = R.w$ :  

```math
\begin{bmatrix}
  r_{11} & r_{12} & r_{13} & ... & r_{1p} \\
  0 & r_{22} & r_{23} & ... & r_{2p} \\
 0 & 0 & r_{33}... & r_{2p} \\
  \vdots & &  & \vdots \\
  0 & 0 & ... & r_{pp}
\end{bmatrix} \times
\begin{bmatrix}
w{1} \\
\vdots \\
w{p}
\end{bmatrix} =
\begin{bmatrix}
\rule[.5ex]{2.5ex}{0.5pt} & q_1^T & \rule[.5ex]{2.5ex}{0.5pt} \\
 & \vdots &  \\
\rule[.5ex]{2.5ex}{0.5pt} & q_p^T & \rule[.5ex]{2.5ex}{0.5pt}
\end{bmatrix} \times
\hat{Y}
```  

In the end you can apply this pseudo-code algorithm to retrieve $w$ vector :  

----
*For j from p to 1 with step = -1 :*
```math
w_{i} = {{< \hat{Y} ; q_i> - \sum_{j = i+1}^{p}{r_{ij}.w_{j}}} \over {r_ii}}
```
----  

**Now** let's go back to our simple linear (aka degre one polynomial) $\hat{Y} = aX + b(1, ...,1)^T$:  
The columns of $A$ form a basis *B*$=(A_1, A_2)$ of $col(A)$.  
Using the Graham-shmidt algorithm we can obtain an orthogonal basis (which we norm) $B_{GS}=GS(A_1, A_2)=(q_1, q_2)$ for $col(A)$.
We can define $Q$ and $R$ matrix as explained before and write :  
```math
\begin{bmatrix}
  r_{11} & r_{12} \\
  0 & r_{22} 
\end{bmatrix} \times
\begin{bmatrix}
a \\
b
\end{bmatrix} =
\begin{bmatrix}
\rule[.5ex]{2.5ex}{0.5pt} & q_1^T & \rule[.5ex]{2.5ex}{0.5pt} \\
\rule[.5ex]{2.5ex}{0.5pt} & q_2^T & \rule[.5ex]{2.5ex}{0.5pt}  
\end{bmatrix} \times
\hat{Y}
```  


### 3 - Polynomial regression : Matrix inversion method

So, in **§2** we played a bit with Graham-Schmidt algorithm. But the easiest to get the $w$ vector is definitelly to use matrix inversion method (also discussed in **§A)2**).

Let's do that !  

**Note** :  
For the moment we defined for linear fit the following :  
```math
A = \begin{bmatrix}
  X_1 & 1 \\
  X_2 & 1\\
  \vdots \\
  X_n & 1
\end{bmatrix}, \in M_{(n,2)} ({\mathbb{R}}),
```

```math
w = \begin{bmatrix}
  a \\
  b
\end{bmatrix},\in \mathbb{R}^{2}
```  

Let's upgrade the problem to fitting with a d-degree polynome $\hat{Y} = P(X)$ with $P(X) = \sum_{k=0}^{d} {{p_k}.X^k}$ :

```math
A = \begin{bmatrix}
  {X_1}^d & ... & {X_1}^0 \\
  {X_2}^d & ... & {X_2}^0\\
  \vdots \\
  {X_n}^d & ... & {X_n}^0
\end{bmatrix}, \in M_{(n,d)} ({\mathbb{R}}),
```

```math
w = \begin{bmatrix}
  p_0 \\
  \vdots \\
  p_d \\
\end{bmatrix},\in \mathbb{R}^{d}
```  
The processus thought in **§2** : $E \in  col(A)^{\perp} \Leftrightarrow \in ker(A^T) \implies w = {(A^T.A)^{-1}}A.Y$ is still valid.


### 4 - (X, Y) Multivariate polynomial regression : Matrix inversion method

Let's add a dimension to our problem : 
Let $(n, m) \in \mathbb{N}\times\mathbb{N}$, $(X, Y) \in \mathbb{R}^{m}\times\mathbb{R}^{n}$ be two sets of real values.  
```math
\text{Rewrite : } X=(X_{j})_{1 \leq j \leq m} \text{ and }  Y=(Y_{i})_{1 \leq i \leq n}  
```

Now w define a discrete real field $(Z_{i,j})_{({1 \leq i \leq n}, {1 \leq j \leq m})}$.

We want to make a regression with two polynomials $P_{x}$ and $P_{y}$ of degrees $d_x$ and $d_y$ such as :  
```math
{1 \leq k \leq n}, {1 \leq j \leq m}\text{ : } \hat{Z}_{ij} = P_{x}({X_{j}}) + P_{y}({Y_{i}})
```
```math
P_{x} \text{ and } P_{y} \text{ minimize } E = Z - \hat{Z}
```
```math
Z = \begin{bmatrix}
  {Z_{11}} & ... & {Z_{1m}} \\
  {Z_{21}} & ... & {X_{2m}}\\
  \vdots \\
  {Z_{n1}} & ... & {Z_{nm}}
\end{bmatrix}, \in M_{(n,m)} ({\mathbb{R}}),
```
```math
\hat{Z}_{ij} = P_{x}({X_{j}}) + P_{y}({Y_{i}}) = \sum_{j=0}^{d_x}{p_{x,j}.X^j} + \sum_{i=0}^{d_y}{p_{y,i}.Y^k}
```
Here, we want to reframe a bit our problem to use previous results.  

Let f be a bijection from  $M_{(n,m)}({\mathbb{R}})$ to ${\mathbb{R}}^{nxm}$ such as :
```math
f(Z) = Z_{vectorized} = \begin{bmatrix}
  {Z_{11}} \\
  {Z_{21}} \\
  \vdots \\
  {Z_{n1}} \\
  {Z_{12}} \\
  \vdots \\
  {Z_{n2}} \\
   \vdots \\
   {Z_{1m}} \\
   \vdots \\
   {Z_{nm}}
\end{bmatrix}, \in {\mathbb{R}}),
```
One can define a matrix $A \in  $M_{(nxm,d_{x}+d_{y}+1)}({\mathbb{R}})$ :

$$
A = \left(\begin{array}{cccc|cccc|c}
x_1 & x_1^2 & \cdots & x_1^{d_x} & y_1 & y_1^2 & \cdots & y_1^{d_y} & 1 \\
x_1 & x_1^2 & \cdots & x_1^{d_x} & y_2 & y_2^2 & \cdots & y_2^{d_y} & 1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
x_1 & x_1^2 & \cdots & x_1^{d_x} & y_m & y_m^2 & \cdots & y_m^{d_y} & 1 \\
\hline
x_2 & x_2^2 & \cdots & x_2^{d_x} & y_1 & y_1^2 & \cdots & y_1^{d_y} & 1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
x_2 & x_2^2 & \cdots & x_2^{d_x} & y_m & y_m^2 & \cdots & y_m^{d_y} & 1 \\
\hline
x_n & x_n^2 & \cdots & x_n^{d_x} & y_m & y_m^2 & \cdots & y_m^{d_y} & 1
\end{array}\right)
$$

And a coefficient vector $w \in {\mathbb{R}}^{d_{x}+d_{y}+1}$ :
```math
w = \begin{bmatrix}
  {p_{x,0}} \\
  \vdots \\
  p_{x,d_x} \\
  {p_{y,0}} \\
  \vdots \\
  p_{y,d_x} \\
\end{bmatrix}
```  
We can write the equation :  
```math
\hat{Z}_{vectorized} = A.w
```

And apply the same algorithm as in **§A).3)** to obtain our coefficients.

```math
w = (A^{T}.A)^{-1}.A^T.Z_{vectorized}
```


=================================================================================

## B) Python

First few scripts were done in Python.
Few different approach were tried, they are summarized chronically.
Going from "usual" linear fit to multivariate fit... Meaning the last one is also the more achieved one.


### 0 - Setting up environment  

Install miniconda using [following tutorial](https://www.anaconda.com/docs/getting-started/miniconda/main).  

In shell execute following command : `conda env create -f environment.yml`.  
Activate environment with `conda activate benv`.  


### 1 - 2D Linear regression "naive approach" : Least Square maxima search in bounded phase space  `MSE_phase_space_search.py`

Script that runs a MSE resolution approach described in **§A).1**   
(Unefficient one - just a try - you can skip it)  

**Run the script**  
Modify hardcoded values for lattice limits `(amin, amax, bmin, bmax)` and resolution `(dx, dy)`and definition of data to fit $Y$ with subsequent random noise.    

Random is generated according to the choice of parameters. Then regression is computed and result is plotted in `Matplotlib` Figure.  

Then run `python3 MSE_phase_space_search.py`  

### 2 - 2D Linear regression : Vector projection over linear span space `Linear_regression_MSE_vector_projection.py`  

Script that runs a Linear regression approach described in **§A).2**.
Y data is randomly generated with random linear coefficients.  
Yet, it has been modified to fit with polynomes of higher degree (`int d`).  

**Run the script**  
Modify hardcoded values for :
* `(int) nb` : number of samples

Random is generated according to the choice of parameters. Then regression is computed and result is plotted in `Matplotlib` Figure.  

Then run `python3 Linear_regression_MSE_vector_projection.py`  

### 3 - Linear/Polynomial regression : Matrix inversion method `Linear_regression_MSE_matrix_inversion.py`

Script that runs a Polynomial regression approach described in **§A).3**.  

**Run the script**  
Modify hardcoded values for :
* `(int) nb` : number of samples
* `(float np.dnarray) coeff_th` : coefficients of the polynom you want -> degree of the polynom fit depend on size of coeff_th vector : $\sum{k=0}^{d}{coeff_{th, k}*X^{d-k}}$  

Random is generated according to the choice of parameters. Then regression is computed and result is plotted in `Matplotlib` Figure.  

Then run `python3 Linear_regression_MSE_matrix_inversion.py`

### 4 - Multi-Polynomial regression : Matrix inversion method `Multi_Polynomial_regression_MSE_matrix_inversion.py`

Script that runs a Polynomial regression approach described in **§A).3**.  

**Run the script**  
Modify hardcoded values for :
* `(bool) classic_demo` : Polynomial(X^2, X, Y^2, Y)
* `(bool) ellipsis_demo` : 3D-Ellipis related data
* `(bool) gaussian_demo` : 2D-Gaussian related data

Random is generated according to the choice of parameters. Then regression is computed and result is plotted in `Matplotlib` Figure.  

Then run `python3 Multi_Polynomial_regression_MSE_matrix_inversion.py`


=================================================================================

## B) C++

Each C++ script can be divided in two parts : computing and plotting.  
The computing part is done with `Eigen` library for matrix and vector operations.  
The plotting part is done with `OpenGL` library, graphical engine is in `plot_graph.cpp/hpp` and `plot_graph_3D.cpp/hpp` files. 

**Note : I built the 3D engine over functions and doc I found on internet, it is still a bit sketchy (not convenient) on the rotation part, but it works.**


### 0 - Setting up environment

Install following libraries in order to compile the code :  
`sudo apt install cmake`  
`sudo apt install libgl-dev libglew-dev libglfw3-dev libglm-dev freeglut3-dev libeigen3-dev`  

### 1 - Compile with Cmake

Go into `/Cpp` folder  
Run `cmake -B build -S .` to configure cmake  
Edit `CmakeLists.txt` -> Modify `set(MAIN_FILE file_to_compile.cpp)` line  
file_to_compile.cpp can be :  
* `Linear_regression_MSE_matrix_inversion.cpp` -> (Linear fit of (X, Y) data)
* `Polynomial_regression_MSE_matrix_inversion.cpp` -> (Polynomial fit of (X, Y) data)  
* `Multi_Polynomial_regression_MSE_matrix_inversion.cpp` -> (Polynomial fit of (X, Y, Z) data)  
Run `cmake --build build` to compile  
Run `./build/regression` to execute program  

### 2 - Linear regression : Matrix inversion method `Linear_regression_MSE_matrix_inversion.cpp`

Script that runs a Linear regression approach described in **§A).3**.  
It first generate noised data, then does regression and plot it using graphical engine.  

**Set-up the script**  
Modify hardcoded values for :
*  `int n` : points number in X vector
*  `int d` : degree of polynom fit
*  `float xmax` : max value in X vector
*  `float xmin` : min value in X vector
*  `float xmin` : min value in X vector
*  `Eigen::VectorXf w_th` : coefficients for linear regression in order to generate data:  
$Y = w_{th}[0].X + w_{th}[1].(1, ..., 1)^T$  

**Run the script**  
Follow *1 - Compile with Cmake* instructions.  

### 3 - Polynomial regression : Matrix inversion method `Polynomial_regression_MSE_matrix_inversion.cpp`

Script that runs a Polynomial regression approach described in **§A).3**.
It first generate noised data, then does regression and plot it using graphical engine.  

**Set-up the script**  
Modify hardcoded values for :
*  `int n` : points number in X vector
*  `int d` : degree of polynom fit
*  `float xmax` : max value in X vector
*  `float xmin` : min value in X vector
*  `float xmin` : min value in X vector
*  `Eigen::VectorXf w_th` : coefficients for polynomial regression in order to generate data:  
$\sum{k=0}^{d}{w_{th, k}*X^{d-k}}$  

**Run the script**  
Follow *1 - Compile with Cmake* instructions.  

### 3 - Polynomial regression : Matrix inversion method `Multi_Polynomial_regression_MSE_matrix_inversion.cpp`

Script that runs a Multi-polynomial regression approach described in **§A).4**.
It first generate noised data, then does regression and plot it using 3D graphical engine.  

**Set-up the script**  
Modify hardcoded values for :
*  `bool demo_simple` : Demo of algorithm with Z = fct(X^2; Y^2, Y^1)
*  `bool demo_gaussian` :  Demo of algorithm with Z = exp(X^2+ Y^2)  


**Run the script**  
Follow *1 - Compile with Cmake* instructions.









