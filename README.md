# Polynomial-regression
Few scripts using different programming languages and approaches to tackle fitting problems.

## Summary
As personal project I wanted to study one "simple" mathematical problem along with differents programming languages.
I choosed polynomial regression. I've seen online multiple gradient/machine learning based approaches which I don't like.
It feels like "using a bulldozer to crack an egg" while you can use more elegant linear-algebra based solutions.

I begun with Python, see /Python repo for scripts.
Then I switched to C++, see /Cpp repo for scripts. 

Copy-pasted few graphic engine functions from [kavan010 git](https://github.com/kavan010) for C++.
For reminders and proofs of algebra theorems used, please see [this website](https://textbooks.math.gatech.edu/ila/1553/index2.html)

------------------------------

**I might push some script using Rust later** 

----------------------------------


## Python

First few scripts were done in Python...
Few different approach were tried, they are summarized chronically.
Going from "usual" 2D linear fit to 3D multivariate fit.

### Setting up environment

Install miniconda using [following tutorial](https://www.anaconda.com/docs/getting-started/miniconda/main).

In shell execute following command : `conda env create -f environment.yml`.


### 2D Linear regression : search for maximum in bounded phase space  

**At first I decided to go from scratch and have zero look online and just give it a try in the morning with what I thought overnight.**

Let $n \in \mathbb{N}$, $(X, Y) \in \mathbb{R^n}\times\mathbb{R^n}$ be two set of $n$ real values.

```math
X=(X_{k})_{1 \leq k \leq n} and Y=(Y_{k})_{1 \leq k \leq n}
```

Objective is to find $(a, b) \in \mathbb{R}\times\mathbb{R}$ such as a linear approximation $\hat{Y}$ is as close as possible of $Y$ :
```math
\hat{Y}(a, b, X) = aX + b \times (1, ..., 1)^T
```

Let write the mean ${\sum_{k=1}^{n} Y_{k} }\over{n}$ as $\bar{Y}$ .  

One usually maximises $R^2$ defined as ${SS_{mean} - SS_{fit}}\over{SS_{mean}}$ as fitting metrics with :

```math
{SS_{mean}} = {\sum_{k=1}^{n} {({Y}_k - \bar{Y})}^2\over{n}}
```
```math
{SS_{fit}} = {\sum_{k=1}^{n} {(\hat{Y}_k - \bar{Y})}^2\over{n}}
```

Realistically, one just want to minimise ${SS_{fit}}$. Therefore our goal is to find a vector $(\hat{a}, \hat{b}) = argmin(\hat{Y}(a, b, X))$.

Hypothesis, In order not to compute indefinitely :
* One can set limits to the vector space that contains $(a, b)$. Let's call this bounded space $A \times B$
* One can define a lattice subspace of $A \times B$ with definite resolution over $A$ and $B$.
Meaning we will search for all existing $(a, b)$ over a definite lattice subspace of $A \times B$.

Limits of this approach :
* Global maxima might be out of lattice subspace of $A \times B$, we have to "guess" good limits to search in phase space
* Great chance that lattice resolutions over A and B are set such as best $(\hat{a}, \hat{b}) is not on one point of the lattice.

### Run the script
Modify hardcoded values for lattice limits and resolution and definition of $Y$ with subsequent random noise.







