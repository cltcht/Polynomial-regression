# Polynomial-regression
Few scripts using different programming languages and approaches to tackle fitting problems.

## Summary
As personal project I wanted to study one mathematical problem along with differents programming languages.
I begun with Python see /Python repo for scripts and mathematics.
Then I switched to Cpp. Copy-pasted few graphic engine functions from [kavan010 git](https://github.com/kavan010).
For reminders and proof of algebra theorems used, please see [this website](https://textbooks.math.gatech.edu/ila/1553/index2.html)

** I might push some script using Rust ** 


## Python
First few scripts were done in Python.
At first I decided to go from scratch and have zero look online and just give it a try in the morning with what I thought overnight.
### 2D Linear regression : global maximum of phase space  
Let $n \in \mathbb{N}$, $(X, Y) \in \mathbb{R^n}\times\mathbb{R^n}$ be two set of $n$ real values.
Let's rewrite $X = (X_k)_{1 \leqslant k \leqslant n} and Y = (Y_k)_{1 \leqslant k \leqslant n}$.
Objective is to find $(a, b) \in \mathbb{R}\times\mathbb{R}$ such as an approximation $\hat{Y}(a, b, X)$ is as close as possible of $Y$.
One usually uses $R^2$ defined as ${SS_mean - SS_fit}over{SS-mean}$ as fitting metrics with :\newline
$SS_mean=\sum_{k=0}^{n} Y_k over n$ and $SS_fit=\sum_{k=0}^{n} \hat{Y_k} over n$

