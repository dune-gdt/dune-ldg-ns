# dune-ldg-ns — design

LDG discretization in space and monolithic fractional-step θ scheme in time for the incompressible
Navier–Stokes equations in 2D and 3D, built as a DUNE module on top of
[dune-gdt](https://github.com/dune-gdt/dune-gdt) `main`.

Status: **v0.1, validated skeleton.** Every formula below is implemented. The numbers in §7 were produced by the
code in this repository. §8 lists what is missing.

---

## 1. Problem

Find $u:\Omega\times(0,T]\to\mathbb R^d$, $p:\Omega\times(0,T]\to\mathbb R$ ($d=2,3$) with

$$
\partial_t u - \nu\Delta u + (u\cdot\nabla)u + \nabla p = f,\qquad \nabla\cdot u = 0 \quad\text{in }\Omega,
$$

$u = g$ on $\Gamma_D$ (weakly imposed), do-nothing $\nu\partial_n u - p n = 0$ on $\Gamma_N$ (outflow), $u(0)=u_0$.
If $\Gamma_N=\emptyset$, $p$ is fixed by $\int_\Omega p = 0$ (bordered system, `mean_pressure_constraint`).

## 2. Spaces

Let $\mathcal T_h$ be a conforming, affine grid of simplices or cubes. Let $S_h$ be the scalar DG space
$Q_k/P_k$ (`DiscontinuousLagrangeSpace<GV, 1>`), and let $N=\dim S_h$.

* velocity $u_h\in S_h^d$ (`DiscontinuousLagrangeSpace<GV, d>` with *dimwise* numbering, so the DoF vector is
  component-blocked: $[u_0;u_1;\dots]$, each block of length $N$ in the scalar numbering),
* pressure $p_h\in Q_h$ = DG $Q_{k-1}/P_{k-1}$ (inf-sup stable, default) or equal order $k$ with pressure jump
  stabilization $\gamma h_F[[p]][[q]]$,
* auxiliary gradient $\sigma_h\in S_h^{d\times d}$, eliminated locally (mass matrix is element-block diagonal).

Face notation: on an inner face $F=\partial K^-\cap\partial K^+$, $n$ is the outer normal of $K^-$ (the "inside"
element of the dune-gdt intersection), $[[w]] = w^- - w^+$, $\{w\}=\tfrac12(w^-+w^+)$.
The LDG switch is $\beta_F=\beta\cdot n$ with a global vector $\beta$ (default $0$, i.e. central LDG fluxes).

## 3. Spatial discretization (LDG, Cockburn–Kanschat–Schötzau 2004)

### 3.1 Numerical fluxes

| trace | inner face | $\Gamma_D$ | $\Gamma_N$ |
|---|---|---|---|
| $\hat u$ (for $\sigma$) | $\{u\}+\beta_F[[u]]$ | $g$ | $u^-$ |
| $\hat\sigma n$ | $\{\sigma\}n-\beta_F[[\sigma]]n-C_{11}[[u]]$ | $\sigma^- n - C_{11}(u-g)$ | natural (dropped) |
| $\hat p$ | $\{p\}$ | $p^-$ | natural (dropped) |
| $\hat u\cdot n$ (continuity) | $\{u\}\cdot n$ | $g\cdot n$ | $u^-\cdot n$ |

with $C_{11}=\nu\eta/h_F$ and $h_F=\operatorname{diam}F$. Any $\eta>0$ is stable for LDG. The default is $\eta=4k^2$.

### 3.2 Gradient: $M\sigma_{ij} = G_j u_i + b_{ij}(g)$

Insert $\hat u$ into $\int_K\sigma_{ij}\tau = -\int_K u_i\partial_j\tau+\int_{\partial K}\hat u_i n_j\tau$ and integrate
by parts back:

$$
(G_j u)(\tau) = \sum_K\int_K \partial_j u\,\tau
 \;-\;\sum_{F\,\text{inner}}\int_F [[u]]\,n_j\,\big((\tfrac12-\beta_F)\tau^- + (\tfrac12+\beta_F)\tau^+\big)
 \;-\;\int_{\Gamma_D} u\,n_j\,\tau,
\qquad b_{ij}(\tau)=\int_{\Gamma_D} g_i n_j\tau .
$$

(Check: $\hat u-u^- = -(\tfrac12-\beta_F)[[u]]$ and $\hat u-u^+ = (\tfrac12+\beta_F)[[u]]$.) This is implemented by
`DirectionalDerivative(j, +1)`, `WeightedAverageJumpNormal(j, -1, β)`, `BoundaryNormal(j, -1)` and the right hand side
`BoundaryDataNormal(g, i, j)`.

### 3.3 Viscous and pressure blocks

Eliminating $\sigma$ gives the primal LDG form. The same scalar matrix applies to every velocity component:

$$
A_\text{visc} = \nu\sum_j G_j^\top M^{-1} G_j + J,\qquad J(u,v) = \sum_{F\,\text{inner}}\int_F\tfrac{\nu\eta}{h_F}[[u]][[v]]
+\int_{\Gamma_D}\tfrac{\nu\eta}{h_F}uv,
$$

$$
F_i(v) = (f_i,v) + \int_{\Gamma_D}\tfrac{\nu\eta}{h_F}g_i v - \nu\sum_j (G_j v)^\top M^{-1}b_{ij}.
$$

$M^{-1}$ is the exact element-block inverse (`element_block_inverse`), so $A_\text{visc}$ is sparse with the DG
neighbour stencil (plus the neighbours-of-neighbours coupling that is inherent to LDG with $\beta\ne 0$). The
divergence is $B_j = -G_j(\beta{=}0)$ with pressure test functions:

$$
(Bu)(q) = -\sum_K\int_K \nabla\cdot u\,q+\sum_{F\,\text{inner}}\int_F[[u]]\cdot n\{q\}+\int_{\Gamma_D}u\cdot n\,q,
\qquad g_B(q)=\int_{\Gamma_D} g\cdot n\,q .
$$

The momentum pressure term is $B^\top p$, which is the consistent $\hat p=\{p\}$ flux, so the saddle point matrix is
symmetric in its off-diagonal blocks. On $\Gamma_N$, neither $G$ nor $B$ has a boundary term. This gives exactly the
do-nothing condition.

### 3.4 Convection (Oseen linearization $w\approx u$)

For each component, the skew-symmetrized form (Temam, Di Pietro–Ern) with optional upwinding is used:

$$
\begin{aligned}
t_h(w;u,v) ={}& \sum_K\int_K (w\cdot\nabla u)v + \tfrac12(\nabla\cdot w)uv
 - \sum_{F\,\text{inner}}\int_F \{w\}\!\cdot\! n\,[[u]]\{v\} + \tfrac12[[w]]\!\cdot\! n\,\{uv\}
 - [\![\text{upw}]\!]\,\tfrac12|\{w\}\!\cdot\! n|\,[[u]][[v]] \\
&- \int_{\partial\Omega}(w\cdot n)^-uv,\qquad\text{rhs: } -\int_{\Gamma_D}(w\cdot n)^- g\,v .
\end{aligned}
$$

Face-wise cancellation gives, **for any** $w$ (no discrete divergence constraint on $w$ is needed):

$$
t_h(w;v,v)=\sum_{F}\tfrac{\text{upw}}2\int_F|\{w\}\cdot n|[[v]]^2+\tfrac12\int_{\partial\Omega}|w\cdot n|v^2\;\ge 0 .
$$

On the outflow boundary, the $-(w\cdot n)^-$ term damps backflow (directional do-nothing). The coefficients of the
four inner-face blocks are derived in `local/integrands/convection.hh`.

### 3.5 Semi-discrete system

With $T(w)$ the matrix of $t_h(w;\cdot,\cdot)$ and $F^\text{conv}(w)$ its $\Gamma_D$ data:

$$
M\dot u_i + \big(A_\text{visc}+T(u)\big)u_i + B_i^\top p = F_i + F_i^\text{conv}(u),\qquad \sum_j B_ju_j - Cp = g_B .
$$

Write $R(t,u)=(A_\text{visc}+T(u))u - F(t)-F^\text{conv}(t;u)$ (`LdgNavierStokesOperator::spatial_residual`).

## 4. Time discretization: monolithic FS-θ

### 4.1 Substeps

$\theta = 1-\tfrac{\sqrt2}2$, $\theta'=1-2\theta$, $\alpha=\theta'/(1-\theta)=2-\sqrt2$, $\beta=1-\alpha$
(Bristeau–Glowinski–Périaux 1987; Turek 1999). A macro step $K$ has three substeps $(\tau, c_\text{imp}, c_\text{exp})$:

| substep | $\tau$ | $c_\text{imp}$ | $c_\text{exp}$ |
|---|---|---|---|
| 1 | $\theta K$ | $\alpha\theta K$ | $\beta\theta K$ |
| 2 | $\theta' K$ | $\beta\theta' K$ | $\alpha\theta' K$ |
| 3 | $\theta K$ | $\alpha\theta K$ | $\beta\theta K$ |

Each substep $t_a\to t_b=t_a+\tau$ solves the coupled problem

$$
Mu^b + c_\text{imp}R(t_b,u^b) + \tau B^\top\lambda^b = Mu^a - c_\text{exp}R(t_a,u^a),\qquad Bu^b - C\lambda^b = g_B(t_b).
$$

The pressure enters fully implicitly, scaled with $\tau$. $c_\text{imp}=\alpha\theta K=\beta\theta' K$ is identical in all
three substeps, so the implicit operator has the same structure throughout. The data are weighted like the operator
($c_\text{imp}F(t_b)+c_\text{exp}F(t_a)$), which keeps second order for time-dependent forcing. Applying $f$ fully at
one endpoint with weight $\tau$ drops to first order (checked on a scalar ODE: EOC 1.00 vs. 2.01).

`crank-nicolson`, `backward-euler` and a general one-step `theta` share the same code path
(`one_step_theta_substeps`).

### 4.2 Nonlinear solver

Each substep uses Picard (Oseen) iteration: the matrix is $M + c_\text{imp}(A_\text{visc}+T(w))$, and $w$ is updated until
$\|u^{(m+1)}-u^{(m)}\|_M\le\text{tol}\,\|u^{(m+1)}\|_M$. It converges in 6–10 iterations for Re ≤ 100 at the step
sizes of §7. Newton (adding the $(\cdot\cdot\nabla)w$ block) is on the roadmap.

### 4.3 Pressure

The multiplier $\lambda^b$ is only a **first-order** approximation of $p(t_b)$. This is an inherent property of FS-θ as a
DAE integrator: the implicit/explicit split of $R$ does not carry over to $B^\top p$. A model DAE shows exactly that:
velocity EOC 1.97, multiplier EOC 1.03. A second-order pressure is recovered at any output time from the discrete
pressure Poisson equation

$$
\Big(\sum_j B_jM^{-1}B_j^\top\Big)p = -\sum_j B_jM^{-1}R_j(t,u) - \tfrac{d}{dt}g_B(t)
$$

(`recover_pressure`: bordered with the mean constraint, factorized once, $\dot g_B$ by central differences). This is
the recommended output for drag/lift and $\Delta p$.

### 4.4 Linear solver

The monolithic matrix
$\begin{bmatrix} I_d\otimes(aM+c(A_\text{visc}+T)) & \tau B^\top & (q)\\ \tau B & -\tau C & 0\\ (\tau q^\top) & 0 & 0\end{bmatrix}$
is assembled from triplets. It is factorized with Eigen `SparseLU` (symbolic analysis once, because the pattern is fixed)
or UMFPACK if available. This is fine for 2D and small 3D problems. For 3D, see §8.

## 5. Validation cases

### 5.1 Taylor–Green vortex (2D, 3D)

$u=(-\cos\pi x\sin\pi y,\ \sin\pi x\cos\pi y\,[,0])F(t)$ and
$p=-\tfrac14(\cos2\pi x+\cos2\pi y)F(t)^2$, with $F=e^{-2\pi^2\nu t}$ on $[-1,1]^d$. The exact velocity is imposed on $\partial\Omega$.
The 3D case is the $z$-invariant extrusion, which is an exact solution and exercises all 3D code paths. The
classical 3D TGV transition case at Re = 1600 has no closed form and is not included.
Temporal EOCs are measured against a $K/4$ reference on the same mesh, so the spatial error cancels.

### 5.2 Kovasznay flow (2D, steady)

Re = 40, $\lambda = \mathrm{Re}/2-\sqrt{\mathrm{Re}^2/4+4\pi^2}$, $\Omega=[-0.5,1]\times[-0.5,1.5]$. This case checks spatial
rates: $k+1$ for $u$ in $L^2$, $k$ for $p$.

### 5.3 DFG 2D-2 / 2D-3 (Schäfer–Turek 1996)

Channel $[0,2.2]\times[0,0.41]$ with a cylinder of radius 0.05 at $(0.2,0.2)$, $\nu=10^{-3}$, parabolic inflow with
$U_\max=1.5$ (2D-2) or $1.5\sin(\pi t/8)$ (2D-3), and do-nothing outflow. $c_{D/L}=2F_{D/L}/(U_\text{mean}^2 D)$ with
$U_\text{mean}=1$ and $D=0.1$.

The forces are computed from the **LDG flux** on the cylinder faces,
$F=\int_S p\,n-\nu\hat\sigma n$ with $\nu\hat\sigma n=\nu\sigma n-\tfrac{\nu\eta}{h_F}(u-g)$, which is the traction consistent
with the discrete momentum balance (`body_force`). $\Delta p=p(0.15,0.2)-p(0.25,0.2)$.

Reference values:

| case | quantity | reference | source |
|---|---|---|---|
| 2D-2 | $c_{D,\max}$ | [3.22, 3.24] | Schäfer–Turek 1996 |
| 2D-2 | $c_{L,\max}$ | [0.99, 1.01] | Schäfer–Turek 1996 |
| 2D-2 | St | [0.295, 0.305] | Schäfer–Turek 1996 |
| 2D-2 | $c_{D,\max}$ / $c_{L,\max}$ / St | 3.2200 / 0.9859 / 0.30188 (level 6, Δt = 1/200) | FEATFLOW |
| 2D-3 | $c_{D,\max}$, $t$ | 2.950921575 at $t=3.93625$ | John 2004 |
| 2D-3 | $c_{L,\max}$, $t$ | 0.47795 at $t=5.693125$ | John 2004 |
| 2D-3 | $\Delta p(8)$ | −0.1116 | John 2004 |

The grid is affine, so the cylinder is a polygon. Quantitative runs need a fine $h_c$ on the cylinder or curved
(isoparametric) elements; see §8. The mesh is generated from `grids/dfg-cylinder-2d.geo` with gmsh (MSH 2.2), and
read with `Dune::GmshReader` into `ALUGrid<2,2,simplex,conforming>`.

## 6. Architecture (mapping to dune-gdt)

```
dune/ldg-ns/
  local/integrands/ldg.hh          scalar LDG integrands (dune-gdt Local{Unary,Binary,Quaternary}*IntegrandInterface)
  local/integrands/convection.hh   Oseen convection integrands (element, inner face, boundary, boundary data)
  operators/ldg-navier-stokes.hh   LdgNavierStokesOperator: spaces, assembly (BilinearForm + MatrixOperator +
                                   VectorFunctional, grid walks with ApplyOn filters), data vectors, residual
  solvers/saddle-point.hh          monolithic block matrix + SparseLU/UMFPACK
  timestepping/fractional-step-theta.hh  substeps, Picard, steady solver, projection, pressure recovery
  functionals/benchmark-quantities.hh    body_force (LDG flux), point_value
  testcases/{interface,taylor-green,kovasznay,dfg-cylinder}.hh
  tools/{block-inverse,boundary-info,errors}.hh
```

Design decisions:

* **Scalar blocks, vector unknowns.** Every block is assembled once as a scalar $N\times N$ (or $P\times N$) matrix and
  reused for all $d$ components. This cuts assembly work by a factor of $d$ ($d^2$ for $G$), and keeps the integrands
  simple. It relies on the dimwise numbering of `DiscontinuousLagrangeSpace<GV, d>` matching the scalar numbering
  block-wise. The constructor checks this.
* **Eigen storage.** `EigenRowMajorSparseMatrix` is the dune-xt matrix. Its backend is used directly for the sparse
  products $G^\top M^{-1}G$ and the triplet assembly of the block system.
* **Time as a parameter.** All data are `GridFunction`s with a `"t"` parameter, so dune-gdt's parametric machinery
  passes time through every local integrand.
* **Thread safety.** Integrands own their local functions and follow the dune-gdt copy-per-thread rule.
  `CenterBasedBoundaryInfo` is stateless (dune-xt's `FunctionBasedBoundaryInfo` keeps mutable local functions and is
  not safe in TBB walks).

## 7. Results (this repository, standalone build, gcc 13.3, Release)

**Kovasznay, Re = 40, $Q_k/Q_{k-1}$, $\eta=4k^2$, steady Picard** (`test_kovasznay`):

| $k$ | $h$ | $\|u-u_h\|/\|u\|$ | EOC | $\|p-p_h\|$ | EOC | Picard its |
|---|---|---|---|---|---|---|
| 1 | 0.25 | 7.812e-02 | – | 1.514e-01 | – | 29 |
| 1 | 0.125 | 1.893e-02 | 2.05 | 7.165e-02 | 1.08 | 27 |
| 1 | 0.0625 | 4.767e-03 | 1.99 | 3.381e-02 | 1.08 | 25 |
| 2 | 0.25 | 1.038e-02 | – | 1.084e-02 | – | 25 |
| 2 | 0.125 | 1.312e-03 | 2.98 | 2.261e-03 | 2.26 | 25 |
| 2 | 0.0625 | 1.641e-04 | 3.00 | 5.266e-04 | 2.10 | 25 |

**Taylor–Green 2D, ν = 0.1, $Q_2/Q_1$ on 8×8, T = 1, errors vs. K/4 reference** (`test_taylor_green`):

| scheme | $K$ | $\|u-u_\text{ref}\|$ | EOC | $\|p_\text{rec}-p_\text{ref}\|$ | EOC | $\|\lambda-p_\text{ref}\|$ | EOC | $\|Bu-g_B\|$ |
|---|---|---|---|---|---|---|---|---|
| FS-θ | 0.2 | 5.32e-04 | – | 1.31e-03 | – | 9.96e-04 | – | 6e-16 |
| FS-θ | 0.1 | 1.31e-04 | 2.02 | 7.73e-05 | 4.09 | 5.25e-04 | 0.92 | 9e-16 |
| FS-θ | 0.05 | 3.24e-05 | 2.02 | 2.45e-05 | 1.66 | 2.57e-04 | 1.03 | 1e-15 |
| CN | 0.2 / 0.1 / 0.05 | 5.40e-03 / 1.91e-03 / 2.65e-04 | 1.50 / 2.85 | | | | | |
| BE | 0.2 / 0.1 / 0.05 | 5.26e-02 / 2.57e-02 / 1.13e-02 | 1.03 / 1.19 | | | | | |

These results show: (i) second order in time for FS-θ velocity, (ii) a first-order multiplier and a recovered pressure that
is second order asymptotically (the 4.09 is pre-asymptotic cancellation), (iii) Crank–Nicolson reaching second order only
late, because it does not damp the stiff error components of the initial projection, while FS-θ is strongly A-stable,
$|R(-\infty)|=\beta/\alpha=1/\sqrt2$ (`test_fstheta_coefficients`), and (iv) the constraint satisfied to round-off.

**Taylor–Green 3D** ($z$-invariant, $Q_1/Q_0$, $4^3$, $K=0.1$, $T=0.3$): relative velocity error 0.23 (spatially
unresolved, as expected at $h=0.5$), $\|Bu-g_B\|=2\cdot10^{-15}$, ≤ 8 Picard iterations. This is a smoke test only.

**DFG**: see README (smoke run only; no gmsh in the development environment, so no quantitative benchmark numbers yet).

## 8. Open items (priority order)

1. **Quantitative DFG runs**: gmsh mesh from `grids/dfg-cylinder-2d.geo`, $h$-sequence, $K\in\{1/100,1/200,1/400\}$;
   Strouhal post-processing script for 2D-2.
2. **Curved boundary** for the cylinder (isoparametric $P_2$ geometry via dune-curvedgrid or ALUGrid with a
   projection). Otherwise, $c_D$ is limited by the polygonal geometry error.
3. **Iterative solver for 3D**: FGMRES (dune-istl) with a block-triangular preconditioner. Use AMG on the velocity
   block ($aM+c(A+T)$, per component) and a pressure Schur approximation
   $S^{-1}\approx c\,M_p^{-1}+a\,(BM^{-1}B^\top)^{-1}$ (Cahouet–Chabard type, also valid for the Oseen case at
   moderate Re).
4. **Newton linearization** (Jacobian of $T(u)u$) with a Picard warm start.
5. **Performance**: $T(w)$ is re-assembled through a full grid walk in every Picard step. Instead, assemble into a fixed
   pattern and reuse the `MatrixOperator` (`matrixoperator.scaling`), and use TBB walks (`use_tbb`).
6. **3D DFG 3D-2Z/3D-3Z**, parallel (MPI) runs, and Python bindings following `python/gdt`.
7. **Upstream candidates for dune-gdt**: `element_block_inverse`, the scalar LDG integrands, and `CenterBasedBoundaryInfo`.

## 9. Pinned versions

| component | version / commit |
|---|---|
| dune-gdt | `main` @ `98478081fd85b5fa9003c581497b2fc5094f1df7` (2026-09-22) |
| dune-gdt vcpkg registry | `1d038f0d75b30a9ef00772ff390b0a7e03594863` |
| dune-common | 2.10 @ `145a243305ca3ee096600feb9e06405729830553` |
| dune-geometry | 2.10 @ `5673e95ac364ad3498aed9eaf65e0d224384d15a` |
| dune-grid | 2.10 @ `954436b88247e904628ec4d7c8bb7b2eaac08900` |
| dune-istl | 2.10 @ `21c67275b17e93918365177f93f42e4aaa9afd23` |
| dune-localfunctions | 2.10 @ `149c7866a11cab6ae5602b50b9f815088daa7a49` |
| dune-alugrid | 2.10 @ `bf551bd6740ba01d30feea9daaec4d77cdaed47c` |
| Eigen | 3.4 branch (dune-gdt pins 3.4.0#5 via vcpkg) |
| oneTBB | v2021.13.0 |
| Boost | 1.86.0 |
| googletest | v1.15.2 |
| compiler | GCC 13.3, C++20 |

## References

* B. Cockburn, G. Kanschat, D. Schötzau, *The local discontinuous Galerkin method for the Oseen equations*,
  Math. Comp. 73 (2004) 569–593.
* B. Cockburn, G. Kanschat, D. Schötzau, *A locally conservative LDG method for the incompressible Navier–Stokes
  equations*, Math. Comp. 74 (2005) 1067–1095.
* D. A. Di Pietro, A. Ern, *Mathematical Aspects of Discontinuous Galerkin Methods*, Springer 2012, Ch. 6.
* M. O. Bristeau, R. Glowinski, J. Périaux, *Numerical methods for the Navier–Stokes equations*, Comput. Phys.
  Rep. 6 (1987) 73–187.
* S. Turek, *Efficient Solvers for Incompressible Flow Problems*, Springer 1999.
* M. Schäfer, S. Turek, *Benchmark computations of laminar flow around a cylinder*, Notes Numer. Fluid Mech. 52,
  Vieweg 1996.
* V. John, *Reference values for drag and lift of a two-dimensional time-dependent flow around a cylinder*,
  Int. J. Numer. Meth. Fluids 44 (2004) 777–788.
* FEATFLOW benchmark pages, DFG 2D-2 and 2D-3 result tables,
  https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html
* L. I. G. Kovasznay, *Laminar flow behind a two-dimensional grid*, Proc. Camb. Phil. Soc. 44 (1948) 58–62.
