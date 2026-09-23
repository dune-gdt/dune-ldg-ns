# dune-ldg-ns

Local discontinuous Galerkin (LDG) discretization of the incompressible Navier–Stokes equations in 2D and 3D, with a
monolithic **fractional-step θ** time integrator. It is a DUNE module on top of
[dune-gdt](https://github.com/dune-gdt/dune-gdt) (`main`).

* LDG fluxes after Cockburn–Kanschat–Schötzau: central or β-switched $\hat u$, $C_{11}=\nu\eta/h_F$, and equal
  treatment of every velocity component through scalar blocks.
* Skew-symmetric, upwinded DG convection that is coercive for any linearization velocity. Picard (Oseen) iteration.
* FS-θ (θ = 1 − √2/2), Crank–Nicolson, backward Euler and one-step θ, all through the same substep code. Second-order
  pressure recovery.
* Weak Dirichlet and do-nothing outflow boundaries, and drag/lift from the consistent LDG traction.
* Test cases: Taylor–Green (2D, 3D), Kovasznay (2D steady), DFG 2D-2 / 2D-3.

The formulation, algorithm, architecture, validation results and roadmap are in **[doc/design.md](doc/design.md)**.

## Validation (summary)

| test | result |
|---|---|
| Kovasznay Re = 40, $Q_1/Q_0$, $Q_2/Q_1$ | $L^2$ EOC $u$: 1.99 / 3.00, $p$: 1.08 / 2.10 |
| Taylor–Green 2D, FS-θ, $Q_2/Q_1$ | temporal EOC $u$: 2.02, recovered $p$: → 2, multiplier $\lambda$: 1.03 |
| Taylor–Green 3D ($z$-invariant) | runs, $\|Bu-g_B\|\approx 10^{-15}$ |
| FS-θ coefficients | identities, 2nd order with forcing, $\lvert R(-\infty)\rvert = 1/\sqrt2$ |
| DFG 2D-3 | smoke run on a crude mesh only, see below |

## Layout

```
dune/ldg-ns/        headers (integrands, operator, saddle point solver, time stepper, test cases, tools)
test/               gtest binaries: fstheta_coefficients, kovasznay, taylor_green
src/                ldg_ns_dfg_cylinder (+ dfg_2d_2.ini, dfg_2d_3.ini)
grids/              dfg-cylinder-2d.geo (gmsh)
doc/design.md       design spec
dev/standalone/     standalone development build (no vcpkg), see below
dev/tools/          dfg_mesh.py (crude DFG mesh without gmsh, smoke tests only)
```

## Building as a DUNE module (on top of a dune-gdt build)

dune-gdt `main` builds with CMake presets and vcpkg (see its `AGENTS.md`). Build dune-gdt first, then point this module
at that build tree and reuse dune-gdt's vcpkg installation:

```bash
# 1. dune-gdt (pinned: main @ 98478081fd85b5fa9003c581497b2fc5094f1df7)
git clone https://github.com/dune-gdt/dune-gdt.git && cd dune-gdt
git checkout 98478081fd85b5fa9003c581497b2fc5094f1df7
cmake --preset=release && cmake --build --preset=release -- -j"$(( $(nproc) - 1 ))"
cd ..

# 2. dune-ldg-ns
export DUNE_GDT_DIR=$PWD/dune-gdt DUNE_GDT_PRESET=release
cd dune-ldg-ns
cmake --preset=release
cmake --build --preset=release
ctest --preset=release
```

> **Not yet verified.** The module build (`CMakeLists.txt`, `CMakePresets.json`, `dune.module`) follows the standard
> `dune_project()` conventions. It could not be exercised in the development environment, because vcpkg's download
> mirrors were not reachable there. Expect small adjustments, such as how `dune-gdt_DIR` / `dune-xt` targets are found
> from the dune-gdt build tree. All numbers above come from the standalone build.

## Standalone development build

`dev/standalone/CMakeLists.txt` compiles dune-xt's sources from a dune-gdt checkout into a static library. It links
DUNE 2.10 core modules and dune-alugrid, oneTBB, Eigen 3.4, Boost 1.86 and googletest, all installed or checked out
by hand. `dev/standalone/config.h` replaces the generated `config.h`. The exact commits are listed in
[doc/design.md §9](doc/design.md#9-pinned-versions).

```bash
cmake -S dev/standalone -B build-standalone -G Ninja \
  -DDUNE_GDT_SOURCE_DIR=... -DDUNE_PREFIX=... -DBOOST_INCLUDE_DIR=... -DBOOST_SOURCE_DIR=... \
  -DEIGEN3_INCLUDE_DIR=... -DGTEST_SOURCE_DIR=...
cmake --build build-standalone -j2
cd build-standalone && ./test_fstheta_coefficients && ./test_kovasznay && ./test_taylor_green
```

Runtimes (2 cores, Release): `test_kovasznay` ≈ 5 min, `test_taylor_green` ≈ 3–4 min. The temporal study can be
narrowed with `DUNE_LDG_NS_TG_SCHEMES=fs-theta` and `DUNE_LDG_NS_TG_TIME_STEPS=0.2,0.1,0.05`.

## DFG benchmarks

```bash
gmsh -2 -format msh22 -setnumber h 0.02 -setnumber hc 0.005 grids/dfg-cylinder-2d.geo -o dfg-cylinder-2d.msh
./ldg_ns_dfg_cylinder dfg_2d_3.ini -grid.filename dfg-cylinder-2d.msh -timestepping.dt 0.005
```

This writes `t, c_D, c_L, Δp, Picard iterations` per macro step as CSV. Reference values (Schäfer–Turek 1996,
John 2004, FEATFLOW) are in [doc/design.md §5.3](doc/design.md#53-dfg-2d-2--2d-3-schäferturek-1996).
**No quantitative DFG results yet.** gmsh was not available during development. A smoke run on a crude Delaunay
mesh (`dev/tools/dfg_mesh.py`, ~900 triangles, affine cylinder) only shows that the pipeline runs end to end.

## License

Dual licensed under the BSD 2-Clause License and GPL-2.0+ with runtime exception, like dune-gdt.
