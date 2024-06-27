# dolfinx_optim

`dolfinx_optim` is a convex optimization add-on package to the `FEniCSx` finite-element library. It provides a simple Domain-Specific Language through FEniCSx `dolfinx` Python interface for solving convex optimization problems. In particular, it relies on the `Mosek` mathematical programming library. `Mosek` provides a state-of-the art interior-point solver for linear programming (LP), convex quadratic programming (QP), second-order conic programming (SOCP) and semi-definite programming (SDP).

* Github repository: https://github.com/bleyerj/dolfinx_optim
* Online documentation: https://bleyerj.github.io/dolfinx_optim/


## Prerequisites
**dolfinx_optim** requires: 
* **FEniCSx** (v.0.8), see [installation instructions here](https://fenicsproject.org/download/).
* **MOSEK** (>= version 10 with its Python Fusion interface), see [installation instructions here](https://www.mosek.com/downloads/). The Python interface can be simply installed via `pip`:

```
pip install -f https://download.mosek.com/stable/wheel/index.html Mosek
```

Mosek is a commercial software so users need a valid Mosek license. Free unlimited licenses are available for education and research purposes, see the [Academic License section](https://www.mosek.com/products/academic-licenses/).

## Installation and usage
Simply clone the [`dolfinx_optim` public repository](https://github.com/bleyerj/dolfinx_optim)
```
https://github.com/bleyerj/dolfinx_optim
```
and install the package by typing
```
pip install dolfinx_optim/ --user
```

## Features
Current version supports variational problem formulations resulting in Linear Programs, Second-Order Cone Programs, Semi-Definite Programs and Power or Exponential cone programs.

## License

All this work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/>) ![license](https://i.creativecommons.org/l/by-sa/4.0/88x31.png).

## Citing

Papers related to this project can be cited as:

- [Bleyer, Jeremy. "Automating the formulation and resolution of convex variational problems: applications from image processing to computational mechanics." ACM Transactions on Mathematical Software (TOMS) 46.3 (2020): 1-33.](https://doi.org/10.1145/3393881)
```
@article{bleyer2020automating,
  title={Automating the formulation and resolution of convex variational problems: applications from image processing to computational mechanics},
  author={Bleyer, Jeremy},
  journal={ACM Transactions on Mathematical Software (TOMS)},
  volume={46},
  number={3},
  pages={1--33},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```

- [Bleyer, Jeremy. "Applications of conic programming in non-smooth mechanics." Journal of Optimization Theory and Applications (2022): 1-33.](https://link.springer.com/article/10.1007/s10957-022-02105-z)
```
@article{bleyer2022applications,
  title={Applications of conic programming in non-smooth mechanics},
  author={Bleyer, Jeremy},
  journal={Journal of Optimization Theory and Applications},
  pages={1--33},
  year={2022},
  publisher={Springer}
}
```

## About the author

[Jeremy Bleyer](https://bleyerj.github.io/) is a researcher in Solid and Structural Mechanics at [Laboratoire Navier](https://navier-lab.fr), a joint research  (UMR 8205) of [Ecole Nationale des Ponts et Chaussées](http://www.enpc.fr),
[Université Gustave Eiffel](https://www.univ-gustave-eiffel.fr/) and [CNRS](http://www.cnrs.fr).

[{fas}`at` jeremy.bleyer@enpc.fr](mailto:jeremy.bleyer@enpc.fr)

[{fab}`linkedin` jeremy-bleyer](http://www.linkedin.com/in/jérémy-bleyer-0aabb531)

<a href="https://orcid.org/0000-0001-8212-9921">
<img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_32x32.png" width="16" height="16" />
 0000-0001-8212-9921
</a>
