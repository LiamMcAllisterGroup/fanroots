# The secondary fan

This chapter collects the mathematical background on the secondary fan that
underlies **fanroots**.  Readers already comfortable with toric geometry may
skim to {ref}`sec-kaehler-glsm` and {ref}`sec-intersection-numbers` for the
package-specific conventions.

## Heights and triangulations

Let $\mathcal{A} = \{v_1, \dots, v_N\} \subset \mathbb{R}^d$ be a point or
vector configuration with $N$ elements.  A *height vector* $h \in \mathbb{R}^N$
assigns a real "lift" to each element: point $v_i$ is lifted to
$(v_i, h_i) \in \mathbb{R}^{d+1}$.  The lower convex hull of the lifted
configuration projects back to $\mathbb{R}^d$ as a polyhedral subdivision of
$\mathrm{conv}(\mathcal{A})$, called the *regular subdivision* induced by $h$.

When every maximal cell of the subdivision is a simplex *and* every element of
$\mathcal{A}$ appears as a vertex of some cell, the subdivision is a
*fine regular triangulation* (FRT).  FRTs are the input used by
[CYTools](https://github.com/LiamMcAllisterGroup/cytools) to construct
the toric variety $X_T$ associated to a triangulation $T$.

The set of heights that induce the same triangulation $T$ is a convex rational
polyhedral cone, the *secondary cone* $\Sigma(T)$.  Concretely, $h$ belongs to
$\Sigma(T)$ if and only if every *circuit* $C \subset \mathcal{A}$ of the
configuration satisfies its corresponding linear inequality.  A circuit is a
minimally affinely dependent subset; each circuit contributes one inequality
(one for each of its two orientations), so the secondary cone is the
intersection of finitely many half-spaces:

$$
\Sigma(T) \;=\; \bigl\{\, h \in \mathbb{R}^N \;\big|\;
  \lambda_C \cdot h \;\ge\; 0 \quad \forall\, C \in \mathcal{C}(T) \,\bigr\},
$$

where $\lambda_C \in \mathbb{R}^N$ is the coefficient vector of the circuit
inequality associated to circuit $C$ and triangulation $T$, and
$\mathcal{C}(T)$ denotes the set of relevant circuit constraints for $T$.

(sec-secondary-cones)=
## Secondary cones and chambers

The collection of all secondary cones, together with their faces, forms a
complete fan in $\mathbb{R}^N$ called the *secondary fan* $\Sigma(\mathcal{A})$.
The interiors of the maximal cones are the *chambers*: open convex regions of
height space in which the induced triangulation is constant.

Two properties are central to **fanroots**:

**Piecewise-constant geometry.**
Any quantity that depends on the triangulation — intersection numbers
$\kappa_{ijk}$, Stanley-Reisner ideal generators, Mori cone generators — is
constant inside each chamber and changes discontinuously at chamber walls
(codimension-1 faces of the fan).  The function to be optimized may therefore
be smooth within each chamber while being only piecewise-smooth globally.

**Exponential chamber count.**
The number of FRT chambers grows roughly exponentially with $N$
[arXiv:2008.01730](https://arxiv.org/abs/2008.01730),
[arXiv:2309.10855](https://arxiv.org/abs/2309.10855),
[arXiv:2602.16909](https://arxiv.org/abs/2602.16909).
At the scales relevant for string-theoretic applications ($N \approx 10$--$50$,
and up to $N \approx 200$ in practice), direct enumeration of all chambers is
infeasible.  **fanroots** therefore never enumerates chambers globally; all
operations are purely local, working in the chamber containing the current
point and its immediate neighbors.

(sec-kaehler-glsm)=
## Kähler parameters and GLSM

The Gauged Linear Sigma Model (GLSM) charge matrix encodes the linear
equivalences among the toric divisors.  For a triangulation $T$ giving a
Calabi–Yau hypersurface with $h^{1,1} = r$, the GLSM matrix

$$
Q \;\in\; \mathbb{R}^{r \times N}
$$

defines a surjective linear map from height space to Kähler parameter space:

$$
t \;=\; Q\, h, \qquad t \in \mathbb{R}^r.
$$

The *Kähler cone* of $X_T$ is the image of the secondary cone $\Sigma(T)$ under
$Q$.  A point $h$ in the interior of $\Sigma(T)$ maps to a point $t$ in the
interior of the Kähler cone, ensuring the associated Kähler class is
geometrically valid (ample line bundle).

**fanroots** works natively in height space $h$.  The corresponding Kähler
parameters are accessible at any time via the property

```python
optimizer.kahler   # returns t = GLSM @ h, shape (h11,)
```

and the current Kähler-cone representative $\tau$ (divisor volumes, see below)
via `optimizer.tau`.

(sec-intersection-numbers)=
## Intersection numbers

For a smooth toric variety $X_T$ associated to triangulation $T$, the triple
intersection numbers $\kappa_{ijk}$ (with $i,j,k = 1, \dots, h^{1,1}$) are
rational numbers determined entirely by the combinatorics of $T$.  They enter
physical quantities through the intersection form:

$$
\tau_i \;=\; \frac{1}{2}\,\kappa_{ijk}\, t^j\, t^k,
$$

where $\tau_i$ is the (real) volume of the $i$-th basis divisor and Einstein
summation is implied over $j$ and $k$.

In **fanroots**, intersection numbers are retrieved from CYTools via

```python
triang.intersection_numbers(in_basis=True, pushed_down=True, as_np_array=True)
```

and stored internally as a dense array of shape $(h^{1,1}, h^{1,1}, h^{1,1})$.
They are cached for the current triangulation and automatically recomputed
whenever the optimiser crosses a chamber wall — that is, whenever the
triangulation changes.  The non-zero values and their indices are accessible via
`optimizer.kappa_nz()` and `optimizer.kappa_vals()`, and the full dense tensor
via `optimizer.kappa`.
