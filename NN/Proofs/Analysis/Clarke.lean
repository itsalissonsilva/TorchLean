/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Alisson M. Silva
-/

module

public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Mathlib.Data.EReal.Basic
public import Mathlib.Topology.Order.LiminfLimsup
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Calculus.LineDeriv.Basic
import Mathlib.Topology.Instances.EReal.Lemmas
public import Mathlib.Analysis.Convex.Hull
public import Mathlib.Topology.MetricSpace.Lipschitz

/-!
# Clarke generalized gradients

This module defines limiting gradients and Clarke's upper generalized directional derivative for
real-valued functions on real Hilbert spaces. The directional derivative is extended-real-valued,
so the definitions do not silently assume local Lipschitz continuity.

The main theorem proves directly that every limiting gradient satisfies Clarke's directional
inequalities. It does not use, or claim, the finite-dimensional representation of the Clarke
generalized gradient as a convex hull of reachable gradients.
-/

@[expose] public section

open Filter
open scoped Gradient Topology NNReal

namespace Proofs

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/--
`g` is a limiting gradient of `F` at `x` if ordinary gradients at differentiability points
converging to `x` converge to `g`.
-/
def IsLimitingGradient (F : E → ℝ) (x g : E) : Prop :=
  ∃ xn : ℕ → E,
    Tendsto xn atTop (nhds x) ∧
    (∀ n, DifferentiableAt ℝ F (xn n)) ∧
    Tendsto (fun n ↦ ∇ F (xn n)) atTop (nhds g)

/-- At a differentiability point, the ordinary gradient is a limiting gradient. -/
theorem isLimitingGradient_gradient_of_differentiableAt
    {F : E → ℝ} {x : E} (hF : DifferentiableAt ℝ F x) :
    IsLimitingGradient F x (∇ F x) := by
  exact ⟨fun _ ↦ x, tendsto_const_nhds, fun _ ↦ hF, tendsto_const_nhds⟩

/--
Clarke's upper generalized directional derivative

`F°(x; v) = limsupₙ→ₓ, t↓0⁺ (F(y + t v) - F(y)) / t`.

The codomain is `EReal`, allowing the definition without a boundedness or local Lipschitz
hypothesis.
-/
noncomputable def clarkeDirectionalDerivative (F : E → ℝ) (x v : E) : EReal :=
  limsup
    (fun p : E × ℝ ↦ (((F (p.1 + p.2 • v) - F p.1) / p.2 : ℝ) : EReal))
    ((nhds x) ×ˢ (nhdsWithin (0 : ℝ) (Set.Ioi 0)))

/--
The inequality-defined Clarke generalized gradient. No local Lipschitz assumption is built into
the definition; under the usual hypotheses it agrees with the standard Clarke subdifferential.
-/
def clarkeGradient (F : E → ℝ) (x : E) : Set E :=
  {g | ∀ v : E, ((inner ℝ g v : ℝ) : EReal) ≤ clarkeDirectionalDerivative F x v}

/--
Every limiting gradient satisfies the inequalities defining the Clarke generalized gradient.

For each direction, positive difference-quotient steps are selected at the differentiability
points witnessing the limiting gradient. The resulting diagonal sequence converges in the filter
used by `clarkeDirectionalDerivative`, making the limiting directional derivative a cluster value
of the defining quotient.
-/
theorem isLimitingGradient_mem_clarkeGradient
    {F : E → ℝ} {x g : E} (h : IsLimitingGradient F x g) :
    g ∈ clarkeGradient F x := by
  rcases h with ⟨xn, hxn, hdiff, hgrad⟩
  intro v
  let q : E → ℝ → ℝ := fun y t ↦ (F (y + t • v) - F y) / t
  have hq_tendsto (n : ℕ) :
      Tendsto (q (xn n)) (nhdsWithin (0 : ℝ) (Set.Ioi 0))
        (nhds ((fderiv ℝ F (xn n)) v)) := by
    have hline := ((hdiff n).hasFDerivAt.hasLineDerivAt v).tendsto_slope_zero_right
    simpa [q, div_eq_mul_inv, mul_comm] using hline
  have hchoose : ∀ n : ℕ, ∃ t : ℝ,
      dist (q (xn n) t) ((fderiv ℝ F (xn n)) v) < (1 : ℝ) / ((n : ℝ) + 1) ∧
        t ∈ Set.Ioo 0 ((1 : ℝ) / ((n : ℝ) + 1)) := by
    intro n
    have hclose := (Metric.tendsto_nhds.1 (hq_tendsto n))
      ((1 : ℝ) / ((n : ℝ) + 1)) Nat.one_div_pos_of_nat
    exact (hclose.and (Ioo_mem_nhdsGT Nat.one_div_pos_of_nat)).exists
  choose tn htn_close htn_interval using hchoose
  have hε : Tendsto (fun n : ℕ ↦ (1 : ℝ) / ((n : ℝ) + 1)) atTop (nhds 0) :=
    tendsto_one_div_add_atTop_nhds_zero_nat
  have htn : Tendsto tn atTop (nhds (0 : ℝ)) := by
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hε
      (fun n ↦ (htn_interval n).1.le) (fun n ↦ (htn_interval n).2.le)
  have htn_within : Tendsto tn atTop (nhdsWithin (0 : ℝ) (Set.Ioi 0)) :=
    tendsto_nhdsWithin_iff.2
      ⟨htn, Filter.Eventually.of_forall fun n ↦ (htn_interval n).1⟩
  have hderiv : Tendsto (fun n ↦ (fderiv ℝ F (xn n)) v) atTop
      (nhds (inner ℝ g v)) := by
    have hinner : Tendsto (fun n ↦ inner ℝ (∇ F (xn n)) v) atTop
        (nhds (inner ℝ g v)) :=
      hgrad.inner (𝕜 := ℝ) (tendsto_const_nhds : Tendsto (fun _ : ℕ ↦ v) atTop (nhds v))
    simpa only [inner_gradient_left] using hinner
  have hdist : Tendsto
      (fun n ↦ dist ((fderiv ℝ F (xn n)) v) (q (xn n) (tn n)))
      atTop (nhds 0) := by
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hε
      (fun _ ↦ dist_nonneg)
      (fun n ↦ by simpa only [dist_comm] using (htn_close n).le)
  have hquot : Tendsto (fun n ↦ q (xn n) (tn n)) atTop (nhds (inner ℝ g v)) :=
    hderiv.congr_dist hdist
  let Q : E × ℝ → EReal := fun p ↦ (q p.1 p.2 : EReal)
  have hp : Tendsto (fun n ↦ (xn n, tn n)) atTop
      ((nhds x) ×ˢ (nhdsWithin (0 : ℝ) (Set.Ioi 0))) :=
    hxn.prodMk htn_within
  have hcluster_seq : MapClusterPt ((inner ℝ g v : ℝ) : EReal) atTop
      (Q ∘ fun n ↦ (xn n, tn n)) := by
    apply Filter.Tendsto.mapClusterPt
    change Tendsto (fun n ↦ (q (xn n) (tn n) : EReal)) atTop
      (nhds ((inner ℝ g v : ℝ) : EReal))
    exact EReal.tendsto_coe.2 hquot
  have hcluster : MapClusterPt ((inner ℝ g v : ℝ) : EReal)
      ((nhds x) ×ˢ (nhdsWithin (0 : ℝ) (Set.Ioi 0))) Q :=
    hcluster_seq.of_comp hp
  exact hcluster.le_limsup

omit [CompleteSpace E] in
/-- The Clarke set is convex, including when a directional value is infinite. -/
theorem convex_clarkeGradient (F : E → ℝ) (x : E) :
    Convex ℝ (clarkeGradient F x) := by
  rw [show clarkeGradient F x = ⋂ v : E,
      {g : E | ((inner ℝ g v : ℝ) : EReal) ≤ clarkeDirectionalDerivative F x v} by
    ext g
    simp [clarkeGradient]]
  apply convex_iInter
  intro v
  cases h : clarkeDirectionalDerivative F x v using EReal.rec with
  | bot =>
      convert (convex_empty : Convex ℝ (∅ : Set E)) using 1
      ext g
      simp
  | coe r =>
      intro a ha b hb s t hs ht hst
      simp only [Set.mem_ofPred_eq, EReal.coe_le_coe_iff] at ha hb ⊢
      rw [inner_add_left, inner_smul_left, inner_smul_left]
      simp only [starRingEnd_apply, star_trivial]
      calc
        s * inner ℝ a v + t * inner ℝ b v ≤ s * r + t * r :=
          add_le_add (mul_le_mul_of_nonneg_left ha hs)
            (mul_le_mul_of_nonneg_left hb ht)
        _ = r := by rw [← add_mul, hst, one_mul]
  | top =>
      convert (convex_univ : Convex ℝ (Set.univ : Set E)) using 1
      ext g
      simp

/-- Convexifying reachable gradients preserves the directional inequalities. -/
theorem convexHull_isLimitingGradient_subset_clarkeGradient (F : E → ℝ) (x : E) :
    convexHull ℝ {g | IsLimitingGradient F x g} ⊆ clarkeGradient F x := by
  exact convexHull_min (fun _ hg ↦ isLimitingGradient_mem_clarkeGradient hg)
    (convex_clarkeGradient F x)

omit [CompleteSpace E] in
/-- A local Lipschitz constant bounds both sides of Clarke's directional derivative. -/
theorem clarkeDirectionalDerivative_bounds_of_lipschitzOnWith
    {F : E → ℝ} {x : E} {K : ℝ≥0} {s : Set E}
    (hs : s ∈ nhds x) (hF : LipschitzOnWith K F s) (v : E) :
    ((-((K : ℝ) * ‖v‖) : ℝ) : EReal) ≤ clarkeDirectionalDerivative F x v ∧
      clarkeDirectionalDerivative F x v ≤ (((K : ℝ) * ‖v‖ : ℝ) : EReal) := by
  let l := (nhds x) ×ˢ (nhdsWithin (0 : ℝ) (Set.Ioi 0))
  have hy : Tendsto (fun p : E × ℝ ↦ p.1) l (nhds x) := tendsto_fst
  have ht : Tendsto (fun p : E × ℝ ↦ p.2) l (nhds (0 : ℝ)) :=
    tendsto_snd.mono_right nhdsWithin_le_nhds
  have hshift : Tendsto (fun p : E × ℝ ↦ p.1 + p.2 • v) l (nhds x) := by
    simpa using hy.add (ht.smul_const v)
  have hpos : ∀ᶠ p : E × ℝ in l, 0 < p.2 :=
    tendsto_snd.eventually self_mem_nhdsWithin
  have hbounds : ∀ᶠ p : E × ℝ in l,
      -((K : ℝ) * ‖v‖) ≤ (F (p.1 + p.2 • v) - F p.1) / p.2 ∧
        (F (p.1 + p.2 • v) - F p.1) / p.2 ≤ (K : ℝ) * ‖v‖ := by
    filter_upwards [hy.eventually hs, hshift.eventually hs, hpos] with p hp hq ht
    have hb := hF.dist_le_mul _ hq _ hp
    rw [Real.dist_eq, dist_eq_norm, add_sub_cancel_left, norm_smul,
      Real.norm_eq_abs, abs_of_pos ht] at hb
    have ha : |(F (p.1 + p.2 • v) - F p.1) / p.2| ≤ (K : ℝ) * ‖v‖ := by
      rw [abs_div, abs_of_pos ht, div_le_iff₀ ht]
      nlinarith [hb]
    exact abs_le.1 ha
  constructor
  · exact le_trans (le_liminf_of_le (h := hbounds.mono fun _ hp ↦ EReal.coe_le_coe hp.1))
      (liminf_le_limsup (u := fun p : E × ℝ ↦
        (((F (p.1 + p.2 • v) - F p.1) / p.2 : ℝ) : EReal)) (f := l))
  · exact limsup_le_of_le (h := hbounds.mono fun _ hp ↦ EReal.coe_le_coe hp.2)

omit [CompleteSpace E] in
/-- Every Clarke gradient has norm at most any Lipschitz constant valid near the point. -/
theorem norm_le_of_mem_clarkeGradient
    {F : E → ℝ} {x g : E} {K : ℝ≥0} {s : Set E}
    (hs : s ∈ nhds x) (hF : LipschitzOnWith K F s)
    (hg : g ∈ clarkeGradient F x) : ‖g‖ ≤ (K : ℝ) := by
  have h := (hg g).trans (clarkeDirectionalDerivative_bounds_of_lipschitzOnWith hs hF g).2
  have hreal : inner ℝ g g ≤ (K : ℝ) * ‖g‖ := EReal.coe_le_coe_iff.1 h
  rw [real_inner_self_eq_norm_sq] at hreal
  nlinarith [norm_nonneg g, K.coe_nonneg]

/-- The directional construction recovers the ordinary directional derivative of the identity. -/
theorem clarkeDirectionalDerivative_id (x v : ℝ) :
    clarkeDirectionalDerivative id x v = (v : EReal) := by
  have hpos : ∀ᶠ p : ℝ × ℝ in
      (nhds x) ×ˢ (nhdsWithin (0 : ℝ) (Set.Ioi 0)), 0 < p.2 :=
    tendsto_snd.eventually self_mem_nhdsWithin
  unfold clarkeDirectionalDerivative
  calc
    _ = limsup (fun _ : ℝ × ℝ ↦ (v : EReal))
        ((nhds x) ×ˢ (nhdsWithin (0 : ℝ) (Set.Ioi 0))) := by
      apply limsup_congr
      filter_upwards [hpos] with p hp
      congr 1
      simp [smul_eq_mul, hp.ne']
    _ = _ := limsup_const _

end Proofs
