/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Alisson M. Silva
-/

module

public import NN.Proofs.Analysis.Clarke
public import NN.Proofs.Autograd.FDeriv.MlpMse
public import Mathlib.Analysis.InnerProductSpace.ProdL2
public import Mathlib.Analysis.Calculus.ContDiff.RCLike

/-!
# Generalized backpropagation for a two-layer ReLU MLP

`NN.Proofs.Autograd.FDeriv.MlpMse` identifies the usual four parameter-gradient formulas with the
ordinary Fréchet gradient when every hidden preactivation is nonzero. This module extends that
result to kink points for one fixed input-target pair.

The four parameter blocks are bundled in an `L²` product Hilbert space. At a kink, the first-layer
bias is perturbed so each zero preactivation approaches from the negative side. This matches
TorchLean's convention `reluDerivSpec 0 = 0`, while preserving the network output and all four
closed-form backpropagation blocks. Consequently, the selected tuple is a limiting gradient and
belongs to the inequality-defined Clarke generalized gradient.

This is a theorem about the real-valued specification and closed-form formulas in `MlpMse`; it does
not identify a floating-point runtime execution with a Clarke gradient.
-/

@[expose] public section

open Filter
open scoped Gradient Topology NNReal

namespace Proofs
namespace Autograd

open Spec _root_.TorchLean

noncomputable section

variable {inDim hidDim outDim : Nat}

private def hiddenBiasObjective
    (W1 : Mat hidDim inDim) (W2 : Mat outDim hidDim)
    (b2 : Vec outDim) (x : Vec inDim) (t : Vec outDim) : Vec hidDim → ℝ :=
  fun b1 ↦ mse t (mlpVecMat W1 b1 W2 b2 x)

private def hiddenBiasBackprop
    (W1 : Mat hidDim inDim) (W2 : Mat outDim hidDim)
    (b2 : Vec outDim) (x : Vec inDim) (t : Vec outDim) (b1 : Vec hidDim) :
    Vec hidDim :=
  (reluDerivCLM (z1 W1 b1 x))
    ((matCLM (toMatrix W2)).adjoint (mseGrad (y W1 b1 W2 b2 x) t))

private def negativeKinkDirection (z : Vec hidDim) : Vec hidDim :=
  WithLp.toLp 2 fun i ↦ if z i = 0 then -1 else 0

@[simp] private theorem negativeKinkDirection_apply
    (z : Vec hidDim) (i : Fin hidDim) :
    negativeKinkDirection z i = if z i = 0 then -1 else 0 := rfl

private def negativeKinkPerturbation (z : Vec hidDim) (ε : ℝ) : Vec hidDim :=
  ε • negativeKinkDirection z

private def perturbHiddenBias
    (W1 : Mat hidDim inDim) (x : Vec inDim) (b1 : Vec hidDim) (ε : ℝ) : Vec hidDim :=
  b1 + negativeKinkPerturbation (z1 W1 b1 x) ε

@[simp] private theorem negativeKinkPerturbation_apply
    (z : Vec hidDim) (ε : ℝ) (i : Fin hidDim) :
    negativeKinkPerturbation z ε i = if z i = 0 then -ε else 0 := by
  by_cases h : z i = 0 <;>
    simp [negativeKinkPerturbation, negativeKinkDirection, h]

private theorem z1_perturbHiddenBias_apply
    (W1 : Mat hidDim inDim) (x : Vec inDim) (b1 : Vec hidDim) (ε : ℝ)
    (i : Fin hidDim) :
    z1 W1 (perturbHiddenBias W1 x b1 ε) x i =
      if z1 W1 b1 x i = 0 then -ε else z1 W1 b1 x i := by
  have hz :
      z1 W1 (perturbHiddenBias W1 x b1 ε) x =
        z1 W1 b1 x + negativeKinkPerturbation (z1 W1 b1 x) ε := by
    ext j
    simp only [z1, affineMat, affine, perturbHiddenBias]
    abel_nf
  rw [hz]
  by_cases h : z1 W1 b1 x i = 0 <;> simp [h]

private theorem z1_perturbHiddenBias_ne_zero
    (W1 : Mat hidDim inDim) (x : Vec inDim) (b1 : Vec hidDim) {ε : ℝ}
    (hε : 0 < ε) :
    ∀ i : Fin hidDim, z1 W1 (perturbHiddenBias W1 x b1 ε) x i ≠ 0 := by
  intro i
  rw [z1_perturbHiddenBias_apply]
  by_cases h : z1 W1 b1 x i = 0
  · simp [h, ne_of_gt hε]
  · simp [h]

@[simp] private theorem reluVec_apply (z : Vec hidDim) (i : Fin hidDim) :
    reluVec z i = Activation.Math.reluSpec (z i) := rfl

private theorem reluVec_z1_perturbHiddenBias
    (W1 : Mat hidDim inDim) (x : Vec inDim) (b1 : Vec hidDim) {ε : ℝ}
    (hε : 0 < ε) :
    reluVec (z1 W1 (perturbHiddenBias W1 x b1 ε) x) =
      reluVec (z1 W1 b1 x) := by
  ext i
  simp only [reluVec_apply, z1_perturbHiddenBias_apply]
  by_cases h : z1 W1 b1 x i = 0
  · have hneg : -ε ≤ 0 := neg_nonpos.mpr hε.le
    simp [h, Activation.Math.reluSpec_eq_max, hneg]
  · simp [h]

private theorem reluDerivCLM_z1_perturbHiddenBias
    (W1 : Mat hidDim inDim) (x : Vec inDim) (b1 : Vec hidDim) {ε : ℝ}
    (hε : 0 < ε) :
    reluDerivCLM (z1 W1 (perturbHiddenBias W1 x b1 ε) x) =
      reluDerivCLM (z1 W1 b1 x) := by
  ext dx i
  simp only [reluDerivCLM_apply, z1_perturbHiddenBias_apply]
  by_cases h : z1 W1 b1 x i = 0
  · have hnotpos : ¬ -ε > 0 := not_lt_of_ge (neg_nonpos.mpr hε.le)
    simp [h, Activation.Math.reluDerivSpec, hnotpos]
  · simp [h]

private theorem y_perturbHiddenBias
    (W1 : Mat hidDim inDim) (W2 : Mat outDim hidDim)
    (b2 : Vec outDim) (x : Vec inDim) (b1 : Vec hidDim) {ε : ℝ}
    (hε : 0 < ε) :
    y W1 (perturbHiddenBias W1 x b1 ε) W2 b2 x = y W1 b1 W2 b2 x := by
  simpa [y, mlpVecMat, z1] using congrArg (affineMat W2 b2)
    (reluVec_z1_perturbHiddenBias W1 x b1 hε)

private def hiddenBiasApproximants
    (W1 : Mat hidDim inDim) (x : Vec inDim) (b1 : Vec hidDim) :
    ℕ → Vec hidDim :=
  fun n ↦ perturbHiddenBias W1 x b1 ((1 : ℝ) / ((n : ℝ) + 1))

private theorem hiddenBiasApproximants_tendsto
    (W1 : Mat hidDim inDim) (x : Vec inDim) (b1 : Vec hidDim) :
    Tendsto (hiddenBiasApproximants W1 x b1) atTop (nhds b1) := by
  have hε : Tendsto (fun n : ℕ ↦ (1 : ℝ) / ((n : ℝ) + 1)) atTop (nhds 0) :=
    tendsto_one_div_add_atTop_nhds_zero_nat
  have hb : Tendsto (fun _ : ℕ ↦ b1) atTop (nhds b1) := tendsto_const_nhds
  change Tendsto
    (fun n : ℕ ↦ b1 + ((1 : ℝ) / ((n : ℝ) + 1)) •
      negativeKinkDirection (z1 W1 b1 x)) atTop (nhds b1)
  simpa using hb.add (hε.smul_const (negativeKinkDirection (z1 W1 b1 x)))

private theorem gradient_eq_adjoint_fderiv_one
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    (F : E → ℝ) (u : E) :
    ∇ F u = (fderiv ℝ F u).adjoint (1 : ℝ) := by
  refine ext_inner_right ℝ fun v ↦ ?_
  rw [inner_gradient_left]
  rw [ContinuousLinearMap.adjoint_inner_left]
  simp

private theorem differentiableAt_hiddenBiasObjective_of_noKinks
    (W1 : Mat hidDim inDim) (W2 : Mat outDim hidDim)
    (b2 : Vec outDim) (x : Vec inDim) (t : Vec outDim) (b1 : Vec hidDim)
    (h : ∀ i : Fin hidDim, z1 W1 b1 x i ≠ 0) :
    DifferentiableAt ℝ (hiddenBiasObjective W1 W2 b2 x t) b1 := by
  have hf := hasFDerivAt_mlp_wrt_b1 W1 b1 W2 b2 x h
  have hg := hasFDerivAt_mse t (mlpVecMat W1 b1 W2 b2 x)
  change DifferentiableAt ℝ
    (fun b1 : Vec hidDim ↦ mse t (mlpVecMat W1 b1 W2 b2 x)) b1
  exact (hg.comp b1 hf).differentiableAt

private theorem gradient_hiddenBiasObjective_eq_backprop_of_noKinks
    (W1 : Mat hidDim inDim) (W2 : Mat outDim hidDim)
    (b2 : Vec outDim) (x : Vec inDim) (t : Vec outDim) (b1 : Vec hidDim)
    (h : ∀ i : Fin hidDim, z1 W1 b1 x i ≠ 0) :
    ∇ (hiddenBiasObjective W1 W2 b2 x t) b1 =
      hiddenBiasBackprop W1 W2 b2 x t b1 := by
  rw [gradient_eq_adjoint_fderiv_one]
  change (fderiv ℝ (fun b1 : Vec hidDim ↦ mse t (mlpVecMat W1 b1 W2 b2 x)) b1).adjoint
      (1 : ℝ) =
    (reluDerivCLM (z1 W1 b1 x))
      ((matCLM (toMatrix W2)).adjoint (mseGrad (y W1 b1 W2 b2 x) t))
  exact grad_b1_mse W1 b1 W2 b2 x t h

/-- Matrix-vector evaluation as a continuous bilinear map on the parameter Hilbert spaces. -/
def matVecBilinear {m n : Nat} : Mat m n →L[ℝ] Vec n →L[ℝ] Vec m := by
  let L : Mat m n →ₗ[ℝ] (Vec n →L[ℝ] Vec m) :=
    { toFun := fun W ↦ matCLM (toMatrix W)
      map_add' := by
        intro A B
        ext x i
        simp [matCLM, toMatrix_add]
      map_smul' := by
        intro c A
        ext x i
        simp [matCLM, toMatrix_smul, smul_eq_mul] }
  exact ⟨L, LinearMap.continuous_of_finiteDimensional L⟩

@[simp] private theorem matVecBilinear_apply {m n : Nat} (W : Mat m n) (x : Vec n) :
    matVecBilinear W x = matCLM (toMatrix W) x := rfl

section ProductSlices

variable {E F : Type*}
  [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F]

private def l2Inl : E →L[ℝ] WithLp 2 (E × F) :=
  (WithLp.prodContinuousLinearEquiv 2 ℝ E F).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.inl ℝ E F)

private def l2Inr : F →L[ℝ] WithLp 2 (E × F) :=
  (WithLp.prodContinuousLinearEquiv 2 ℝ E F).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.inr ℝ E F)

@[simp] private theorem l2Inl_apply (x : E) :
    l2Inl (F := F) x = WithLp.toLp 2 (x, 0) := rfl

@[simp] private theorem l2Inr_apply (y : F) :
    l2Inr (E := E) y = WithLp.toLp 2 (0, y) := rfl

private theorem hasFDerivAt_l2FstSlice (f : WithLp 2 (E × F) → ℝ)
    (p : WithLp 2 (E × F)) (hf : DifferentiableAt ℝ f p) :
    HasFDerivAt (fun x : E ↦ f (WithLp.toLp 2 (x, p.snd)))
      ((fderiv ℝ f p).comp (l2Inl (F := F))) p.fst := by
  have hemb : HasFDerivAt (fun x : E ↦ WithLp.toLp 2 (x, p.snd))
      (l2Inl (F := F)) p.fst := by
    have hpair : HasFDerivAt (fun x : E ↦ (x, p.snd))
        (ContinuousLinearMap.inl ℝ E F) p.fst :=
      hasFDerivAt_prodMk_left p.fst p.snd
    simpa [l2Inl, Function.comp_def] using
      (WithLp.prodContinuousLinearEquiv 2 ℝ E F).symm.toContinuousLinearMap.hasFDerivAt.comp
        p.fst hpair
  convert hf.hasFDerivAt.comp p.fst hemb using 1
  all_goals rfl

private theorem hasFDerivAt_l2SndSlice (f : WithLp 2 (E × F) → ℝ)
    (p : WithLp 2 (E × F)) (hf : DifferentiableAt ℝ f p) :
    HasFDerivAt (fun y : F ↦ f (WithLp.toLp 2 (p.fst, y)))
      ((fderiv ℝ f p).comp (l2Inr (E := E))) p.snd := by
  have hemb : HasFDerivAt (fun y : F ↦ WithLp.toLp 2 (p.fst, y))
      (l2Inr (E := E)) p.snd := by
    have hpair : HasFDerivAt (fun y : F ↦ (p.fst, y))
        (ContinuousLinearMap.inr ℝ E F) p.snd :=
      hasFDerivAt_prodMk_right p.fst p.snd
    simpa [l2Inr, Function.comp_def] using
      (WithLp.prodContinuousLinearEquiv 2 ℝ E F).symm.toContinuousLinearMap.hasFDerivAt.comp
        p.snd hpair
  convert hf.hasFDerivAt.comp p.snd hemb using 1
  all_goals rfl

variable [CompleteSpace E] [CompleteSpace F]

private theorem gradient_l2FstSlice (f : WithLp 2 (E × F) → ℝ)
    (p : WithLp 2 (E × F)) (hf : DifferentiableAt ℝ f p) :
    (∇ f p).fst = ∇ (fun x : E ↦ f (WithLp.toLp 2 (x, p.snd))) p.fst := by
  refine ext_inner_right ℝ fun u ↦ ?_
  calc
    inner ℝ (∇ f p).fst u = inner ℝ (∇ f p) (l2Inl (F := F) u) := by
      change inner ℝ (∇ f p).fst u =
        inner ℝ (∇ f p).fst u + inner ℝ (∇ f p).snd 0
      simp only [inner_zero_right, add_zero]
    _ = (fderiv ℝ f p) (l2Inl (F := F) u) := by rw [inner_gradient_left]
    _ = (fderiv ℝ (fun x : E ↦ f (WithLp.toLp 2 (x, p.snd))) p.fst) u := by
      rw [(hasFDerivAt_l2FstSlice f p hf).fderiv]
      rfl
    _ = inner ℝ (∇ (fun x : E ↦ f (WithLp.toLp 2 (x, p.snd))) p.fst) u := by
      rw [inner_gradient_left]

private theorem gradient_l2SndSlice (f : WithLp 2 (E × F) → ℝ)
    (p : WithLp 2 (E × F)) (hf : DifferentiableAt ℝ f p) :
    (∇ f p).snd = ∇ (fun y : F ↦ f (WithLp.toLp 2 (p.fst, y))) p.snd := by
  refine ext_inner_right ℝ fun v ↦ ?_
  calc
    inner ℝ (∇ f p).snd v = inner ℝ (∇ f p) (l2Inr (E := E) v) := by
      change inner ℝ (∇ f p).snd v =
        inner ℝ (∇ f p).fst 0 + inner ℝ (∇ f p).snd v
      simp only [inner_zero_right, zero_add]
    _ = (fderiv ℝ f p) (l2Inr (E := E) v) := by rw [inner_gradient_left]
    _ = (fderiv ℝ (fun y : F ↦ f (WithLp.toLp 2 (p.fst, y))) p.snd) v := by
      rw [(hasFDerivAt_l2SndSlice f p hf).fderiv]
      rfl
    _ = inner ℝ (∇ (fun y : F ↦ f (WithLp.toLp 2 (p.fst, y))) p.snd) v := by
      rw [inner_gradient_left]

end ProductSlices

/--
The four parameter blocks of a two-layer MLP, equipped with the Hilbert product norm in the order
`(W1, b1, W2, b2)`.
-/
abbrev MlpMseParameters (inDim hidDim outDim : Nat) :=
  WithLp 2
    (Mat hidDim inDim ×
      WithLp 2 (Vec hidDim × WithLp 2 (Mat outDim hidDim × Vec outDim)))

/-- Bundle the four parameter blocks in the order `(W1, b1, W2, b2)`. -/
def packMlpMseParameters
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim) :
    MlpMseParameters inDim hidDim outDim :=
  WithLp.toLp 2 (W1, WithLp.toLp 2 (b1, WithLp.toLp 2 (W2, b2)))

/-- The scalar MSE objective for a fixed input and target, as a function of all parameters. -/
def mlpMseObjective (x : Vec inDim) (t : Vec outDim) :
    MlpMseParameters inDim hidDim outDim → ℝ :=
  fun p ↦ mse t (mlpVecMat p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x)

/--
The tuple of the four closed-form parameter-gradient blocks from `MlpMse`, ordered as
`(W1, b1, W2, b2)`.
-/
def mlpMseBackprop
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim) :
    MlpMseParameters inDim hidDim outDim :=
  let h := (reluDerivCLM (z1 W1 b1 x))
    ((matCLM (toMatrix W2)).adjoint (mseGrad (y W1 b1 W2 b2 x) t))
  let δ := mseGrad (y W1 b1 W2 b2 x) t
  packMlpMseParameters (outer h x) h (outer δ (a1 W1 b1 x)) δ

private theorem differentiableAt_mlpMseObjective_of_noKinks
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim)
    (h : ∀ i : Fin hidDim, z1 W1 b1 x i ≠ 0) :
    DifferentiableAt ℝ (mlpMseObjective x t) (packMlpMseParameters W1 b1 W2 b2) := by
  let p : MlpMseParameters inDim hidDim outDim := packMlpMseParameters W1 b1 W2 b2
  let projW1 : MlpMseParameters inDim hidDim outDim →L[ℝ] Mat hidDim inDim :=
    WithLp.fstL 2 ℝ _ _
  let projTail : MlpMseParameters inDim hidDim outDim →L[ℝ]
      WithLp 2 (Vec hidDim × WithLp 2 (Mat outDim hidDim × Vec outDim)) :=
    WithLp.sndL 2 ℝ _ _
  let projB1 : MlpMseParameters inDim hidDim outDim →L[ℝ] Vec hidDim :=
    (WithLp.fstL 2 ℝ _ _).comp projTail
  let projOutput : MlpMseParameters inDim hidDim outDim →L[ℝ]
      WithLp 2 (Mat outDim hidDim × Vec outDim) :=
    (WithLp.sndL 2 ℝ _ _).comp projTail
  let projW2 : MlpMseParameters inDim hidDim outDim →L[ℝ] Mat outDim hidDim :=
    (WithLp.fstL 2 ℝ _ _).comp projOutput
  let projB2 : MlpMseParameters inDim hidDim outDim →L[ℝ] Vec outDim :=
    (WithLp.sndL 2 ℝ _ _).comp projOutput
  have hW1 : DifferentiableAt ℝ
      (fun q : MlpMseParameters inDim hidDim outDim ↦ q.fst) p := by
    change DifferentiableAt ℝ projW1 p
    exact projW1.differentiableAt
  have hb1 : DifferentiableAt ℝ
      (fun q : MlpMseParameters inDim hidDim outDim ↦ q.snd.fst) p := by
    change DifferentiableAt ℝ projB1 p
    exact projB1.differentiableAt
  have hz : DifferentiableAt ℝ
      (fun q : MlpMseParameters inDim hidDim outDim ↦ z1 q.fst q.snd.fst x) p := by
    change DifferentiableAt ℝ (fun q : MlpMseParameters inDim hidDim outDim ↦
      (matApplyLin x) q.fst + q.snd.fst) p
    exact ((matApplyLin x).differentiableAt.comp p hW1).add hb1
  have ha : DifferentiableAt ℝ
      (fun q : MlpMseParameters inDim hidDim outDim ↦
        reluVec (z1 q.fst q.snd.fst x)) p := by
    have hcomp := (hasFDerivAt_reluVec (x := z1 W1 b1 x) h).comp p hz.hasFDerivAt
    convert hcomp.differentiableAt using 1
    all_goals rfl
  have hW2 : DifferentiableAt ℝ
      (fun q : MlpMseParameters inDim hidDim outDim ↦ q.snd.snd.fst) p := by
    change DifferentiableAt ℝ projW2 p
    exact projW2.differentiableAt
  have hb2 : DifferentiableAt ℝ
      (fun q : MlpMseParameters inDim hidDim outDim ↦ q.snd.snd.snd) p := by
    change DifferentiableAt ℝ projB2 p
    exact projB2.differentiableAt
  have hy0 := (matVecBilinear (m := outDim) (n := hidDim)).hasFDerivAt_of_bilinear
    hW2.hasFDerivAt ha.hasFDerivAt
  have hy : DifferentiableAt ℝ
      (fun q : MlpMseParameters inDim hidDim outDim ↦
        mlpVecMat q.fst q.snd.fst q.snd.snd.fst q.snd.snd.snd x) p := by
    have hsum := hy0.add hb2.hasFDerivAt
    convert hsum.differentiableAt using 1
    funext q
    simp [mlpVecMat, z1, affineMat, affine, matVecBilinear_apply]
  have hmse := (hasFDerivAt_mse t (mlpVecMat W1 b1 W2 b2 x)).differentiableAt
  have hcomp := hmse.comp p (by simpa [p, packMlpMseParameters] using hy)
  convert hcomp using 1
  all_goals rfl

private theorem gradient_firstWeightsObjective_eq_backprop_of_noKinks
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim)
    (h : ∀ i : Fin hidDim, z1 W1 b1 x i ≠ 0) :
    ∇ (fun W1' : Mat hidDim inDim ↦ mse t (mlpVecMat W1' b1 W2 b2 x)) W1 =
      outer (hiddenBiasBackprop W1 W2 b2 x t b1) x := by
  rw [gradient_eq_adjoint_fderiv_one]
  exact grad_W1_mse W1 b1 W2 b2 x t h

private theorem gradient_secondWeightsObjective_eq_backprop
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim) :
    ∇ (fun W2' : Mat outDim hidDim ↦ mse t (mlpVecMat W1 b1 W2' b2 x)) W2 =
      outer (mseGrad (y W1 b1 W2 b2 x) t) (a1 W1 b1 x) := by
  rw [gradient_eq_adjoint_fderiv_one]
  exact grad_W2_mse W1 b1 W2 b2 x t

private theorem gradient_outputBiasObjective_eq_backprop
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim) :
    ∇ (fun b2' : Vec outDim ↦ mse t (mlpVecMat W1 b1 W2 b2' x)) b2 =
      mseGrad (y W1 b1 W2 b2 x) t := by
  rw [gradient_eq_adjoint_fderiv_one]
  exact grad_b2_mse W1 b1 W2 b2 x t

/--
Away from hidden ReLU kinks, the ordinary joint gradient is the tuple of TorchLean's four
closed-form parameter-gradient blocks.
-/
theorem gradient_mlpMseObjective_eq_backprop_of_noKinks
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim)
    (h : ∀ i : Fin hidDim, z1 W1 b1 x i ≠ 0) :
    ∇ (mlpMseObjective x t) (packMlpMseParameters W1 b1 W2 b2) =
      mlpMseBackprop W1 b1 W2 b2 x t := by
  let p := packMlpMseParameters W1 b1 W2 b2
  let F : MlpMseParameters inDim hidDim outDim → ℝ := mlpMseObjective x t
  let Fr : WithLp 2 (Vec hidDim × WithLp 2 (Mat outDim hidDim × Vec outDim)) → ℝ :=
    fun r ↦ F (WithLp.toLp 2 (W1, r))
  let Fo : WithLp 2 (Mat outDim hidDim × Vec outDim) → ℝ :=
    fun o ↦ Fr (WithLp.toLp 2 (b1, o))
  have hF : DifferentiableAt ℝ F p := by
    simpa [F, p] using differentiableAt_mlpMseObjective_of_noKinks W1 b1 W2 b2 x t h
  have hFr : DifferentiableAt ℝ Fr p.snd := by
    simpa [Fr, p, F, packMlpMseParameters] using
      (hasFDerivAt_l2SndSlice F p hF).differentiableAt
  have hFo : DifferentiableAt ℝ Fo p.snd.snd := by
    simpa [Fo, Fr, p, packMlpMseParameters] using
      (hasFDerivAt_l2SndSlice Fr p.snd hFr).differentiableAt
  rw [WithLp.ext_iff]
  apply Prod.ext
  · calc
      (∇ F p).fst = ∇ (fun W1' ↦ F (WithLp.toLp 2 (W1', p.snd))) W1 := by
        simpa [p, packMlpMseParameters] using gradient_l2FstSlice F p hF
      _ = outer (hiddenBiasBackprop W1 W2 b2 x t b1) x := by
        simpa [F, mlpMseObjective, p, packMlpMseParameters] using
          gradient_firstWeightsObjective_eq_backprop_of_noKinks W1 b1 W2 b2 x t h
  · rw [WithLp.ext_iff]
    apply Prod.ext
    · calc
        (∇ F p).snd.fst = (∇ Fr p.snd).fst := by
          exact congrArg WithLp.fst (gradient_l2SndSlice F p hF)
        _ = ∇ (fun b1' ↦ Fr (WithLp.toLp 2 (b1', p.snd.snd))) b1 := by
          simpa [p, packMlpMseParameters] using gradient_l2FstSlice Fr p.snd hFr
        _ = hiddenBiasBackprop W1 W2 b2 x t b1 := by
          change ∇ (hiddenBiasObjective W1 W2 b2 x t) b1 =
            hiddenBiasBackprop W1 W2 b2 x t b1
          exact gradient_hiddenBiasObjective_eq_backprop_of_noKinks W1 W2 b2 x t b1 h
    · rw [WithLp.ext_iff]
      apply Prod.ext
      · calc
          (∇ F p).snd.snd.fst = (∇ Fr p.snd).snd.fst := by
            exact congrArg (fun r ↦ r.snd.fst) (gradient_l2SndSlice F p hF)
          _ = (∇ Fo p.snd.snd).fst := by
            exact congrArg WithLp.fst (gradient_l2SndSlice Fr p.snd hFr)
          _ = ∇ (fun W2' ↦ Fo (WithLp.toLp 2 (W2', p.snd.snd.snd))) W2 := by
            simpa [p, packMlpMseParameters] using gradient_l2FstSlice Fo p.snd.snd hFo
          _ = outer (mseGrad (y W1 b1 W2 b2 x) t) (a1 W1 b1 x) := by
            simpa [Fo, Fr, F, mlpMseObjective, p, packMlpMseParameters] using
              gradient_secondWeightsObjective_eq_backprop W1 b1 W2 b2 x t
      · calc
          (∇ F p).snd.snd.snd = (∇ Fr p.snd).snd.snd := by
            exact congrArg (fun r ↦ r.snd.snd) (gradient_l2SndSlice F p hF)
          _ = (∇ Fo p.snd.snd).snd := by
            exact congrArg WithLp.snd (gradient_l2SndSlice Fr p.snd hFr)
          _ = ∇ (fun b2' ↦ Fo (WithLp.toLp 2 (p.snd.snd.fst, b2'))) b2 := by
            simpa [p, packMlpMseParameters] using gradient_l2SndSlice Fo p.snd.snd hFo
          _ = mseGrad (y W1 b1 W2 b2 x) t := by
            simpa [Fo, Fr, F, mlpMseObjective, p, packMlpMseParameters] using
              gradient_outputBiasObjective_eq_backprop W1 b1 W2 b2 x t

private def mlpMseApproximants
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim) (x : Vec inDim) :
    ℕ → MlpMseParameters inDim hidDim outDim :=
  fun n ↦ packMlpMseParameters W1 (hiddenBiasApproximants W1 x b1 n) W2 b2

private theorem mlpMseApproximants_tendsto
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim) (x : Vec inDim) :
    Tendsto (mlpMseApproximants W1 b1 W2 b2 x) atTop
      (nhds (packMlpMseParameters W1 b1 W2 b2)) := by
  let o : WithLp 2 (Mat outDim hidDim × Vec outDim) := WithLp.toLp 2 (W2, b2)
  have hb := hiddenBiasApproximants_tendsto W1 x b1
  have ho : Tendsto (fun _ : ℕ ↦ o) atTop (nhds o) := tendsto_const_nhds
  have ht : Tendsto
      (fun n ↦ WithLp.toLp 2 (hiddenBiasApproximants W1 x b1 n, o)) atTop
      (nhds (WithLp.toLp 2 (b1, o))) :=
    (WithLp.prodContinuousLinearEquiv 2 ℝ (Vec hidDim)
      (WithLp 2 (Mat outDim hidDim × Vec outDim))).symm.continuousAt.tendsto.comp
        (hb.prodMk_nhds ho)
  have hW : Tendsto (fun _ : ℕ ↦ W1) atTop (nhds W1) := tendsto_const_nhds
  let e := WithLp.prodContinuousLinearEquiv 2 ℝ (Mat hidDim inDim)
    (WithLp 2 (Vec hidDim × WithLp 2 (Mat outDim hidDim × Vec outDim)))
  convert e.symm.continuousAt.tendsto.comp (hW.prodMk_nhds ht) using 1
  all_goals rfl

private theorem mlpMseBackprop_perturbHiddenBias
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim) {ε : ℝ} (hε : 0 < ε) :
    mlpMseBackprop W1 (perturbHiddenBias W1 x b1 ε) W2 b2 x t =
      mlpMseBackprop W1 b1 W2 b2 x t := by
  have hy := y_perturbHiddenBias W1 W2 b2 x b1 hε
  have ha : a1 W1 (perturbHiddenBias W1 x b1 ε) x = a1 W1 b1 x :=
    reluVec_z1_perturbHiddenBias W1 x b1 hε
  simp only [mlpMseBackprop]
  rw [reluDerivCLM_z1_perturbHiddenBias W1 x b1 hε, hy, ha]

/--
For the real two-layer affine--ReLU--affine MLP with MSE, the selected four-block backpropagation
tuple is a limiting gradient with respect to all trainable parameters, including at ReLU kinks.
-/
theorem mlpMseBackprop_isLimitingGradient
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim) :
    IsLimitingGradient (mlpMseObjective x t) (packMlpMseParameters W1 b1 W2 b2)
      (mlpMseBackprop W1 b1 W2 b2 x t) := by
  refine ⟨mlpMseApproximants W1 b1 W2 b2 x,
    mlpMseApproximants_tendsto W1 b1 W2 b2 x, ?_, ?_⟩
  · intro n
    apply differentiableAt_mlpMseObjective_of_noKinks
    apply z1_perturbHiddenBias_ne_zero
    exact Nat.one_div_pos_of_nat
  · have hgrad : ∀ n : ℕ,
        ∇ (mlpMseObjective x t) (mlpMseApproximants W1 b1 W2 b2 x n) =
          mlpMseBackprop W1 b1 W2 b2 x t := by
      intro n
      change ∇ (mlpMseObjective x t)
        (packMlpMseParameters W1 (hiddenBiasApproximants W1 x b1 n) W2 b2) =
          mlpMseBackprop W1 b1 W2 b2 x t
      rw [gradient_mlpMseObjective_eq_backprop_of_noKinks]
      · apply mlpMseBackprop_perturbHiddenBias
        exact Nat.one_div_pos_of_nat
      · apply z1_perturbHiddenBias_ne_zero
        exact Nat.one_div_pos_of_nat
    exact tendsto_const_nhds.congr'
      (Filter.Eventually.of_forall fun n ↦ (hgrad n).symm)

/--
The selected four-block backpropagation tuple belongs unconditionally to the inequality-defined
Clarke generalized gradient of the real MSE objective.
-/
theorem mlpMseBackprop_mem_clarkeGradient
    (W1 : Mat hidDim inDim) (b1 : Vec hidDim)
    (W2 : Mat outDim hidDim) (b2 : Vec outDim)
    (x : Vec inDim) (t : Vec outDim) :
    mlpMseBackprop W1 b1 W2 b2 x t ∈
      clarkeGradient (mlpMseObjective x t) (packMlpMseParameters W1 b1 W2 b2) := by
  exact isLimitingGradient_mem_clarkeGradient
    (mlpMseBackprop_isLimitingGradient W1 b1 W2 b2 x t)

end

/-- Coordinatewise ReLU is nonexpansive for TorchLean's Euclidean norm. -/
theorem lipschitzWith_reluVec {n : Nat} : LipschitzWith 1 (reluVec (n := n)) := by
  apply LipschitzWith.of_dist_le_mul
  intro a b
  simp only [NNReal.coe_one, one_mul, dist_eq_norm]
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).1
  simp only [EuclideanSpace.real_norm_sq_eq]
  apply Finset.sum_le_sum
  intro i _
  have h := (LipschitzWith.id.max (LipschitzWith.const (0 : ℝ))).dist_le_mul (a i) (b i)
  have h' : |reluVec a i - reluVec b i| ≤ |a i - b i| := by
    by_cases ha : a i = 0 <;> by_cases hb : b i = 0 <;>
      simp_all [reluVec, reluFun, euclideanEquiv, Activation.Math.reluSpec, Real.dist_eq]
  simpa only [PiLp.sub_apply, Real.norm_eq_abs, sq_abs] using
    (sq_le_sq₀ (abs_nonneg _) (abs_nonneg _)).2 h'

variable {inDim hidDim outDim : Nat}

/-- The joint parameter objective is locally Lipschitz, also at ReLU kinks. -/
theorem locallyLipschitz_mlpMseObjective (x : Vec inDim) (t : Vec outDim) :
    LocallyLipschitz (mlpMseObjective (hidDim := hidDim) x t) := by
  let P := MlpMseParameters inDim hidDim outDim
  let w1 : P →L[ℝ] Mat hidDim inDim := WithLp.fstL 2 ℝ _ _
  let tail : P →L[ℝ] WithLp 2 (Vec hidDim × WithLp 2 (Mat outDim hidDim × Vec outDim)) :=
    WithLp.sndL 2 ℝ _ _
  let b1 : P →L[ℝ] Vec hidDim := (WithLp.fstL 2 ℝ _ _).comp tail
  let out : P →L[ℝ] WithLp 2 (Mat outDim hidDim × Vec outDim) :=
    (WithLp.sndL 2 ℝ _ _).comp tail
  let w2 : P →L[ℝ] Mat outDim hidDim := (WithLp.fstL 2 ℝ _ _).comp out
  let b2 : P →L[ℝ] Vec outDim := (WithLp.sndL 2 ℝ _ _).comp out
  let z : P →L[ℝ] Vec hidDim := (matApplyLin x).comp w1 + b1
  have ha : LocallyLipschitz (fun p : P ↦ reluVec (z p)) :=
    lipschitzWith_reluVec.locallyLipschitz.comp z.lipschitzWith.locallyLipschitz
  have hmul : ContDiff ℝ 1 (fun q : Mat outDim hidDim × Vec hidDim ↦
      matVecBilinear q.1 q.2) :=
    (matVecBilinear.contDiff.comp contDiff_fst).clm_apply contDiff_snd
  have hy0 := hmul.locallyLipschitz.comp (w2.lipschitzWith.locallyLipschitz.prodMk ha)
  have hadd : ContDiff ℝ 1 (fun q : Vec outDim × Vec outDim ↦ q.1 + q.2) :=
    contDiff_fst.add contDiff_snd
  have hy := hadd.locallyLipschitz.comp (hy0.prodMk b2.lipschitzWith.locallyLipschitz)
  have hmse : ContDiff ℝ 1 (mse t) :=
    contDiff_const.mul ((contDiff_id.sub contDiff_const).norm_sq (𝕜 := ℝ))
  convert hmse.locallyLipschitz.comp hy using 1
  funext p
  simp [mlpMseObjective, mlpVecMat, affineMat, affine, z, w1, b1, tail, w2, b2, out,
    matVecBilinear_apply, ContinuousLinearMap.comp_apply, WithLp.fstL, WithLp.sndL,
    matApplyLin, matApplyLM, matCLM]
  rfl

/-- One local constant controls every generalized direction and every Clarke gradient. -/
theorem mlpMseObjective_clarke_bounds
    (x : Vec inDim) (t : Vec outDim) (p : MlpMseParameters inDim hidDim outDim) :
    ∃ K : ℝ≥0,
      (∀ v, ((-((K : ℝ) * ‖v‖) : ℝ) : EReal) ≤
          clarkeDirectionalDerivative (mlpMseObjective x t) p v ∧
        clarkeDirectionalDerivative (mlpMseObjective x t) p v ≤
          (((K : ℝ) * ‖v‖ : ℝ) : EReal)) ∧
      (∀ g ∈ clarkeGradient (mlpMseObjective x t) p, ‖g‖ ≤ (K : ℝ)) := by
  obtain ⟨K, s, hs, hF⟩ := locallyLipschitz_mlpMseObjective x t p
  exact ⟨K, fun v ↦ clarkeDirectionalDerivative_bounds_of_lipschitzOnWith hs hF v,
    fun _ hg ↦ norm_le_of_mem_clarkeGradient hs hF hg⟩

/-- At every parameter, the selected vector belongs to a nonempty compact convex Clarke set. -/
theorem mlpMseObjective_clarke_geometry
    (x : Vec inDim) (t : Vec outDim) (p : MlpMseParameters inDim hidDim outDim) :
    (clarkeGradient (mlpMseObjective x t) p).Nonempty ∧
      IsCompact (clarkeGradient (mlpMseObjective x t) p) ∧
      Convex ℝ (clarkeGradient (mlpMseObjective x t) p) := by
  obtain ⟨K, s, hs, hF⟩ := locallyLipschitz_mlpMseObjective x t p
  refine ⟨?_, isCompact_clarkeGradient_of_lipschitzOnWith hs hF,
    convex_clarkeGradient _ _⟩
  obtain ⟨W1, b1, W2, b2⟩ := p
  exact ⟨_, mlpMseBackprop_mem_clarkeGradient W1 b1 W2 b2 x t⟩

end Autograd
end Proofs
