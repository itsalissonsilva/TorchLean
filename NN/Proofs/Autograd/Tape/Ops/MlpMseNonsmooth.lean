/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Alisson M. Silva
-/

module

public import NN.Proofs.Autograd.FDeriv.MlpMseNonsmooth
public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# A real-valued MLP/MSE tape at ReLU kinks

The fused loss node below supplies its JVP by composing the affine, selected ReLU, affine,
and loss linearizations. Its VJP is the adjoint of that linearization, not the closed-form
parameter tuple. The theorem independently identifies `Graph.backpropVec` with that tuple.
The encoding is an arbitrary linear isometry, so no particular flattening layout is assumed.
This graph is a specification-level tape; it does not identify a separately lowered runtime graph.
-/

@[expose] public section

namespace Proofs.Autograd

open Spec TorchLean
open TapeNodes
open scoped Gradient

noncomputable section

variable {i h o : Nat}

/-- The selected joint linearization obtained by following the forward computation. -/
def mlpMseSelectedLinearization (x : Vec i) (t : Vec o) (p : MlpMseParameters i h o) :
    MlpMseParameters i h o →L[ℝ] ℝ := by
  let w1 : MlpMseParameters i h o →L[ℝ] Mat h i := WithLp.fstL 2 ℝ _ _
  let tail : MlpMseParameters i h o →L[ℝ] WithLp 2 (Vec h × WithLp 2 (Mat o h × Vec o)) :=
    WithLp.sndL 2 ℝ _ _
  let b1 : MlpMseParameters i h o →L[ℝ] Vec h := (WithLp.fstL 2 ℝ _ _).comp tail
  let out : MlpMseParameters i h o →L[ℝ] WithLp 2 (Mat o h × Vec o) :=
    (WithLp.sndL 2 ℝ _ _).comp tail
  let w2 : MlpMseParameters i h o →L[ℝ] Mat o h := (WithLp.fstL 2 ℝ _ _).comp out
  let b2 : MlpMseParameters i h o →L[ℝ] Vec o := (WithLp.sndL 2 ℝ _ _).comp out
  let z := z1 p.fst p.snd.fst x
  let a := a1 p.fst p.snd.fst x
  let dy := (matApplyLin a).comp w2 +
    (matCLM (toMatrix p.snd.snd.fst)).comp
      ((reluDerivCLM z).comp ((matApplyLin x).comp w1 + b1)) + b2
  exact ((2 / (o : ℝ)) • innerSL ℝ (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x - t)).comp dy

/-- Coordinate expansion of the selected joint linearization. -/
theorem selectedLinearization_apply (x : Vec i) (t : Vec o)
    (p d : MlpMseParameters i h o) :
    mlpMseSelectedLinearization x t p d =
      inner ℝ (mseGrad (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x) t)
        ((matApplyLin (a1 p.fst p.snd.fst x)) d.snd.snd.fst +
          (matCLM (toMatrix p.snd.snd.fst))
            ((reluDerivCLM (z1 p.fst p.snd.fst x))
              ((matApplyLin x) d.fst + d.snd.fst)) + d.snd.snd.snd) := by
  simp [mlpMseSelectedLinearization, mseGrad, ContinuousLinearMap.comp_apply,
    WithLp.fstL, WithLp.sndL, inner_smul_left, inner_add_right, inner_sub_left]

/-- The adjoint of the selected chain linearization gives the four parameter blocks. -/
theorem mlpMseSelectedLinearization_adjoint_one (x : Vec i) (t : Vec o)
    (p : MlpMseParameters i h o) :
    (mlpMseSelectedLinearization x t p).adjoint 1 =
      mlpMseBackprop p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x t := by
  apply ext_inner_left ℝ
  intro d
  rw [ContinuousLinearMap.adjoint_inner_right]
  simp only [Real.inner_apply, mul_one]
  rw [selectedLinearization_apply, real_inner_comm]
  rw [inner_add_left, inner_add_left, inner_matApply_eq,
    ← ContinuousLinearMap.adjoint_inner_right]
  rw [← ContinuousLinearMap.adjoint_inner_right, reluDerivCLM_adjoint_apply]
  rw [inner_add_left, inner_matApply_eq]
  simp only [mlpMseBackprop, packMlpMseParameters, WithLp.prod_inner_apply,
    add_assoc]
  dsimp only [WithLp.fst, WithLp.snd]
  ring

/-- A single tensor slot stores an isometric encoding of the joint parameter space. -/
abbrev MlpMseTapeContext (i h o : Nat) : List Shape :=
  [.dim (Module.finrank ℝ (MlpMseParameters i h o)) .scalar]

/-- A canonical coordinate isometry, used only to connect Hilbert parameters to tape vectors. -/
def mlpMseTapeEncoding : MlpMseParameters i h o ≃ₗᵢ[ℝ] CtxVec (MlpMseTapeContext i h o) := by
  have hsize : ctxSize (MlpMseTapeContext i h o) =
      Module.finrank ℝ (MlpMseParameters i h o) := by
    simp [ctxSize, Shape.size]
  exact ((stdOrthonormalBasis ℝ (MlpMseParameters i h o)).reindex
    (finCongr hsize.symm)).repr

/-- A fused MLP/MSE node, with the selected chain JVP and its independently defined adjoint VJP. -/
def mlpMseTapeNode
    (e : MlpMseParameters i h o ≃ₗᵢ[ℝ] CtxVec (MlpMseTapeContext i h o))
    (x : Vec i) (t : Vec o) :
    Node (MlpMseTapeContext i h o) .scalar :=
  let A := fun c ↦ vecScalarCLM.comp
    ((mlpMseSelectedLinearization x t (e.symm c)).comp
      e.symm.toContinuousLinearEquiv.toContinuousLinearMap)
  Node.ofFn
    (fun c ↦ vecScalarCLM (mlpMseObjective x t (e.symm c)))
    (fun c dc ↦ A c dc)
    (fun c δ ↦ (A c).adjoint δ)
    (fun c dc δ ↦ (ContinuousLinearMap.adjoint_inner_right (A := A c) (x := dc) (y := δ)).symm)

/-- Execute the fused node using TorchLean's graph evaluator. -/
def mlpMseTape
    (e : MlpMseParameters i h o ≃ₗᵢ[ℝ] CtxVec (MlpMseTapeContext i h o))
    (x : Vec i) (t : Vec o) : Graph (MlpMseTapeContext i h o) [.scalar] :=
  .snoc .nil (mlpMseTapeNode e x t)

/-- The tape's scalar output is the original joint objective. -/
theorem evalVec_mlpMseTape
    (e : MlpMseParameters i h o ≃ₗᵢ[ℝ] CtxVec (MlpMseTapeContext i h o))
    (x : Vec i) (t : Vec o) (p : MlpMseParameters i h o) :
    (unsnocCtx ((mlpMseTape e x t).evalVec (e p))).2 =
      vecScalarCLM (mlpMseObjective x t p) := by
  simp [mlpMseTape, Graph.evalVec, mlpMseTapeNode, unsnocCtx_snocCtx]

/-- Reverse evaluation agrees with the four closed-form blocks at every parameter point. -/
theorem backpropVec_mlpMseTape
    (e : MlpMseParameters i h o ≃ₗᵢ[ℝ] CtxVec (MlpMseTapeContext i h o))
    (x : Vec i) (t : Vec o) (p : MlpMseParameters i h o) :
    e.symm ((mlpMseTape e x t).backpropVec (e p)
      (snocCtx (Γ := MlpMseTapeContext i h o) (τ := .scalar) 0 (vecScalarCLM 1))) =
      mlpMseBackprop p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x t := by
  have hscalar : vecScalarCLM.adjoint (vecScalarCLM 1) = (1 : ℝ) := by
    apply ext_inner_left ℝ
    intro r
    rw [ContinuousLinearMap.adjoint_inner_right]
    change inner ℝ (vecOfFun (fun _ : Fin (Shape.size .scalar) ↦ r))
      (vecOfFun (fun _ : Fin (Shape.size .scalar) ↦ (1 : ℝ))) = inner ℝ r 1
    simp only [inner_scalarVec_left, vecOfFun_ofLp, Real.inner_apply]
  simp only [mlpMseTape, Graph.backpropVec, Graph.evalVec, castCtxVec_rfl,
    unsnocCtx_snocCtx, zero_add, mlpMseTapeNode, Node.vjpVec_ofFn]
  rw [ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.adjoint_comp]
  simp only [ContinuousLinearMap.comp_apply, hscalar]
  rw [mlpMseSelectedLinearization_adjoint_one]
  have he : e.symm.toContinuousLinearEquiv.toContinuousLinearMap.adjoint =
      e.toContinuousLinearEquiv.toContinuousLinearMap := by
    exact e.symm.adjoint_eq_symm
  rw [he]
  simp only [ContinuousLinearEquiv.coe_coe, LinearIsometryEquiv.coe_toContinuousLinearEquiv,
    LinearIsometryEquiv.symm_apply_apply]

/-- The actual fused tape output has the generalized-gradient postcondition. -/
theorem backpropVec_mlpMseTape_mem_clarkeGradient
    (e : MlpMseParameters i h o ≃ₗᵢ[ℝ] CtxVec (MlpMseTapeContext i h o))
    (x : Vec i) (t : Vec o) (p : MlpMseParameters i h o) :
    e.symm ((mlpMseTape e x t).backpropVec (e p)
      (snocCtx (Γ := MlpMseTapeContext i h o) (τ := .scalar) 0 (vecScalarCLM 1))) ∈
      clarkeGradient (mlpMseObjective x t) p := by
  rw [backpropVec_mlpMseTape]
  exact mlpMseBackprop_mem_clarkeGradient p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x t

end

end Proofs.Autograd
