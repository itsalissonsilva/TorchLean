/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Alisson M. Silva
-/

module

public import NN.Proofs.Autograd.FDeriv.MlpMseNonsmooth

/-!
# Finite-depth ReLU networks with independent biases

The list of hidden widths determines a recursively bundled Hilbert parameter
space. Every hidden layer uses TorchLean's selected ReLU; the final layer is affine.
-/

@[expose] public section

namespace Proofs.Autograd.DeepRelu

open Spec TorchLean Filter
open scoped Topology Gradient

noncomputable section

/-- One affine output layer, preceded by the specified list of hidden layers. -/
@[reducible] def Parameters (i o : Nat) : List Nat → Type
  | [] => WithLp 2 (Mat o i × Vec o)
  | h :: hs => WithLp 2 (Mat h i × WithLp 2 (Vec h × Parameters h o hs))

instance parametersNormed (i o : Nat) : (hs : List Nat) → NormedAddCommGroup (Parameters i o hs)
  | [] => inferInstanceAs (NormedAddCommGroup (WithLp 2 (Mat o i × Vec o)))
  | h :: hs =>
      letI := parametersNormed h o hs
      inferInstanceAs (NormedAddCommGroup
        (WithLp 2 (Mat h i × WithLp 2 (Vec h × Parameters h o hs))))

instance parametersInner (i o : Nat) : (hs : List Nat) → InnerProductSpace ℝ (Parameters i o hs)
  | [] => inferInstanceAs (InnerProductSpace ℝ (WithLp 2 (Mat o i × Vec o)))
  | h :: hs =>
      letI := parametersInner h o hs
      inferInstanceAs (InnerProductSpace ℝ
        (WithLp 2 (Mat h i × WithLp 2 (Vec h × Parameters h o hs))))

instance parametersFinite (i o : Nat) :
    (hs : List Nat) → FiniteDimensional ℝ (Parameters i o hs)
  | [] => inferInstanceAs (FiniteDimensional ℝ (WithLp 2 (Mat o i × Vec o)))
  | h :: hs =>
      letI := parametersFinite h o hs
      inferInstanceAs (FiniteDimensional ℝ
        (WithLp 2 (Mat h i × WithLp 2 (Vec h × Parameters h o hs))))

abbrev Joint (i o : Nat) (hs : List Nat) := WithLp 2 (Parameters i o hs × Vec i)

/-- Bundle two continuous linear maps using the Hilbert product. -/
def pairL {E F G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] [NormedAddCommGroup G] [NormedSpace ℝ G]
    (f : E →L[ℝ] F) (g : E →L[ℝ] G) : E →L[ℝ] WithLp 2 (F × G) :=
  (WithLp.prodContinuousLinearEquiv 2 ℝ F G).symm.toContinuousLinearMap.comp (f.prod g)

@[simp] theorem pairL_apply {E F G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] [NormedAddCommGroup G] [NormedSpace ℝ G]
    (f : E →L[ℝ] F) (g : E →L[ℝ] G) (x : E) :
    pairL f g x = WithLp.toLp 2 (f x, g x) := rfl

/-- Forward evaluation, using the same affine and ReLU functions as the two-layer model. -/
def forward {i o : Nat} : (hs : List Nat) → Parameters i o hs → Vec i → Vec o
  | [], p, x => affineMat p.fst p.snd x
  | _ :: hs, p, x => forward hs p.snd.snd (reluVec (z1 p.fst p.snd.fst x))

/-- Selected chain map, differentiating jointly in every parameter and in the input. -/
def selected {i o : Nat} : (hs : List Nat) → Parameters i o hs → Vec i →
    Joint i o hs →L[ℝ] Vec o
  | [], p, x =>
      let pL : Joint i o [] →L[ℝ] Parameters i o [] := WithLp.fstL 2 ℝ _ _
      let xL : Joint i o [] →L[ℝ] Vec i := WithLp.sndL 2 ℝ _ _
      (matApplyLin x).comp ((WithLp.fstL 2 ℝ _ _).comp pL) +
        (matCLM (toMatrix p.fst)).comp xL + (WithLp.sndL 2 ℝ _ _).comp pL
  | h :: hs, p, x =>
      let pL : Joint i o (h :: hs) →L[ℝ] Parameters i o (h :: hs) := WithLp.fstL 2 ℝ _ _
      let rL := (WithLp.sndL 2 ℝ _ _).comp pL
      let xL : Joint i o (h :: hs) →L[ℝ] Vec i := WithLp.sndL 2 ℝ _ _
      let dz := (matApplyLin x).comp ((WithLp.fstL 2 ℝ _ _).comp pL) +
        (matCLM (toMatrix p.fst)).comp xL + (WithLp.fstL 2 ℝ _ _).comp rL
      (selected hs p.snd.snd (reluVec (z1 p.fst p.snd.fst x))).comp
        (pairL ((WithLp.sndL 2 ℝ _ _).comp rL)
          ((reluDerivCLM (z1 p.fst p.snd.fst x)).comp dz))

/-- Every hidden preactivation is nonzero at the current parameters and input. -/
def NoKinks {i o : Nat} : (hs : List Nat) → Parameters i o hs → Vec i → Prop
  | [], _, _ => True
  | _ :: hs, p, x => (∀ j, z1 p.fst p.snd.fst x j ≠ 0) ∧
      NoKinks hs p.snd.snd (reluVec (z1 p.fst p.snd.fst x))

private theorem hasFDerivAt_affine {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {i o : Nat} {W : E → Mat o i} {b : E → Vec o} {x : E → Vec i}
    {dW : E →L[ℝ] Mat o i} {db : E →L[ℝ] Vec o} {dx : E →L[ℝ] Vec i} {q : E}
    (hW : HasFDerivAt W dW q) (hb : HasFDerivAt b db q) (hx : HasFDerivAt x dx q) :
    HasFDerivAt (fun r ↦ affineMat (W r) (b r) (x r))
      ((matApplyLin (x q)).comp dW + (matCLM (toMatrix (W q))).comp dx + db) q := by
  have H := ((matVecBilinear (m := o) (n := i)).hasFDerivAt_of_bilinear hW hx).add hb
  convert H using 1
  · rfl
  · ext d : 1
    change matVecBilinear (dW d) (x q) + matVecBilinear (W q) (dx d) + db d =
      matVecBilinear (W q) (dx d) + matVecBilinear (dW d) (x q) + db d
    abel

private theorem hasFDerivAt_pairL {E F G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] [NormedAddCommGroup G] [NormedSpace ℝ G]
    {f : E → F} {g : E → G} {df : E →L[ℝ] F} {dg : E →L[ℝ] G} {q : E}
    (hf : HasFDerivAt f df q) (hg : HasFDerivAt g dg q) :
    HasFDerivAt (fun r ↦ WithLp.toLp 2 (f r, g r)) (pairL df dg) q :=
  (WithLp.prodContinuousLinearEquiv 2 ℝ F G).symm.toContinuousLinearMap.hasFDerivAt.comp q
    (hf.prodMk hg)

/-- The selected joint chain map is the Fréchet derivative away from every hidden kink. -/
theorem hasFDerivAt_forward {i o : Nat} (hs : List Nat) (p : Parameters i o hs) (x : Vec i)
    (hk : NoKinks hs p x) :
    HasFDerivAt (fun q : Joint i o hs ↦ forward hs q.fst q.snd) (selected hs p x)
      (WithLp.toLp 2 (p, x)) := by
  induction hs generalizing i with
  | nil =>
      let pL : Joint i o [] →L[ℝ] Parameters i o [] := WithLp.fstL 2 ℝ _ _
      let wL := (WithLp.fstL 2 ℝ _ _).comp pL
      let bL := (WithLp.sndL 2 ℝ _ _).comp pL
      let xL : Joint i o [] →L[ℝ] Vec i := WithLp.sndL 2 ℝ _ _
      exact hasFDerivAt_affine (E := Joint i o []) (q := WithLp.toLp 2 (p, x))
        (W := wL) (b := bL) (x := xL) wL.hasFDerivAt bL.hasFDerivAt xL.hasFDerivAt
  | cons h hs ih =>
      let q := WithLp.toLp 2 (p, x)
      let pL : Joint i o (h :: hs) →L[ℝ] Parameters i o (h :: hs) := WithLp.fstL 2 ℝ _ _
      let rL := (WithLp.sndL 2 ℝ _ _).comp pL
      let wL := (WithLp.fstL 2 ℝ _ _).comp pL
      let bL := (WithLp.fstL 2 ℝ _ _).comp rL
      let tL := (WithLp.sndL 2 ℝ _ _).comp rL
      let xL : Joint i o (h :: hs) →L[ℝ] Vec i := WithLp.sndL 2 ℝ _ _
      have hz := hasFDerivAt_affine (E := Joint i o (h :: hs)) (q := q)
        (W := wL) (b := bL) (x := xL) wL.hasFDerivAt bL.hasFDerivAt xL.hasFDerivAt
      have ha := (hasFDerivAt_reluVec (x := z1 p.fst p.snd.fst x) hk.1).comp q hz
      have ht := hasFDerivAt_pairL (q := q) tL.hasFDerivAt ha
      exact (ih p.snd.snd (reluVec (z1 p.fst p.snd.fst x)) hk.2).comp q ht

/-- Shift only the zero preactivations at each hidden layer to their inactive side. -/
def perturb {i o : Nat} : (hs : List Nat) → Parameters i o hs → Vec i → ℝ → Parameters i o hs
  | [], p, _, _ => p
  | _ :: hs, p, x, ε => WithLp.toLp 2 (p.fst, WithLp.toLp 2
      (perturbHiddenBias p.fst x p.snd.fst ε,
        perturb hs p.snd.snd (reluVec (z1 p.fst p.snd.fst x)) ε))

/-- The perturbation preserves every hidden activation and the final output. -/
theorem forward_perturb {i o : Nat} (hs : List Nat) (p : Parameters i o hs) (x : Vec i)
    {ε : ℝ} (hε : 0 < ε) : forward hs (perturb hs p x ε) x = forward hs p x := by
  induction hs generalizing i with
  | nil => rfl
  | cons h hs ih =>
      simp only [forward, perturb, WithLp.toLp_fst, WithLp.toLp_snd]
      rw [reluVec_z1_perturbHiddenBias _ _ _ hε]
      exact ih _ _

/-- The perturbed network has no hidden zero preactivation. -/
theorem noKinks_perturb {i o : Nat} (hs : List Nat) (p : Parameters i o hs) (x : Vec i)
    {ε : ℝ} (hε : 0 < ε) : NoKinks hs (perturb hs p x ε) x := by
  induction hs generalizing i with
  | nil => trivial
  | cons h hs ih =>
      simp only [NoKinks, perturb, WithLp.toLp_fst, WithLp.toLp_snd]
      refine ⟨z1_perturbHiddenBias_ne_zero _ _ _ hε, ?_⟩
      rw [reluVec_z1_perturbHiddenBias _ _ _ hε]
      exact ih _ _

/-- All selected joint chain maps are preserved by the layerwise perturbation. -/
theorem selected_perturb {i o : Nat} (hs : List Nat) (p : Parameters i o hs) (x : Vec i)
    {ε : ℝ} (hε : 0 < ε) : selected hs (perturb hs p x ε) x = selected hs p x := by
  induction hs generalizing i with
  | nil => rfl
  | cons h hs ih =>
      simp only [selected, perturb, WithLp.toLp_fst, WithLp.toLp_snd]
      rw [reluVec_z1_perturbHiddenBias _ _ _ hε,
        reluDerivCLM_z1_perturbHiddenBias _ _ _ hε, ih]

/-- The whole parameter tuple converges, at arbitrary finite depth. -/
theorem perturb_tendsto {i o : Nat} (hs : List Nat) (p : Parameters i o hs) (x : Vec i) :
    Tendsto (fun n : Nat ↦ perturb hs p x ((1 : ℝ) / ((n : ℝ) + 1))) atTop (nhds p) := by
  induction hs generalizing i with
  | nil => exact tendsto_const_nhds
  | cons h hs ih =>
      have hb := hiddenBiasApproximants_tendsto p.fst x p.snd.fst
      have ht := ih p.snd.snd (reluVec (z1 p.fst p.snd.fst x))
      have hr := (WithLp.prodContinuousLinearEquiv 2 ℝ (Vec h)
        (Parameters h o hs)).symm.continuousAt.tendsto.comp (hb.prodMk_nhds ht)
      have hw : Tendsto (fun _ : Nat ↦ p.fst) atTop (nhds p.fst) := tendsto_const_nhds
      exact (WithLp.prodContinuousLinearEquiv 2 ℝ (Mat h i)
        (WithLp 2 (Vec h × Parameters h o hs))).symm.continuousAt.tendsto.comp
          (hw.prodMk_nhds hr)

/-- Recursive reverse rules return every parameter block and the input cotangent. -/
def backward {i o : Nat} : (hs : List Nat) → Parameters i o hs → Vec i → Vec o → Joint i o hs
  | [], p, x, δ => WithLp.toLp 2 (WithLp.toLp 2 (outer δ x, δ),
      (matCLM (toMatrix p.fst)).adjoint δ)
  | _ :: hs, p, x, δ =>
      let tail := backward hs p.snd.snd (reluVec (z1 p.fst p.snd.fst x)) δ
      let dz := reluDerivCLM (z1 p.fst p.snd.fst x) tail.snd
      WithLp.toLp 2 (WithLp.toLp 2 (outer dz x, WithLp.toLp 2 (dz, tail.fst)),
        (matCLM (toMatrix p.fst)).adjoint dz)

/-- Recursive backpropagation is adjoint to the selected joint chain map. -/
theorem backward_inner {i o : Nat} (hs : List Nat) (p : Parameters i o hs)
    (x : Vec i) (d : Joint i o hs) (δ : Vec o) :
    inner ℝ (selected hs p x d) δ = inner ℝ d (backward hs p x δ) := by
  induction hs generalizing i with
  | nil =>
      change inner ℝ ((matApplyLin x) d.fst.fst +
        (matCLM (toMatrix p.fst)) d.snd + d.fst.snd) δ =
          inner ℝ d.fst.fst (outer δ x) + inner ℝ d.fst.snd δ +
            inner ℝ d.snd ((matCLM (toMatrix p.fst)).adjoint δ)
      rw [inner_add_left, inner_add_left, inner_matApply_eq,
        ContinuousLinearMap.adjoint_inner_right]
      abel
  | cons h hs ih =>
      let z := z1 p.fst p.snd.fst x
      let a := reluVec z
      let r := backward hs p.snd.snd a δ
      change inner ℝ (selected hs p.snd.snd a
        (WithLp.toLp 2 (d.fst.snd.snd, reluDerivCLM z
          ((matApplyLin x) d.fst.fst + (matCLM (toMatrix p.fst)) d.snd + d.fst.snd.fst)))) δ = _
      rw [ih]
      change inner ℝ d.fst.snd.snd r.fst + inner ℝ
        (reluDerivCLM z ((matApplyLin x) d.fst.fst +
          (matCLM (toMatrix p.fst)) d.snd + d.fst.snd.fst)) r.snd =
        inner ℝ d.fst.fst (outer (reluDerivCLM z r.snd) x) +
          (inner ℝ d.fst.snd.fst (reluDerivCLM z r.snd) + inner ℝ d.fst.snd.snd r.fst) +
          inner ℝ d.snd ((matCLM (toMatrix p.fst)).adjoint (reluDerivCLM z r.snd))
      rw [reluDerivCLM_inner, inner_add_left, inner_add_left, inner_matApply_eq,
        ContinuousLinearMap.adjoint_inner_right]
      abel

/-- Scalar MSE loss, jointly in all layer parameters. -/
def objective {i o : Nat} (hs : List Nat) (x : Vec i) (t : Vec o)
    (p : Parameters i o hs) : ℝ := mse t (forward hs p x)

/-- Selected loss derivative with the training example held fixed. -/
def lossLinearization {i o : Nat} (hs : List Nat) (p : Parameters i o hs)
    (x : Vec i) (t : Vec o) : Parameters i o hs →L[ℝ] ℝ :=
  (((2 / (o : ℝ)) • innerSL ℝ (forward hs p x - t)).comp (selected hs p x)).comp
    (pairL (ContinuousLinearMap.id ℝ _) 0)

/-- The explicit recursive backward tuple for scalar MSE. -/
def backprop {i o : Nat} (hs : List Nat) (p : Parameters i o hs) (x : Vec i) (t : Vec o) :
    Parameters i o hs := (backward hs p x (mseGrad (forward hs p x) t)).fst

/-- The recursive MSE backward tuple is the adjoint selected loss derivative at seed one. -/
theorem lossLinearization_adjoint_one {i o : Nat} (hs : List Nat)
    (p : Parameters i o hs) (x : Vec i) (t : Vec o) :
    (lossLinearization hs p x t).adjoint 1 = backprop hs p x t := by
  apply ext_inner_left ℝ
  intro d
  rw [ContinuousLinearMap.adjoint_inner_right]
  have H := backward_inner hs p x (WithLp.toLp 2 (d, 0)) (mseGrad (forward hs p x) t)
  simpa [lossLinearization, backprop, ContinuousLinearMap.comp_apply, mseGrad,
    inner_smul_right, inner_sub_left, real_inner_comm, WithLp.prod_inner_apply,
    Real.inner_apply] using H

/-- The selected MSE map is the joint parameter derivative at smooth parameter tuples. -/
theorem hasFDerivAt_objective {i o : Nat} (hs : List Nat) (p : Parameters i o hs)
    (x : Vec i) (t : Vec o) (hk : NoKinks hs p x) :
    HasFDerivAt (objective hs x t) (lossLinearization hs p x t) p := by
  have hi := hasFDerivAt_pairL (hasFDerivAt_id p) (hasFDerivAt_const x p)
  have hf := (hasFDerivAt_forward hs p x hk).comp p hi
  exact (hasFDerivAt_mse t (forward hs p x)).comp p hf

/-- Joint smooth-gradient correctness for arbitrary hidden widths and depth. -/
theorem gradient_eq_backprop {i o : Nat} (hs : List Nat) (p : Parameters i o hs)
    (x : Vec i) (t : Vec o) (hk : NoKinks hs p x) :
    ∇ (objective hs x t) p = backprop hs p x t := by
  rw [gradient_eq_adjoint_fderiv_one, (hasFDerivAt_objective hs p x t hk).fderiv]
  exact lossLinearization_adjoint_one hs p x t

/-- The scalar selected chain map is unchanged by the perturbation. -/
theorem lossLinearization_perturb {i o : Nat} (hs : List Nat)
    (p : Parameters i o hs) (x : Vec i) (t : Vec o) {ε : ℝ} (hε : 0 < ε) :
    lossLinearization hs (perturb hs p x ε) x t = lossLinearization hs p x t := by
  simp only [lossLinearization, forward_perturb hs p x hε, selected_perturb hs p x hε]

/-- Every recursive backward block stays exactly constant along the smooth approximants. -/
theorem backprop_perturb {i o : Nat} (hs : List Nat)
    (p : Parameters i o hs) (x : Vec i) (t : Vec o) {ε : ℝ} (hε : 0 < ε) :
    backprop hs (perturb hs p x ε) x t = backprop hs p x t := by
  rw [← lossLinearization_adjoint_one, ← lossLinearization_adjoint_one,
    lossLinearization_perturb hs p x t hε]

/-- Generalized backpropagation at every parameter for any finite list of hidden widths. -/
theorem backprop_isLimitingGradient {i o : Nat} (hs : List Nat)
    (p : Parameters i o hs) (x : Vec i) (t : Vec o) :
    IsLimitingGradient (objective hs x t) p (backprop hs p x t) := by
  refine ⟨fun n ↦ perturb hs p x ((1 : ℝ) / ((n : ℝ) + 1)),
    perturb_tendsto hs p x, ?_, ?_⟩
  · intro n
    exact (hasFDerivAt_objective hs _ x t (noKinks_perturb hs p x
      (by positivity))).differentiableAt
  · have H : ∀ n : Nat, ∇ (objective hs x t)
        (perturb hs p x ((1 : ℝ) / ((n : ℝ) + 1))) = backprop hs p x t := by
      intro n
      have hε : (0 : ℝ) < 1 / ((n : ℝ) + 1) := by positivity
      rw [gradient_eq_backprop hs _ x t (noKinks_perturb hs p x hε),
        backprop_perturb hs p x t hε]
    exact tendsto_const_nhds.congr' (Filter.Eventually.of_forall fun n ↦ (H n).symm)

/-- Recursive backpropagation satisfies Clarke's support inequalities at every parameter tuple. -/
theorem backprop_mem_clarkeGradient {i o : Nat} (hs : List Nat)
    (p : Parameters i o hs) (x : Vec i) (t : Vec o) :
    backprop hs p x t ∈ clarkeGradient (objective hs x t) p :=
  isLimitingGradient_mem_clarkeGradient (backprop_isLimitingGradient hs p x t)

/-- One hidden layer recovers TorchLean's original objective definition. -/
theorem objective_singleton {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (t : Vec o) : objective [h] x t p = mlpMseObjective x t p := rfl

/-- One hidden layer recovers the four original closed-form backward blocks. -/
theorem backprop_singleton {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (t : Vec o) :
    backprop [h] p x t = mlpMseBackprop p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x t := rfl

end

end Proofs.Autograd.DeepRelu
