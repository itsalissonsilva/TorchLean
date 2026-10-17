/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Alisson M. Silva
-/

module

public import NN.Proofs.Autograd.FDeriv.MlpMseNonsmooth
public import NN.Proofs.Autograd.FDeriv.DeepRelu

/-!
# Shared-parameter finite batches

A single negative hidden-bias shift resolves every sample's zero preactivations.
Unlike the single-example construction, activations need not stay constant;
the proof uses convergence and eventual constancy of the selected masks.
The output loss may be MSE or a differentiable scalar function whose gradient
is continuous at each current sample output.
-/

@[expose] public section

namespace Proofs.Autograd.MlpMseBatch

open Spec TorchLean Filter
open scoped Topology Gradient BigOperators

noncomputable section

def step (n : Nat) : ℝ := 1 / ((n : ℝ) + 1)

def negativeOnes (h : Nat) : Vec h := WithLp.toLp 2 fun _ ↦ (-1 : ℝ)

/-- The same bias shift is applied to all examples; weights and output bias stay fixed. -/
def approximant {i h o : Nat} (p : MlpMseParameters i h o) (n : Nat) :
    MlpMseParameters i h o :=
  packMlpMseParameters p.fst (p.snd.fst + step n • negativeOnes h) p.snd.snd.fst p.snd.snd.snd

theorem step_tendsto : Tendsto step atTop (nhds 0) :=
  tendsto_one_div_add_atTop_nhds_zero_nat

private theorem pack_tendsto {i h o : Nat}
    {w : Nat → Mat h i} {b : Nat → Vec h} {v : Nat → Mat o h} {c : Nat → Vec o}
    {W : Mat h i} {B : Vec h} {V : Mat o h} {C : Vec o}
    (hw : Tendsto w atTop (nhds W)) (hb : Tendsto b atTop (nhds B))
    (hv : Tendsto v atTop (nhds V)) (hc : Tendsto c atTop (nhds C)) :
    Tendsto (fun n ↦ packMlpMseParameters (w n) (b n) (v n) (c n)) atTop
      (nhds (packMlpMseParameters W B V C)) := by
  have ho := (WithLp.prodContinuousLinearEquiv 2 ℝ (Mat o h)
    (Vec o)).symm.continuousAt.tendsto.comp (hv.prodMk_nhds hc)
  have hr := (WithLp.prodContinuousLinearEquiv 2 ℝ (Vec h)
    (WithLp 2 (Mat o h × Vec o))).symm.continuousAt.tendsto.comp (hb.prodMk_nhds ho)
  exact (WithLp.prodContinuousLinearEquiv 2 ℝ (Mat h i)
    (WithLp 2 (Vec h × WithLp 2 (Mat o h × Vec o)))).symm.continuousAt.tendsto.comp
      (hw.prodMk_nhds hr)

theorem approximant_tendsto {i h o : Nat} (p : MlpMseParameters i h o) :
    Tendsto (approximant p) atTop (nhds p) := by
  have hb : Tendsto (fun n ↦ p.snd.fst + step n • negativeOnes h) atTop
      (nhds (p.snd.fst + (0 : ℝ) • negativeOnes h)) :=
    tendsto_const_nhds.add (step_tendsto.smul_const (negativeOnes h))
  change Tendsto (fun n ↦ packMlpMseParameters p.fst
    (p.snd.fst + step n • negativeOnes h) p.snd.snd.fst p.snd.snd.snd) atTop
      (nhds (packMlpMseParameters p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd))
  exact pack_tendsto (W := p.fst) (B := p.snd.fst)
    (V := p.snd.snd.fst) (C := p.snd.snd.snd) tendsto_const_nhds
    (by simpa using hb) tendsto_const_nhds tendsto_const_nhds

private theorem preactivation_shift {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (n : Nat) (j : Fin h) :
    z1 (approximant p n).fst (approximant p n).snd.fst x j =
      z1 p.fst p.snd.fst x j - step n := by
  simp [approximant, packMlpMseParameters, z1, affineMat, affine, negativeOnes]
  ring

private theorem eventually_scalar_mask (z : ℝ) :
    ∀ᶠ n in atTop, z - step n ≠ 0 ∧
      Activation.Math.reluDerivSpec (z - step n) = Activation.Math.reluDerivSpec z := by
  by_cases hz : 0 < z
  · filter_upwards [step_tendsto.eventually (gt_mem_nhds hz)] with n hn
    have hp : 0 < z - step n := sub_pos.mpr hn
    exact ⟨ne_of_gt hp, by simp [Activation.Math.reluDerivSpec, hz, hp]⟩
  · exact Filter.Eventually.of_forall fun n ↦ by
      have hs : 0 < step n := by dsimp [step]; positivity
      have hn : z - step n < 0 := sub_neg.mpr (lt_of_le_of_lt (le_of_not_gt hz) hs)
      exact ⟨ne_of_lt hn, by simp [Activation.Math.reluDerivSpec, hz, not_lt_of_ge hn.le]⟩

/-- Each sample eventually has the original selected mask and no hidden kink. -/
theorem eventually_noKinks_mask {i h o : Nat} (p : MlpMseParameters i h o) (x : Vec i) :
    ∀ᶠ n in atTop, (∀ j, z1 (approximant p n).fst (approximant p n).snd.fst x j ≠ 0) ∧
      reluDerivCLM (z1 (approximant p n).fst (approximant p n).snd.fst x) =
        reluDerivCLM (z1 p.fst p.snd.fst x) := by
  have H := (eventually_all.2 fun j : Fin h ↦
    eventually_scalar_mask (z1 p.fst p.snd.fst x j))
  filter_upwards [H] with n hn
  refine ⟨fun j ↦ by simpa only [preactivation_shift] using (hn j).1, ?_⟩
  ext d j
  simp only [reluDerivCLM_apply, preactivation_shift, (hn j).2]

private theorem continuous_outer {m n : Nat} :
    Continuous (fun p : Vec m × Vec n ↦ outer p.1 p.2) := by
  exact (PiLp.continuous_toLp 2 _).comp (continuous_pi fun j ↦
    (PiLp.continuous_toLp 2 _).comp (continuous_pi fun k ↦
      ((PiLp.continuous_apply 2 _ j).comp continuous_fst).mul
        ((PiLp.continuous_apply 2 _ k).comp continuous_snd)))

/-- Although the activations move, each sample's full backward tuple converges. -/
theorem backprop_tendsto {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (t : Vec o) :
    Tendsto (fun n ↦ mlpMseBackprop (approximant p n).fst (approximant p n).snd.fst
      (approximant p n).snd.snd.fst (approximant p n).snd.snd.snd x t) atTop
        (nhds (mlpMseBackprop p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x t)) := by
  let b := fun n ↦ p.snd.fst + step n • negativeOnes h
  let a := fun n ↦ a1 p.fst (b n) x
  let δ := fun n ↦ mseGrad (y p.fst (b n) p.snd.snd.fst p.snd.snd.snd x) t
  let H := fun n ↦ reluDerivCLM (z1 p.fst p.snd.fst x)
    ((matCLM (toMatrix p.snd.snd.fst)).adjoint (δ n))
  have hb : Tendsto b atTop (nhds p.snd.fst) := by
    simpa [b] using (tendsto_const_nhds (x := p.snd.fst)).add
      (step_tendsto.smul_const (negativeOnes h))
  have hz : Tendsto (fun n ↦ z1 p.fst (b n) x) atTop (nhds (z1 p.fst p.snd.fst x)) :=
    tendsto_const_nhds.add hb
  have ha : Tendsto a atTop (nhds (a1 p.fst p.snd.fst x)) :=
    lipschitzWith_reluVec.continuous.continuousAt.tendsto.comp hz
  have hy : Tendsto (fun n ↦ y p.fst (b n) p.snd.snd.fst p.snd.snd.snd x) atTop
      (nhds (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x)) :=
    ((matCLM (toMatrix p.snd.snd.fst)).continuous.continuousAt.tendsto.comp ha).add_const _
  have hδ : Tendsto δ atTop
      (nhds (mseGrad (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x) t)) :=
    (hy.sub_const t).const_smul (2 / (o : ℝ))
  have hH : Tendsto H atTop (nhds (reluDerivCLM (z1 p.fst p.snd.fst x)
      ((matCLM (toMatrix p.snd.snd.fst)).adjoint
        (mseGrad (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x) t)))) :=
    (reluDerivCLM (z1 p.fst p.snd.fst x)).continuous.continuousAt.tendsto.comp
      ((matCLM (toMatrix p.snd.snd.fst)).adjoint.continuous.continuousAt.tendsto.comp hδ)
  have hw := (matApplyLin (m := h) x).adjoint.continuous.continuousAt.tendsto.comp hH
  simp only [Function.comp_def, matApplyLin_adjoint_apply] at hw
  have hv := continuous_outer.continuousAt.tendsto.comp (hδ.prodMk_nhds ha)
  have hp := pack_tendsto hw hH hv hδ
  apply hp.congr'
  filter_upwards [eventually_noKinks_mask p x] with n hn
  change packMlpMseParameters (outer (H n) x) (H n) (outer (δ n) (a n)) (δ n) = _
  simp only [mlpMseBackprop, approximant, packMlpMseParameters, WithLp.toLp_fst,
    WithLp.toLp_snd] at hn ⊢
  rw [hn.2]

private theorem gradient_sum {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [CompleteSpace E] {N : Nat} (f : Fin N → E → ℝ) (p : E)
    (hf : ∀ k, DifferentiableAt ℝ (f k) p) :
    ∇ (fun q ↦ ∑ k, f k q) p = ∑ k, ∇ (f k) p := by
  rw [gradient, fderiv_fun_sum (fun k _ ↦ hf k)]
  simp only [map_sum, gradient]

private theorem gradient_scale {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [CompleteSpace E] (f : E → ℝ) (p : E) (c : ℝ) (hf : DifferentiableAt ℝ f p) :
    ∇ (fun q ↦ c • f q) p = c • ∇ f p := by
  rw [gradient, fderiv_fun_const_smul hf c]
  simp only [map_smul, gradient]

/-- Continuous output gradients give convergence for losses other than MSE. -/
theorem lossBackprop_tendsto {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (ℓ : Vec o → ℝ)
    (hc : ContinuousAt (∇ ℓ) (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x)) :
    Tendsto (fun n ↦ DeepRelu.lossBackprop [h] (approximant p n) x ℓ) atTop
      (nhds (DeepRelu.lossBackprop [h] p x ℓ)) := by
  let b := fun n ↦ p.snd.fst + step n • negativeOnes h
  let a := fun n ↦ a1 p.fst (b n) x
  let δ := fun n ↦ ∇ ℓ (y p.fst (b n) p.snd.snd.fst p.snd.snd.snd x)
  let H := fun n ↦ reluDerivCLM (z1 p.fst p.snd.fst x)
    ((matCLM (toMatrix p.snd.snd.fst)).adjoint (δ n))
  have hb : Tendsto b atTop (nhds p.snd.fst) := by
    simpa [b] using (tendsto_const_nhds (x := p.snd.fst)).add
      (step_tendsto.smul_const (negativeOnes h))
  have hz : Tendsto (fun n ↦ z1 p.fst (b n) x) atTop (nhds (z1 p.fst p.snd.fst x)) :=
    tendsto_const_nhds.add hb
  have ha : Tendsto a atTop (nhds (a1 p.fst p.snd.fst x)) :=
    lipschitzWith_reluVec.continuous.continuousAt.tendsto.comp hz
  have hy : Tendsto (fun n ↦ y p.fst (b n) p.snd.snd.fst p.snd.snd.snd x) atTop
      (nhds (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x)) :=
    ((matCLM (toMatrix p.snd.snd.fst)).continuous.continuousAt.tendsto.comp ha).add_const _
  have hδ : Tendsto δ atTop
      (nhds (∇ ℓ (y p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x))) := hc.tendsto.comp hy
  have hH := (reluDerivCLM (z1 p.fst p.snd.fst x)).continuous.continuousAt.tendsto.comp
    ((matCLM (toMatrix p.snd.snd.fst)).adjoint.continuous.continuousAt.tendsto.comp hδ)
  have hw := (matApplyLin (m := h) x).adjoint.continuous.continuousAt.tendsto.comp hH
  simp only [Function.comp_def, matApplyLin_adjoint_apply] at hw
  have hv := continuous_outer.continuousAt.tendsto.comp (hδ.prodMk_nhds ha)
  have hp := pack_tendsto hw hH hv hδ
  apply hp.congr'
  filter_upwards [eventually_noKinks_mask p x] with n hn
  change packMlpMseParameters (outer (H n) x) (H n) (outer (δ n) (a n)) (δ n) = _
  simp only [DeepRelu.lossBackprop, DeepRelu.backward, DeepRelu.forward, approximant,
    packMlpMseParameters, WithLp.toLp_fst, WithLp.toLp_snd] at hn ⊢
  rw [hn.2]
  rfl

/-- Mean of sample-specific scalar output losses, with the empty mean equal to zero. -/
def lossObjective {i h o N : Nat} (x : Fin N → Vec i) (ℓ : Fin N → Vec o → ℝ)
    (p : MlpMseParameters i h o) : ℝ :=
  (N : ℝ)⁻¹ * ∑ k, ℓ k (DeepRelu.forward [h] p (x k))

/-- Mean reverse value seeded by the sample-specific output gradients. -/
def lossBackprop {i h o N : Nat} (p : MlpMseParameters i h o)
    (x : Fin N → Vec i) (ℓ : Fin N → Vec o → ℝ) : MlpMseParameters i h o :=
  (N : ℝ)⁻¹ • ∑ k, DeepRelu.lossBackprop [h] p (x k) (ℓ k)

/-- The common-tail argument applies to differentiable losses with continuous output gradients. -/
theorem lossBackprop_isLimitingGradient {i h o N : Nat} (p : MlpMseParameters i h o)
    (x : Fin N → Vec i) (ℓ : Fin N → Vec o → ℝ)
    (hdℓ : ∀ k, Differentiable ℝ (ℓ k))
    (hc : ∀ k, ContinuousAt (∇ (ℓ k)) (DeepRelu.forward [h] p (x k))) :
    IsLimitingGradient (lossObjective x ℓ) p (lossBackprop p x ℓ) := by
  have he := eventually_all.2 fun k : Fin N ↦ eventually_noKinks_mask p (x k)
  obtain ⟨K, hK⟩ := eventually_atTop.1 he
  let q := fun n ↦ approximant p (n + K)
  have hk : ∀ n k, DeepRelu.NoKinks [h] (q n) (x k) :=
    fun n k ↦ ⟨(hK (n + K) (Nat.le_add_left K n) k).1, trivial⟩
  have hd : ∀ n k, DifferentiableAt ℝ (fun r ↦ ℓ k (DeepRelu.forward [h] r (x k)))
      (q n) := by
    intro n k
    exact (DeepRelu.hasFDerivAt_loss [h] _ (x k) (ℓ k) (hk n k) (hdℓ k _)).differentiableAt
  have hs : ∀ n, DifferentiableAt ℝ
      (fun r ↦ ∑ k, ℓ k (DeepRelu.forward [h] r (x k))) (q n) :=
    fun n ↦ DifferentiableAt.fun_sum fun k _ ↦ hd n k
  have hg : ∀ n, ∇ (lossObjective x ℓ) (q n) = lossBackprop (q n) x ℓ := by
    intro n
    change ∇ (fun r ↦ (N : ℝ)⁻¹ • ∑ k, ℓ k (DeepRelu.forward [h] r (x k))) (q n) = _
    rw [gradient_scale _ _ _ (hs n), gradient_sum _ _ (hd n)]
    exact congrArg ((N : ℝ)⁻¹ • ·) (Finset.sum_congr rfl fun k _ ↦
      DeepRelu.gradient_loss_eq [h] _ (x k) (ℓ k) (hk n k) (hdℓ k _))
  refine ⟨q, (approximant_tendsto p).comp (tendsto_add_atTop_nat K),
    fun n ↦ (hs n).const_smul ((N : ℝ)⁻¹), ?_⟩
  have hb := (tendsto_finsetSum Finset.univ
    (fun k _ ↦ lossBackprop_tendsto p (x k) (ℓ k) (hc k))).const_smul ((N : ℝ)⁻¹)
  exact (hb.comp (tendsto_add_atTop_nat K)).congr'
    (Filter.Eventually.of_forall fun n ↦ (hg n).symm)

/-- The finite-batch differentiable-loss reverse value satisfies Clarke's inequalities. -/
theorem lossBackprop_mem_clarkeGradient {i h o N : Nat} (p : MlpMseParameters i h o)
    (x : Fin N → Vec i) (ℓ : Fin N → Vec o → ℝ)
    (hdℓ : ∀ k, Differentiable ℝ (ℓ k))
    (hc : ∀ k, ContinuousAt (∇ (ℓ k)) (DeepRelu.forward [h] p (x k))) :
    lossBackprop p x ℓ ∈ clarkeGradient (lossObjective x ℓ) p :=
  isLimitingGradient_mem_clarkeGradient (lossBackprop_isLimitingGradient p x ℓ hdℓ hc)

/-- Mean MSE over a finite batch, sharing the entire parameter tuple. -/
def objective {i h o N : Nat} (x : Fin N → Vec i) (t : Fin N → Vec o)
    (p : MlpMseParameters i h o) : ℝ := (N : ℝ)⁻¹ * ∑ k, mlpMseObjective (x k) (t k) p

/-- Mean of the selected backward tuples for the shared parameter space. -/
def backprop {i h o N : Nat} (p : MlpMseParameters i h o)
    (x : Fin N → Vec i) (t : Fin N → Vec o) : MlpMseParameters i h o :=
  (N : ℝ)⁻¹ • ∑ k, mlpMseBackprop p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd (x k) (t k)

/-- Finite batches admit one smooth joint parameter sequence for all sample gradients. -/
theorem backprop_isLimitingGradient {i h o N : Nat} (p : MlpMseParameters i h o)
    (x : Fin N → Vec i) (t : Fin N → Vec o) :
    IsLimitingGradient (objective x t) p (backprop p x t) := by
  have he := eventually_all.2 fun k : Fin N ↦ eventually_noKinks_mask p (x k)
  obtain ⟨K, hK⟩ := eventually_atTop.1 he
  let q := fun n ↦ approximant p (n + K)
  have hd : ∀ n k, DifferentiableAt ℝ (mlpMseObjective (x k) (t k)) (q n) := by
    intro n k
    exact differentiableAt_mlpMseObjective_of_noKinks _ _ _ _ _ _
      (hK (n + K) (Nat.le_add_left K n) k).1
  have hs : ∀ n, DifferentiableAt ℝ
      (fun r ↦ ∑ k, mlpMseObjective (x k) (t k) r) (q n) :=
    fun n ↦ DifferentiableAt.fun_sum fun k _ ↦ hd n k
  have hg : ∀ n, ∇ (objective x t) (q n) = backprop (q n) x t := by
    intro n
    change ∇ (fun r ↦ (N : ℝ)⁻¹ • ∑ k, mlpMseObjective (x k) (t k) r) (q n) =
      (N : ℝ)⁻¹ • ∑ k, mlpMseBackprop (q n).fst (q n).snd.fst (q n).snd.snd.fst
        (q n).snd.snd.snd (x k) (t k)
    rw [gradient_scale _ _ _ (hs n), gradient_sum _ _ (hd n)]
    congr 1
    apply Finset.sum_congr rfl
    intro k _
    exact gradient_mlpMseObjective_eq_backprop_of_noKinks _ _ _ _ _ _
      (hK (n + K) (Nat.le_add_left K n) k).1
  refine ⟨q, (approximant_tendsto p).comp (tendsto_add_atTop_nat K),
    fun n ↦ (hs n).const_smul ((N : ℝ)⁻¹), ?_⟩
  have hb0 := tendsto_finsetSum Finset.univ (fun k _ ↦ backprop_tendsto p (x k) (t k))
  have hb := hb0.const_smul ((N : ℝ)⁻¹)
  exact (hb.comp (tendsto_add_atTop_nat K)).congr'
    (Filter.Eventually.of_forall fun n ↦ (hg n).symm)

/-- The shared-parameter finite-batch backward value belongs to the joint Clarke set. -/
theorem backprop_mem_clarkeGradient {i h o N : Nat} (p : MlpMseParameters i h o)
    (x : Fin N → Vec i) (t : Fin N → Vec o) :
    backprop p x t ∈ clarkeGradient (objective x t) p :=
  isLimitingGradient_mem_clarkeGradient (backprop_isLimitingGradient p x t)

end

end Proofs.Autograd.MlpMseBatch
