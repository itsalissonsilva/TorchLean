/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Alisson M. Silva
-/

module

public import NN.Proofs.Autograd.Tape.Ops.MlpMseNonsmooth
public import NN.Proofs.Autograd.Tape.Nodes.Losses.MSE
public import NN.Proofs.Autograd.Tape.Nodes.Matrix
public import NN.Proofs.Autograd.Tape.Nodes.Arithmetic
public import NN.Proofs.Autograd.Tape.Nodes.Elementwise
import NN.Proofs.Autograd.Tape.Ops.Norm.CtxVecEval

/-!
# Primitive MLP/MSE tape

The graph uses the existing matrix multiplication, addition, ReLU, and MSE nodes.
All four parameter blocks are inputs, alongside the fixed example and target.
-/

@[expose] public section

namespace Proofs.Autograd

open Spec TorchLean TapeNodes
open scoped BigOperators

noncomputable section

namespace PrimitiveMlp

abbrev M (m n : Nat) : Shape := .dim m (.dim n .scalar)
abbrev C (n : Nat) : Shape := M n 1
abbrev Inputs (i h o : Nat) : List Shape := [M h i, C h, M o h, C o, C i, C o]
abbrev Saved (h o : Nat) : List Shape := [C h, C h, C h, C o, C o, .scalar]

/-- Row-major coordinates of a Hilbert-space weight matrix. -/
def matrixCoordinates {m n : Nat} : Mat m n →L[ℝ] Vec (M m n).size := by
  let f : Mat m n →ₗ[ℝ] Vec (M m n).size :=
    { toFun := fun W ↦ WithLp.toLp 2 fun k ↦ W (k.divNat (n := Matmul.vecSize n))
        (Fin.cast (Matmul.vecSize_eq n) (k.modNat (n := Matmul.vecSize n)))
      map_add' := by intros; ext; simp
      map_smul' := by intros; ext; simp }
  exact ⟨f, f.continuous_of_finiteDimensional⟩

/-- A vector represented as a one-column matrix. -/
def column {n : Nat} : Vec n →L[ℝ] Vec (C n).size := by
  let f : Vec n →ₗ[ℝ] Vec (C n).size :=
    { toFun := fun v ↦ WithLp.toLp 2 fun k ↦ v (Fin.cast (by simp [Shape.size]) k)
      map_add' := by intros; rfl
      map_smul' := by intros; rfl }
  exact ⟨f, f.continuous_of_finiteDimensional⟩

private theorem matrixCoordinates_entry {m n : Nat} (W : Mat m n)
    (a : Fin m) (b : Fin n) :
    matrixCoordinates W (Matmul.idxMN a b) = W a b := by
  change W (Matmul.idxMN a b).divNat
    (Fin.cast (Matmul.vecSize_eq n) (Matmul.idxMN a b).modNat) = W a b
  have he := (finProdFinEquiv (m := m) (n := Matmul.vecSize n)).symm_apply_apply
    (a, Fin.cast (Matmul.vecSize_eq n).symm b)
  have ha := congrArg Prod.fst he
  have hb := congrArg Prod.snd he
  change (Matmul.idxMN a b).divNat = a at ha
  change (Matmul.idxMN a b).modNat = Fin.cast (Matmul.vecSize_eq n).symm b at hb
  rw [ha, hb]
  simp

private theorem column_apply {n : Nat} (v : Vec n) (k : Fin (C n).size) :
    column v k = v (Fin.cast (by simp [Shape.size]) k) := rfl

private theorem column_entry {n : Nat} (v : Vec n) (j : Fin n) (k : Fin 1) :
    column v (Matmul.idxMN j k) = v j := by
  rw [column_apply]
  apply congrArg v
  apply Fin.ext
  have hk : k = 0 := Subsingleton.elim _ _
  change k.val + 1 * j.val = j.val
  simp [hk]

private theorem matmul_column {m n : Nat} (W : Mat m n) (v : Vec n) :
    Matmul.matmulVec (matrixCoordinates W) (column v) = column (matApplyLin v W) := by
  ext k
  change (∑ j : Fin n, matrixCoordinates W (Matmul.idxMN k.divNat j) *
    column v (Matmul.idxMN j (Fin.cast (Matmul.vecSize_eq 1) k.modNat))) =
      column (matApplyLin v W) k
  simp only [matrixCoordinates_entry, column_entry]
  rw [column_apply]
  have hk : k.divNat = Fin.cast (by simp [Shape.size] : (C m).size = m) k := by
    apply Fin.ext
    change k.val / 1 = k.val
    exact Nat.div_one _
  rw [hk]
  rfl

/-- Encode the six graph inputs; the last two slots are not trainable parameters. -/
def inputVector {i h o : Nat} (p : MlpMseParameters i h o) (x : Vec i) (t : Vec o) :
    CtxVec (Inputs i h o) :=
  flattenCtx (.cons (vecToTensor (matrixCoordinates p.fst))
    (.cons (vecToTensor (column p.snd.fst))
      (.cons (vecToTensor (matrixCoordinates p.snd.snd.fst))
        (.cons (vecToTensor (column p.snd.snd.snd))
          (.cons (vecToTensor (column x)) (.cons (vecToTensor (column t)) .nil))))))

/-- Linear insertion of parameter tangents; example and target tangents are zero. -/
def parameterMap {i h o : Nat} : MlpMseParameters i h o →L[ℝ] CtxVec (Inputs i h o) := by
  let P := MlpMseParameters i h o
  let tail : P →L[ℝ] WithLp 2 (Vec h × WithLp 2 (Mat o h × Vec o)) := WithLp.sndL 2 ℝ _ _
  let out : P →L[ℝ] WithLp 2 (Mat o h × Vec o) := (WithLp.sndL 2 ℝ _ _).comp tail
  let w1 : P →L[ℝ] Vec (M h i).size := matrixCoordinates.comp
    (WithLp.fstL 2 ℝ (Mat h i) (WithLp 2 (Vec h × WithLp 2 (Mat o h × Vec o))))
  let b1 := column.comp ((WithLp.fstL 2 ℝ (Vec h) _).comp tail)
  let w2 := matrixCoordinates.comp ((WithLp.fstL 2 ℝ (Mat o h) _).comp out)
  let b2 := column.comp ((WithLp.sndL 2 ℝ (Mat o h) _).comp out)
  let join {a b : Nat} (f : P →L[ℝ] Vec a) (g : P →L[ℝ] Vec b) :=
    (Graph.appendCLM a b).comp (f.prod g)
  exact join w1 (join b1 (join w2 (join b2
    (join (0 : P →L[ℝ] Vec (C i).size)
      (join (0 : P →L[ℝ] Vec (C o).size) (0 : P →L[ℝ] Vec 0))))))

theorem parameterMap_apply {i h o : Nat} (p : MlpMseParameters i h o) :
    parameterMap p = inputVector p 0 0 := by
  simp only [inputVector, flattenCtx_cons, tensorToVec_vecToTensor, flattenCtx_nil, map_zero]
  rfl

private theorem column_inner {n : Nat} (u v : Vec n) :
    inner ℝ (column u) (column v) = inner ℝ u v := by
  change inner ℝ (castVec (by simp [Shape.size] : n = (C n).size) u)
    (castVec (by simp [Shape.size] : n = (C n).size) v) = _
  exact inner_castVec_castVec _ _ _

private theorem column_norm {n : Nat} (v : Vec n) : ‖column v‖ = ‖v‖ := by
  have h := column_inner v v
  rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq] at h
  nlinarith [norm_nonneg (column v), norm_nonneg v]

private theorem relu_column {n : Nat} (v : Vec n) :
    (vecOfFun fun k : Fin (C n).size ↦ Activation.Math.reluSpec (column v k)) =
      column (reluVec v) := by
  ext k
  change Activation.Math.reluSpec (column v k) = column (reluVec v) k
  simp only [column_apply]
  rfl

private theorem relu_jvp_column {n : Nat} (v d : Vec n) :
    (vecOfFun fun k : Fin (C n).size ↦
      column d k * Activation.Math.reluDerivSpec (column v k)) =
      column (reluDerivCLM v d) := by
  ext k
  change column d k * Activation.Math.reluDerivSpec (column v k) = column (reluDerivCLM v d) k
  simp only [column_apply]
  exact (reluDerivCLM_apply v d _).symm

private theorem snoc_flatten {Γ : List Shape} {s : Shape}
    (xs : TensorPack ℝ Γ) (v : Vec s.size) :
    snocCtx (flattenCtx xs) v = flattenCtx (xs.snoc (vecToTensor v)) := by
  induction xs with
  | nil =>
      simp only [TensorPack.snoc, flattenCtx_cons, tensorToVec_vecToTensor, flattenCtx_nil]
      change snocCtx (Γ := []) (τ := s) 0 v = appendVec v 0
      ext k
      change Fin.append (0 : Fin 0 → ℝ) v (Fin.cast _ k) =
        Fin.append v (0 : Fin 0 → ℝ) k
      have hk : Fin.cast (by simp : s.size = 0 + s.size) k = Fin.natAdd 0 k := by
        apply Fin.ext
        change k.val = 0 + k.val
        omega
      erw [hk, Fin.append_right]
      have hk' : k = Fin.castAdd 0 k := rfl
      conv_rhs => erw [hk', Fin.append_left]
  | @cons a Γ x xs ih =>
      rw [flattenCtx_cons, ← appendVec_snocCtx, ih]
      simp only [TensorPack.snoc, flattenCtx_cons]
      rfl

def idx {Γ : List Shape} (k : Nat) (hk : k < Γ.length) : Idx Γ (Γ.get ⟨k, hk⟩) :=
  ⟨⟨k, hk⟩, rfl⟩

/-- Two matrix products, two bias additions, the selected ReLU, and the MSE loss. -/
def graph (i h o : Nat) : Graph (Inputs i h o) (Saved h o) :=
  let g1 := Graph.snoc Graph.nil (TapeNodes.matmul (idx 0 (by simp [Inputs]))
    (idx 4 (by simp [Inputs])))
  let g2 := Graph.snoc g1 (TapeNodes.add (idx 6 (by simp [Inputs])) (idx 1 (by simp [Inputs])))
  let g3 := Graph.snoc g2 (TapeNodes.relu (idx 7 (by simp [Inputs])))
  let g4 := Graph.snoc g3 (TapeNodes.matmul (idx 2 (by simp [Inputs])) (idx 8 (by simp [Inputs])))
  let g5 := Graph.snoc g4 (TapeNodes.add (idx 9 (by simp [Inputs])) (idx 3 (by simp [Inputs])))
  Graph.snoc g5 (TapeNodes.mseLoss (idx 10 (by simp [Inputs])) (idx 5 (by simp [Inputs])))

private theorem mse_column {n : Nat} (v t : Vec n) :
    (1 / (TorchLean.Tensor.meanDenominator (C n) : ℝ)) * ‖column v - column t‖ ^ 2 =
      mse t v := by
  rw [← map_sub, column_norm]
  by_cases hn : n = 0
  · subst n
    have hv : v = 0 := Subsingleton.elim _ _
    have ht : t = 0 := Subsingleton.elim _ _
    simp [hv, ht, mse]
  · rw [TorchLean.Tensor.meanDenominator_of_size_pos (by
      simpa [Shape.size] using (Nat.pos_of_ne_zero hn))]
    simp [Shape.size, mse, div_eq_mul_inv]

/-- The output slot of the primitive graph computes the original scalar objective. -/
theorem forward {i h o : Nat} (p : MlpMseParameters i h o) (x : Vec i) (t : Vec o) :
    CtxVec.get (idx 11 (by simp [Inputs, Saved])) ((graph i h o).evalVec (inputVector p x t)) =
      vecScalarCLM (mlpMseObjective x t p) := by
  simp only [graph, Graph.evalVec, castCtxVec_rfl, inputVector,
    snoc_flatten, TensorPack.snoc, matmul, add, relu, elemwise, mseLoss,
    Node.forwardVec_ofFn, CtxVec.get_flattenCtx, getIdx, idx,
    TensorPack.get, Inputs, C, M, List.get, List.nil_append, List.cons_append,
    Tensor.cast_shape_self, tensorToVec_vecToTensor,
    Matmul.forward_eq_matmulVec, matmul_column, ← map_add, relu_column, mse_column]
  rfl

private theorem mse_jvp_column {n : Nat} (v t d : Vec n) :
    (1 / (TorchLean.Tensor.meanDenominator (C n) : ℝ)) *
      (2 * inner ℝ (column v - column t) (column d)) = inner ℝ (mseGrad v t) d := by
  rw [← map_sub, column_inner]
  by_cases hn : n = 0
  · subst n
    have hd : d = 0 := Subsingleton.elim _ _
    simp [hd]
  · rw [TorchLean.Tensor.meanDenominator_of_size_pos (by
      simpa [Shape.size] using (Nat.pos_of_ne_zero hn))]
    simp [Shape.size, mseGrad, inner_smul_left]
    ring

private theorem matmul_zero {m n k : Nat} (a : Vec (M m n).size) :
    Matmul.matmulVec a (0 : Vec (M n k).size) = 0 := by
  ext j
  simp [Matmul.matmulVec, vecOfFun]

/-- The primitive JVP composes the selected chain even at zero preactivations. -/
theorem jvp {i h o : Nat} (p d : MlpMseParameters i h o) (x : Vec i) (t : Vec o) :
    CtxVec.get (idx 11 (by simp [Inputs, Saved]))
      ((graph i h o).jvpVec (inputVector p x t) (parameterMap d)) =
        vecScalarCLM (mlpMseSelectedLinearization x t p d) := by
  rw [parameterMap_apply]
  simp only [graph, Graph.jvpVec, Graph.evalVec, castCtxVec_rfl, inputVector,
    snoc_flatten, TensorPack.snoc, matmul, add, relu, elemwise, mseLoss,
    Node.jvpVec_ofFn, Node.forwardVec_ofFn, CtxVec.get_flattenCtx, getIdx, idx,
    TensorPack.get, Inputs, C, M, List.get, List.nil_append, List.cons_append,
    Tensor.cast_shape_self, tensorToVec_vecToTensor,
    Matmul.tensorToVec_add_spec_mat, Matmul.forward_eq_matmulVec, matmul_column,
    map_zero, matmul_zero, ← map_add, add_zero, sub_zero,
    relu_column, relu_jvp_column, mse_jvp_column]
  rw [selectedLinearization_apply]
  rfl

/-- Run the existing reverse evaluator, then retain the four parameter cotangents. -/
def reverse {i h o : Nat} (p : MlpMseParameters i h o) (x : Vec i) (t : Vec o) :
    MlpMseParameters i h o :=
  parameterMap.adjoint ((graph i h o).backpropVec (inputVector p x t)
    (CtxVec.single (idx 11 (by simp [Inputs, Saved])) (vecScalarCLM 1)))

/-- The composition of primitive VJPs returns all four closed-form backward blocks. -/
theorem reverse_eq_backprop {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (t : Vec o) :
    reverse p x t = mlpMseBackprop p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x t := by
  apply ext_inner_left ℝ
  intro d
  rw [reverse, ContinuousLinearMap.adjoint_inner_right,
    ← Graph.backprop_correct_inner, CtxVec.inner_get_single, jvp,
    ← mlpMseSelectedLinearization_adjoint_one x t p,
    ContinuousLinearMap.adjoint_inner_right]
  change inner ℝ (vecOfFun fun _ ↦ mlpMseSelectedLinearization x t p d)
    (vecOfFun fun _ ↦ (1 : ℝ)) = inner ℝ (mlpMseSelectedLinearization x t p d) (1 : ℝ)
  simp only [inner_scalarVec_left, vecOfFun_ofLp, Real.inner_apply]

/-- The primitive reverse output is a reachable gradient, including at ReLU kinks. -/
theorem reverse_isLimitingGradient {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (t : Vec o) :
    IsLimitingGradient (mlpMseObjective x t) p (reverse p x t) := by
  rw [reverse_eq_backprop]
  exact mlpMseBackprop_isLimitingGradient p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd x t

/-- The primitive reverse output satisfies Clarke's support inequalities. -/
theorem reverse_mem_clarkeGradient {i h o : Nat} (p : MlpMseParameters i h o)
    (x : Vec i) (t : Vec o) :
    reverse p x t ∈ clarkeGradient (mlpMseObjective x t) p :=
  isLimitingGradient_mem_clarkeGradient (reverse_isLimitingGradient p x t)

end PrimitiveMlp

end

end Proofs.Autograd
