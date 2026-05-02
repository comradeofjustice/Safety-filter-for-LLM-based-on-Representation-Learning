"""DeepSafe: Advanced Safety-Aware Projection Framework.

NeurIPS-level innovations:
  1. Hyperbolic Projection (Poincaré ball) for hierarchical safety taxonomy
  2. Optimal Transport (Sinkhorn) contrastive learning
  3. Prototype-based representation learning with learnable anchors
  4. Spectral decorrelation (Barlow Twins) for feature diversity
  5. Adversarial topic debiasing (gradient reversal)

Architecture:
  Frozen Embedding (1024d/4096d)
    -> Hyperbolic Exponential Map (tangent -> Poincaré ball)
    -> Möbius Linear Transform
    -> Log Map (hyperbolic -> Euclidean)
    -> Residual MLP with LayerNorm
    -> L2 Normalize -> Output (256d)
"""
