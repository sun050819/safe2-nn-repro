# Minimal Reproducibility Package

## Software Stack
- Python 3.11
- PyTorch 2.2
- CUDA 12.1 (optional, for GPU acceleration)
- Self-contained interval arithmetic implementation (no external library required)

## Contents
- scripts/: core verifier and experiment scripts
- data/mnist/: MNIST MLP raw per-run outputs (rates, h_max, runtime, gate ablation)
- data/two_moons/: two-moons existence experiment raw outputs
- data/scaling/: structured vs dense scaling timing data

## Reproduce
1. Install dependencies: pip install torch numpy
2. Run: python scripts/run_experiments.py (MNIST)
3. Run: python scripts/existence_experiment.py (two-moons, n=200)
4. Run: python scripts/run_scaling.py (scaling)
5. Run: python scripts/safe2_assisted_bab.py (SAFE2-assisted BaB)

## Key Results
- Two-moons (ε=0.5, n=200): classical 64/200 (32%), SAFE2 78/200 (39%), McNemar p=0.013
- SAFE2-assisted BaB: 90.9% node reduction on two-moons (5564 → 507 nodes)

## Notes
- CIFAR-10 training scripts are not included (compute-heavy; available upon acceptance).
- All outputs are generated deterministically given the fixed random seeds in the scripts.
- torch.use_deterministic_algorithms(True) is enabled for reproducibility.
