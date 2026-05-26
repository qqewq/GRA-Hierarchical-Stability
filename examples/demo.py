import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.optimizer import Level, HierarchicalStabilityOptimizer

# Создаём тестовую иерархию из 4 уровней с случайными состояниями
np.random.seed(42)
hierarchy = [Level(np.random.randn(3)) for _ in range(4)]

print("Начальная пена:")
for i, lvl in enumerate(hierarchy):
    print(f"  Уровень {i}: {lvl.foam():.4f}")

opt = HierarchicalStabilityOptimizer(hierarchy, lr=0.05)
opt.optimize(max_iter=200, verbose=True)

print("\nФинальная пена:")
for i, lvl in enumerate(hierarchy):
    print(f"  Уровень {i}: {lvl.foam():.6f}")

sens = opt.get_sensitivity_map()
print("\nКарта чувствительности (dPhi/dh):")
for i, s in enumerate(sens):
    regime = "схлопывание" if s < -1e-6 else ("стагнация" if abs(s) < 1e-6 else "ДЕГРАДАЦИЯ")
    print(f"  Переход {i}->{i+1}: {s:+.6f} ({regime})")
