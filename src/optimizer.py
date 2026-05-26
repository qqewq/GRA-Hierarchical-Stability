import numpy as np
from .foam_utils import gra_nullify, compute_foam, grad_connection


class Level:
    """Простой контейнер для состояния одного уровня иерархии."""
    def __init__(self, state):
        self.state = np.array(state, dtype=float)

    def foam(self):
        return compute_foam(self.state)


class HierarchicalStabilityOptimizer:
    """
    Иерархический оптимизатор, реализующий устойчивую GRA-обнулёнку.
    Параметры:
        hierarchy : список объектов Level
        alpha, beta, gamma : веса функционала (задел на будущее)
        lr : шаг обучения
    """
    def __init__(self, hierarchy, alpha=None, beta=None, gamma=None, lr=0.01):
        self.h = hierarchy
        self.alpha = alpha or [1.0] * len(hierarchy)
        self.beta = beta or [1.0] * (len(hierarchy) - 1)
        self.gamma = gamma or 1.0
        self.lr = lr

    def step(self):
        # 1. GRA-обнуление на каждом уровне
        for level in self.h:
            level.state = gra_nullify(level.state)

        # 2. Коррекция градиента dPhi/dh
        for l in range(len(self.h) - 1):
            dPhi = self.h[l+1].foam() - self.h[l].foam()
            if abs(dPhi) > 1e-6:
                grad = grad_connection(self.h[l], self.h[l+1], dPhi)
                self.h[l+1].state -= self.lr * grad

        # 3. Контроль d^2Phi/dh^2 > 0
        for l in range(1, len(self.h) - 1):
            f_prev = self.h[l-1].foam()
            f_curr = self.h[l].foam()
            f_next = self.h[l+1].foam()
            d2Phi = f_next - 2 * f_curr + f_prev
            if d2Phi <= 0:
                # Регуляризация: подталкиваем состояние к выпуклой конфигурации
                reg = 0.01 * (self.h[l+1].state + self.h[l-1].state - 2 * self.h[l].state)
                self.h[l].state += self.lr * reg

    def optimize(self, max_iter=100, verbose=False):
        for it in range(max_iter):
            old_states = [lvl.state.copy() for lvl in self.h]
            self.step()
            if verbose:
                foams = [lvl.foam() for lvl in self.h]
                print(f"Iter {it:3d} | Foam per level: {[f'{f:.4f}' for f in foams]}")
            # Простейший критерий остановки – малое изменение состояний
            max_diff = max(np.max(np.abs(old_states[i] - self.h[i].state))
                           for i in range(len(self.h)))
            if max_diff < 1e-8:
                if verbose:
                    print(f"Сошлось за {it+1} итераций.")
                break

    def get_sensitivity_map(self):
        """Возвращает список dPhi/dh для всех переходов."""
        sens = []
        for l in range(len(self.h) - 1):
            dPhi = self.h[l+1].foam() - self.h[l].foam()
            sens.append(dPhi)
        return sens
