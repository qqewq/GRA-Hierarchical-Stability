import numpy as np


def compute_foam(state):
    """
    Заглушка: вычисление «пены» как суммы квадратов отклонений от нуля.
    Замените на реальную метрику вашей GRA-системы.
    """
    return np.sum(state ** 2)


def gra_nullify(state):
    """
    Заглушка GRA-обнуления: уменьшаем пену простым сжатием.
    """
    return state * 0.9  # пример


def grad_connection(level_a, level_b, dPhi):
    """
    Заглушка градиента связи C_l(a, b) = ||a - proj(b)||^2.
    Здесь просто возвращаем разность состояний, умноженную на dPhi.
    """
    return dPhi * (level_b.state - level_a.state)
