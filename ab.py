import numpy as np
from scipy import stats


def MDE(alpha, beta, std, n):
    """Минимальный абсолютный эффект, который двусторонний тест найдёт с мощностью 1 - beta при n клиентах в каждой группе."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(1 - beta)
    return (z_alpha + z_beta) * std * np.sqrt(2.0 / n)


def n_for_test(alpha, beta, var, mde):
    """Сколько клиентов нужно в каждой группе, чтобы найти эффект mde (округляем вверх)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(1 - beta)
    return np.ceil((z_alpha + z_beta) ** 2 * (2 * var) / mde ** 2)
