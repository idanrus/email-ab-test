import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from scipy.stats import ttest_ind

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

def custom_chisquare(counts, ratios = None):
    """Считает χ² для сравнения наблюдаемых частот с ожидаемыми."""
    counts = np.array(counts)
    total = counts.sum()
    k = len(counts)

    if ratios is None:
        E = total / k
    else:
        E = np.array(ratios) * total
    O = counts
    df = k - 1
    chi2 = ((O - E) ** 2 / E).sum()
    p_value = 1 - stats.chi2.cdf(chi2, df = df)
    return chi2, p_value

def custom_chi2_contingency(crosstab):
    """Вручную считает критерий χ² независимости для таблицы сопряжённости."""
    rows_sum = crosstab.agg('sum', axis = 1).values
    col_sum = crosstab.agg('sum', axis = 0).values
    total = col_sum.sum()
    E = np.array([[0. for j in range(len(col_sum))] for i in range(len(rows_sum))])
    for i in range(len(rows_sum)):
        for j in range(len(col_sum)):
            E[i,j] = rows_sum[i] * col_sum[j] / total
    chi2 = ((crosstab.values - E) ** 2 / E).sum()
    dof = (len(rows_sum) - 1) * (len(col_sum) - 1)
    p_value = 1 - stats.chi2.cdf(chi2, df = dof)
    return chi2, p_value, dof, E

def custom_anova(*datas):
    """Вручную считает F-статистику и p-value однофакторной ANOVA."""
    datas = [np.asarray(data) for data in datas]
    n = sum(len(g) for g in datas)
    k = len(datas)
    general_mean = np.concatenate(datas).mean()
    ssb = sum(len(g) * (g.mean() - general_mean) ** 2 for g in datas)
    ssw = sum(np.sum((g - np.mean(g)) ** 2) for g in datas)
    df_b = k - 1
    df_w = n-k
    f_stat = (ssb/df_b) / (ssw/df_w)
    p_value = 1 - stats.f.cdf(f_stat, df_b, df_w)
    return f_stat, p_value

def prop_ztest_wrapper(a, b, alternative='two-sided'):
    """Передаёт числа событий и размеры групп в z-тест statsmodels."""
    count = [np.sum(a), np.sum(b)]
    nobs = [len(a), len(b)]
    return proportions_ztest(count, nobs, alternative=alternative)

def run_synthetic_experiments(values, sample_size, rng, test, effect=0, is_proportion=False, n_iters=2000):
    """Повторяет случайные разбиения и возвращает p-value с заданным эффектом или без него."""
    p_values = []
    values = np.asarray(values, dtype=float)
    for _ in range(n_iters):
        a, b = rng.choice(values, size=(2, sample_size), replace=False)

        if effect != 0:
            if is_proportion:
                if effect > 0:
                    zeros_mask = (b == 0)
                    b[zeros_mask] = rng.binomial(1, effect, size=zeros_mask.sum())
                else:
                    ones_mask = (b == 1)
                    b[ones_mask] = rng.binomial(1, 1 + effect, size=ones_mask.sum())
            else:
                b = b + effect

        stat, p_value = test(a, b, alternative='two-sided')
        p_values.append(p_value)

    return np.array(p_values)

def custom_ttest(a, b, alternative = 'two-sided'):
    """Вручную считает статистику, p-value и степени свободы двустороннего теста Уэлча."""
    x1 = a.mean()
    x2 = b.mean()
    std1 = a.std(ddof = 1)
    std2 = b.std(ddof = 1)
    n1 = a.size
    n2 = b.size

    se = (std1**2 / n1 + std2**2 / n2)

    k = se**2 / (std1**4 / (n1**2 * (n1 - 1)) + std2**4 / (n2**2*(n2 - 1)))

    t = (x1-x2) / np.sqrt(se)
    p_value = 2 * (1 - stats.t(df = k).cdf(abs(t)))

    return t, p_value, k

def custom_proportion_ztest(a, b, alternative = 'two-sided'):
    """Вручную считает двусторонний z-тест для двух долей с общей оценкой доли при H0."""
    n1 = len(a)
    n2 = len(b)
    p1 = np.sum(a) / n1
    p2 = np.sum(b) / n2
    P = (p1 * n1 + p2 * n2) / (n1 + n2)

    Z = (p1 - p2) / np.sqrt(P*(1-P)*(1/n1 + 1/n2))
    p_value = 2*(1 - stats.norm.cdf(abs(Z)))

    return Z, p_value


def welch_for_aa(a, b, alternative='two-sided'):
    """Возвращает статистику Уэлча и p-value для симуляций."""
    t, p_value, _ = custom_ttest(a, b, alternative=alternative)
    return t, p_value


def simulation_rate(events, confidence=0.95):
    """Возвращает частоту события в симуляции и интервал Уилсона."""
    events = np.asarray(events, dtype=bool)
    count = events.sum()
    lower, upper = proportion_confint(count, events.size, alpha=1 - confidence, method='wilson')
    return count / events.size, lower, upper


def welch_scipy(a, b):
    """Считает двусторонний тест Уэлча через SciPy."""
    return ttest_ind(a, b, equal_var=False)


def covariate_balance(df, group_col, numeric_features, categorical_features):
    """Собирает средние и доли по группам и результаты сверки со SciPy."""
    grouped = df.groupby(group_col)
    groups = list(grouped.size().index)
    rows = []
    checks = {}

    for feature, label in numeric_features.items():
        samples = [grouped.get_group(group)[feature].to_numpy() for group in groups]
        statistic, p_value = custom_anova(*samples)
        reference = stats.f_oneway(*samples)
        checks[feature] = statistic, reference.statistic, p_value, reference.pvalue
        rows.append([feature, label, *[sample.mean() for sample in samples], f'{p_value:.3f}'])

    for feature in categorical_features:
        counts = pd.crosstab(df[group_col], df[feature]).reindex(groups)
        statistic, p_value, _, _ = custom_chi2_contingency(counts)
        reference = stats.chi2_contingency(counts)
        checks[feature] = statistic, reference.statistic, p_value, reference.pvalue
        shares = counts.div(counts.sum(axis=1), axis=0) * 100

        for i, (level, group_shares) in enumerate(shares.T.iterrows()):
            feature_p = f'{p_value:.3f}' if i == 0 else ''
            rows.append([feature, f'{level} (%)', *group_shares, feature_p])

    balance = pd.DataFrame(rows, columns=['Признак', 'Значение', *groups, 'p-value'])
    columns = ['Наша статистика', 'Статистика SciPy', 'Наш p-value', 'p-value SciPy']
    comparisons = pd.DataFrame.from_dict(checks, orient='index', columns=columns)
    return balance.set_index(['Признак', 'Значение']), comparisons.rename_axis('Признак')


def run_aa_checks(metrics, sample_size, rng, n_iters):
    """Сверяет тесты на одном разбиении и собирает p-value симуляций A/A."""
    p_values = {}
    checks = {}
    for name, values, test, library_test in metrics:
        a, b = rng.choice(values, size=(2, sample_size), replace=False)
        checks[name] = tuple(test(a, b)), tuple(library_test(a, b))
        p_values[name] = run_synthetic_experiments(values, sample_size, rng, test, n_iters=n_iters)
    return p_values, checks
