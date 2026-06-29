"""pyquant.lsm: цена американского пут-опциона методом Лонгстаффа-Шварца (LSM).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (количественная библиотека, ветка Monte-Carlo на torch).
НАЗНАЧЕНИЕ:  оценка американских опционов с правом досрочного исполнения через
             Least-Squares Monte Carlo: на каждом шаге аппроксимируем стоимость
             продолжения C(S,t) регрессией и сравниваем с немедленной выплатой.
             Использует два независимых набора путей (регрессия/оценка) для
             уменьшения смещения.
ИМПОРТИРУЕТ: torch, dataclass, typing.
ЭКСПОРТИРУЕТ: LSMResult, price_american_put_lsm (+ внутренние _lsm_*).
КЕМ ИСПОЛЬЗУЕТСЯ: оценка американских опционов поверх pyquant.heston_sim.
ССЫЛКА:      [Seydel2017] алгоритмы 3.14/3.15.
=============================================================================

References:
    [Seydel2017] Seydel, Rüdiger. Tools for computational finance.
    Sixth edition. Springer, 2017. Section 3.6.3.
"""


import torch                                            # тензоры и автодифференцирование
from dataclasses import dataclass                       # контейнер результата
from typing import Optional, Union                      # аннотации типов


@dataclass
class LSMResult:
    """Computation results of LSM algorithm.

    Attributes:
        option_price:
            Price of the option at the initial moment of time. Shape: (1, ).
        reg_poly_coefs:
            Polynomial coefficients that were fitted on the regression step.
            Shape: (N_STEPS + 1, REG_POLY_DEGREE + 1). `reg_poly_coefs[0]` and
            `reg_poly_coefs[-1]` are always NaN for the following reasons.
            At the initial moment of time, all paths have the same value, so 
            continuation value can only be calculated for this single point. This
            value is contained in the attribute `initial_cont_value`.
            At the final moment of time, the option is exercised, so continuation
            value is not defined.
        initial_cont_value:
            Continuation value at the initial moment of time. Shape: (1, ).
            At this moment, all paths have the same value, so continuation value
            can only be calculated for this single point.
        reg_x_vals:
            Values of in-the-money subset of underlying paths at each moment of time.
            If there were no ITM paths at some moment of time, the corresponding
            list item will be `None`. Length: N_STEPS + 1.
        reg_y_vals:
            Values of dependent variable for regression for in-the-money
            subset of underlying paths at each moment of time. If there were
            no ITM paths at some moment of time, the corresponding list item
            will be `None`. Length: N_STEPS + 1.

    Notes:
        - `N_STEPS` is the number of time steps in paths of the underlying asset
            which were passed as the argument for `price_american_put_lsm()`
            function.
        - `REG_POLY_DEGREE` is the value of the eponymous argument of
            `price_american_put_lsm()` function.
        - For each index `i`, the tensors `reg_x_vals[i]` and `reg_y_vals[i]`
            have the same length. For explanation of these variables,
            see [Seydel2017], section 3.6.3, algorithm 3.14, item (c).
    """
    option_price: torch.Tensor                          # цена опциона в начальный момент
    reg_poly_coefs: torch.Tensor                        # коэффициенты регрессии по каждому шагу
    initial_cont_value: torch.Tensor                    # стоимость продолжения в t=0
    reg_x_vals: Optional[list[Union[torch.Tensor, None]]] = None  # ITM-значения путей (для диагностики)
    reg_y_vals: Optional[list[Union[torch.Tensor, None]]] = None  # зависимая переменная регрессии


def price_american_put_lsm(
    paths_regression: torch.Tensor,
    paths_pricing: torch.Tensor,
    dt: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    reg_poly_degree: int = 3,
    return_extra: bool = False
) -> LSMResult:
    """Calculates the price of American put option using the LSM algorithm.

    This is the modification of LSM algorithm where it's divided into two steps:
    1. Regression. Find approximation of continuation value C(S, t) via regression.
    2. Pricing. Use the continuation value from step 1 as a sort of barrier:
       when the payoff for a given path crosses C(S, t), exercise the option on this path.

    Two separate sets of paths are used on each step to reduce the bias. These
    sets may have different number of paths, but they must have the same number
    of time steps.

    Gradient computation is disabled on the regression step.

    Args:
        paths_regression: Paths of underlying asset, starting from the same
            point S0 at initial moment of time. Used on regression step.
        paths_pricing: Paths of underlying asset, starting from the same
            point S0 at initial moment of time. Used on pricing step.
        dt: Time step.
        strike: Option strike price.
        rate: Risk-free rate. Note that the input paths must be generated with
            the same risk-free rate as the value of this parameter.
        reg_poly_degree: Degree of polynomial for regression.
        return_extra: If `True`, in addition to other values return the
            in-the-money points which are used as the data for regression.

    Shape:
        - paths_regression, paths_pricing: (N, M + 1), where N is the number of
            generated paths, M is the number of time steps.
        - dt: (1, )
        - strike: (1, )
        - rate: (1, )

    Returns:
        Computation result of LSM algorithm, represented as `LSMResult` object.
        Contains computed option price, polynomial coefficients that were fitted
        on the regression step and continuation value at initial moment of time.
        If `return_extra` is `True`, also contains the in-the-money points which
        were used as the data for regression.
    """
    if not (torch.all(paths_regression[:, 0] == paths_regression[0, 0])  # все пути стартуют из одного S0
            and torch.all(paths_pricing[:, 0] == paths_pricing[0, 0])):
        raise ValueError('Paths of the underlying must start from the same value at initial moment of time')

    if paths_regression.shape[1] != paths_pricing.shape[1]:  # одинаковое число шагов времени
        raise ValueError('`paths1` and `paths2` must have the same number of time steps')

    with torch.no_grad():                               # регрессия без autograd (только подбор политики исполнения)
        result_reg_step = _lsm_regression_step(
            paths_regression, dt, strike, rate, reg_poly_degree, return_extra)

    return _lsm_pricing_step(paths_pricing, dt, strike, rate, reg_poly_degree, result_reg_step)  # оценка на втором наборе


def _lsm_regression_step(
    paths: torch.Tensor,
    dt: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    reg_poly_degree: int,
    return_extra: bool = False
) -> LSMResult:
    """Implementation of algorithm 3.15 from [Seydel2017]."""
    n_paths = paths.shape[0]                             # число путей
    n_steps = paths.shape[1] - 1                         # число шагов времени

    cashflow = torch.where(paths[:, -1] < strike, strike - paths[:, -1], 0)  # выплата на экспирации (put)
    tau = n_steps * torch.ones(n_paths, dtype=torch.int64)  # оптимальный момент остановки (изначально — конец)

    reg_poly_coefs = torch.zeros((paths.shape[1], reg_poly_degree + 1))  # коэффициенты регрессии по шагам
    reg_poly_coefs[-1] *= torch.nan  # на экспирации стоимость продолжения не определена
    reg_poly_coefs[0] *= torch.nan   # в t=0 все пути совпадают — отдельная обработка

    if return_extra:                                    # опциональный сбор данных регрессии
        reg_x_vals = [None] * n_steps
        reg_y_vals = [None] * n_steps

    for j in range(n_steps - 1, 0, -1):                 # идём назад по времени (динамическое программирование)
        itm_mask = paths[:, j] < strike                # пути «в деньгах» (put ITM)
        if torch.sum(itm_mask) == 0:                    # нет ITM-путей — пропустить шаг
            continue
        paths_itm = paths[:, j][itm_mask]              # значения базового на ITM-путях

        A = torch.vander(paths_itm, N=reg_poly_degree + 1)  # матрица Вандермонда (полиномиальный базис)
        y = torch.exp(-rate * (tau[itm_mask] - j) * dt) * cashflow[itm_mask]  # дисконтированный будущий cashflow
        fit_params = torch.linalg.lstsq(A, y).solution  # МНК-регрессия стоимости продолжения
        C_hat = torch.matmul(A, fit_params)  # оценка стоимости продолжения
        reg_poly_coefs[j] = fit_params                  # сохранить коэффициенты шага

        if return_extra:
            reg_x_vals[j] = paths_itm
            reg_y_vals[j] = y

        payoff_itm_now = strike - paths_itm            # немедленная выплата при исполнении сейчас
        stop_now_mask = (payoff_itm_now >= C_hat)      # выгоднее исполнить, чем продолжать
        cashflow[itm_mask] = torch.where(stop_now_mask, payoff_itm_now, cashflow[itm_mask])  # обновить cashflow
        tau[itm_mask] = torch.where(stop_now_mask, j, tau[itm_mask])  # обновить момент остановки

    C_hat = torch.mean(torch.exp(-rate * tau * dt) * cashflow)  # стоимость продолжения в t=0
    payoff_now = torch.maximum(strike - paths[0, 0], torch.tensor(0.0))  # выплата при немедленном исполнении
    option_price = torch.maximum(payoff_now, C_hat)    # цена = max(исполнить, продолжать)

    result = LSMResult(option_price, reg_poly_coefs, C_hat)  # собрать результат (политику исполнения)
    if return_extra:
        result.reg_x_vals = reg_x_vals
        result.reg_y_vals = reg_y_vals
    return result


def _lsm_pricing_step(
    paths: torch.Tensor,
    dt: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    reg_poly_degree: int,
    result_reg_step: LSMResult
) -> LSMResult:
    n_paths = paths.shape[0]                             # число путей оценки
    n_steps = paths.shape[1] - 1                         # число шагов времени
    payoff = torch.zeros(n_paths)                        # дисконтированная выплата по каждому пути
    stopped_mask = torch.zeros(n_paths, dtype=torch.bool)  # уже исполнённые пути

    payoff_now = torch.maximum(strike - paths[0, 0], torch.tensor(0.0))  # выплата немедленного исполнения в t=0
    if torch.all(payoff_now > result_reg_step.initial_cont_value):  # если в t=0 выгоднее исполнить — это и есть цена
        option_price = payoff_now
    else:
        for j in range(1, n_steps - 1):                 # идём вперёд по времени, применяя политику исполнения
            itm_mask = (paths[:, j] < strike) & (~stopped_mask)  # ITM и ещё не исполнённые
            if itm_mask.numel() == 0:
                continue
            paths_itm = paths[:, j][itm_mask]

            vander = torch.vander(paths_itm, N=reg_poly_degree + 1)  # базис из регрессии
            cont_value = torch.matmul(vander, result_reg_step.reg_poly_coefs[j])  # стоимость продолжения по политике

            payoff_itm_now = strike - paths_itm        # выплата при исполнении сейчас
            stop_now_mask = (payoff_itm_now >= cont_value)  # критерий исполнения
            payoff[itm_mask] = torch.where(
                stop_now_mask,
                payoff_itm_now * torch.exp(-rate * j * dt),  # дисконтировать выплату на момент исполнения
                payoff[itm_mask]
            )
            stopped_mask[itm_mask] |= stop_now_mask     # отметить исполнённые пути

        # последний шаг — экспирация: исполняем все ещё не исполненные пути
        stop_now_mask = ~stopped_mask
        payoff_now = torch.maximum(strike - paths[:, -1], torch.zeros(n_paths))  # терминальная выплата
        payoff = torch.where(
            stop_now_mask,
            payoff_now * torch.exp(-rate * n_steps * dt),  # дисконт на экспирацию
            payoff
        )
        option_price = torch.mean(payoff)              # цена = среднее по путям
    result_reg_step.option_price = option_price        # записать цену в результат
    return result_reg_step
