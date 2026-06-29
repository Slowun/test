"""pyquant.barrier: ценообразование барьерных опционов методом Монте-Карло.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (количественная библиотека, ветка Monte-Carlo на torch).
НАЗНАЧЕНИЕ:  расчёт выплаты и цены барьерных опционов (knock-in/knock-out,
             up/down) по симулированным траекториям базового актива.
ИМПОРТИРУЕТ: torch.
ЭКСПОРТИРУЕТ: barrier_option_payoff, price_barrier_option.
КЕМ ИСПОЛЬЗУЕТСЯ: оценка экзотических инструментов поверх pyquant.heston_sim.
=============================================================================
"""

import torch                                            # тензоры


def barrier_option_payoff(
    paths: torch.Tensor,                                # траектории базового актива (N путей × M+1 шагов)
    strike: torch.Tensor,                               # страйк
    barrier: torch.Tensor,                              # уровень барьера
    barrier_type: str,                                  # тип барьера: up/down × in/out
    call: bool                                          # call или put
) -> torch.Tensor:
    if barrier_type not in ('up-in', 'up-out', 'down-in', 'down-out'):  # валидация типа барьера
        raise ValueError("`barrier_type` must be one of: 'up-in', 'up-out', 'down-in', 'down-out'.")
    pass

    if call:                                            # ванильная выплата на терминальной цене:
        payoff = torch.maximum(paths[:, -1] - strike, torch.zeros_like(paths[:, -1]))  # max(S_T-K,0)
    else:
        payoff = torch.maximum(strike - paths[:, -1], torch.zeros_like(paths[:, -1]))  # max(K-S_T,0)

    if barrier_type == 'up-in':                         # активируется при достижении барьера сверху
        condition = torch.max(paths, dim=1).values >= barrier
    elif barrier_type == 'up-out':                      # гасится при достижении барьера сверху
        condition = torch.max(paths, dim=1).values < barrier
    elif barrier_type == 'down-in':                     # активируется при достижении барьера снизу
        condition = torch.min(paths, dim=1).values <= barrier
    else:  # down-out                                   # гасится при достижении барьера снизу
        condition = torch.min(paths, dim=1).values > barrier

    payoff *= condition                                 # обнулить выплату там, где условие барьера не выполнено
    return payoff                                       # вектор выплат по путям


def price_barrier_option(
    paths: torch.Tensor,
    strike: torch.Tensor,
    maturity: torch.Tensor,
    rate: torch.Tensor,
    barrier: torch.Tensor,
    barrier_type: str,
    call: bool
) -> torch.Tensor:
    """Compute the price of a barrier option.

    Args:
        paths: Simulated paths of the underlying.
        strike: Strike price.
        maturity: Expiration time in years (corresponding to `paths[:, -1]`).
        rate: Risk-free rate.
        barrier: Barrier price.
        barrier_type: One of 'up-in', 'up-out', 'down-in', 'down-out'.
        call: If `True`, price call option, otherwise price put option.

    Shape:
        - paths: (N, M + 1), where N is the number of paths, M is the number of
            time steps. The last time point is assumed to be the expiration time.
        - strike, maturity, rate, barrier: (1, ).

    Returns:
        Price of the option. Shape: (1, ).
    """
    if barrier_type not in ('up-in', 'up-out', 'down-in', 'down-out'):  # валидация типа барьера
        raise ValueError("`barrier_type` must be one of: 'up-in', 'up-out', 'down-in', 'down-out'.")

    payoff = barrier_option_payoff(paths, strike, barrier, barrier_type, call)  # выплаты по путям
    return torch.exp(-rate*maturity) * torch.mean(payoff)  # цена = дисконтированное матожидание выплаты
