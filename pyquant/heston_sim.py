"""pyquant.heston_sim: Monte-Carlo симуляция модели Хестона (PyTorch).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (количественная библиотека, ветка Monte-Carlo на torch).
НАЗНАЧЕНИЕ:  генерация траекторий процесса CIR (дисперсия) и модели Хестона
             (цена+дисперсия) по схеме Quadratic-Exponential Андерсена. Нужна
             для оценки экзотических опционов (барьерных, американских) методом
             Монте-Карло, когда нет полуаналитической формулы.
ИМПОРТИРУЕТ: torch (тензоры/autograd) — модуль НЕ грузится, если torch нет.
ЭКСПОРТИРУЕТ: noncentral_chisquare, generate_cir, generate_heston.
КЕМ ИСПОЛЬЗУЕТСЯ: pyquant.barrier, pyquant.lsm (Монте-Карло ценообразование).
ССЫЛКИ:      [Andersen2007] QE-схема; [Grzelak2019] вывод формул.
=============================================================================

References:
    - [Grzelak2019] Oosterlee, C. W., & Grzelak, L. A. (2019). Mathematical
      modeling and computation in finance: with exercises and Python and
      MATLAB compute codes. World Scientific.

    - [Andersen2007] Andersen, L.B., 2007. Efficient simulation of the Heston
      stochastic volatility model. Available at SSRN 946405.
"""


import torch                                            # тензоры и автодифференцирование
from typing import Tuple, Optional                      # аннотации типов

__all__ = ['noncentral_chisquare', 'generate_cir', 'generate_heston']  # публичный API модуля


def noncentral_chisquare(
        df: torch.Tensor,
        nonc: torch.Tensor
) -> torch.Tensor:
    """ Generates samples from a noncentral chi-square distribution.
    Quadratic Exponential scheme from [Andersen2007] is used.

    Args:
        df: Degrees of freedom, must be > 0.
        nonc: Non-centrality parameter, must be >= 0.

    Returns:
        Tensor with generated tensor. Shape: same as `df` and `nonc`, if
        they have the same shape.
    """
    # algorithm is summarized in [Andersen2007, section 3.2.4]
    PSI_CRIT = 1.5  # порог переключения между квадратичной и экспоненциальной схемами
    m = df + nonc                                       # среднее распределения
    s2 = 2*df + 4*nonc                                  # дисперсия распределения
    psi = s2 / m.pow(2)                                 # коэффициент psi = дисперсия/среднее²
    # quadratic                                         # ветка для psi<=PSI_CRIT:
    psi_inv = 1 / psi
    b2 = 2*psi_inv - 1 + (2*psi_inv).sqrt() * (2*psi_inv - 1).sqrt()  # параметр b²
    a = m / (1 + b2)                                    # параметр a
    sample_quad = a * (b2.sqrt() + torch.randn_like(a)).pow(2)  # выборка a·(sqrt(b²)+Z)²
    # exponential                                       # ветка для psi>PSI_CRIT:
    p = (psi - 1) / (psi + 1)                           # вероятность точечной массы в нуле
    beta = (1 - p) / m                                  # параметр экспоненты
    rand = torch.rand_like(p)                           # равномерная выборка
    sample_exp = torch.where((p < rand) & (rand <= 1),  # обратное преобразование CDF
                             beta.pow(-1)*torch.log((1-p)/(1-rand)),
                             torch.zeros_like(rand))    # иначе ноль
    return torch.where(psi <= PSI_CRIT, sample_quad, sample_exp)  # выбрать схему поэлементно


def generate_cir(
        n_paths: int,
        timeline: torch.Tensor,
        init_state: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        eps: torch.Tensor,
        minimum_value: Optional[float] = None,
) -> torch.Tensor:
    """Generates paths of Cox-Ingersoll-Ross (CIR) process.

    CIR process is described by the SDE:
        dv(t) = κ·(θ - v(t))·dt + ε·sqrt(v(t))·dW(t)
    (see [Grzelak2019, section 8.1.2]).

    For path generation, Andersen's Quadratic Exponential scheme is used
    (see [Andersen2007], [Grzelak2019, section 9.3.4]).

    Args:
        n_paths: Number of paths to simulate.
        timeline: Time steps
        init_state: Initial states of the paths, i.e. v(0). Can be any shape.
        kappa: Parameter κ.
        theta: Parameter θ. Can be a scalar tensor (dim=0) or a tensor with shape
            matching timeline (n_steps + 1,) containing values for each timeline point.
        eps: Parameter ε. Can be a scalar tensor (dim=0) or a tensor with shape
            matching timeline (n_steps + 1,) containing values for each timeline point.
        minimum_value: On each step, the value of a process is clamped to the
            range [minimum_value, +∞). This may be needed in certain cases, e.g.
            for automatic differentiation.

    Returns:
        Simulated paths of CIR process. Shape: (*init_state.shape, n_paths, n_steps + 1),
        with time dimension at -2.
    """
    dt_steps = timeline.diff()                          # шаги по времени Δt
    n_steps = dt_steps.shape[0]                          # число шагов
    timeline_len = timeline.shape[0]                     # число узлов времени
    
    # theta и eps могут быть скаляром (постоянны) или вектором по узлам времени:
    if theta.dim() == 0:
        theta_is_scalar = True                          # один theta на все шаги
    elif theta.dim() == 1 and theta.shape[0] == timeline_len:
        theta_is_scalar = False                         # свой theta в каждом узле
    else:
        raise ValueError(f"theta must be a scalar (dim=0) or have shape ({timeline_len},) matching timeline, got {theta.shape}")
    
    if eps.dim() == 0:
        eps_is_scalar = True                            # один eps на все шаги
    elif eps.dim() == 1 and eps.shape[0] == timeline_len:
        eps_is_scalar = False                           # свой eps в каждом узле
    else:
        raise ValueError(f"eps must be a scalar (dim=0) or have shape ({timeline_len},) matching timeline, got {eps.shape}")
    
    # массив траекторий: ось времени на позиции -2, число путей на -? согласно форме init_state
    paths = torch.empty((*init_state.shape, n_paths, n_steps + 1), dtype=init_state.dtype, device=init_state.device)
    paths[..., 0] = init_state.unsqueeze(-1).expand(*init_state.shape, n_paths)  # начальное значение v(0)

    for i in range(0, n_steps):                          # цикл по шагам времени
        dt = dt_steps[i]                                # текущий Δt
        theta_i = theta if theta_is_scalar else theta[i+1]  # параметры на шаге i→i+1 берём из узла i+1
        eps_i = eps if eps_is_scalar else eps[i+1]
        delta = 4 * kappa * theta_i / (eps_i * eps_i)   # степени свободы χ²
        exp = torch.exp(-kappa*dt)                       # коэффициент затухания e^{-κΔt}
        c_bar = 1 / (4*kappa) * eps_i * eps_i * (1 - exp)  # масштабный множитель распределения
        v_cur = paths[..., i]                           # текущее значение дисперсии
        kappa_bar = v_cur * 4*kappa*exp / (eps_i * eps_i * (1 - exp))  # параметр нецентральности
        # [Grzelak2019, definition 8.1.1] — точное распределение CIR через нецентральное χ²
        v_next = c_bar * noncentral_chisquare(delta, kappa_bar)  # следующее значение дисперсии
        if minimum_value is not None:
            v_next = torch.clamp(v_next, min=minimum_value)  # обрезка снизу (нужно для autograd)
        paths[..., i+1] = v_next                         # сохранить шаг
    return paths                                        # траектории CIR


def generate_heston(
        n_paths: int,
        timeline: torch.Tensor,
        init_price: torch.Tensor,
        init_var: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        eps: torch.Tensor,
        rho: torch.Tensor,
        drift: torch.Tensor,
        minimum_var: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates time series following the Heston model.

    Stochastic process of the Heston model is described by a system of SDE's:
        dS(t) = μ·S(t)·dt + sqrt(v(t))·S(t)·dW_1(t);
        dv(t) = κ(θ - v(t))·dt + ε·sqrt(v(t))·dW_2(t).

    Time series are generated using Andersen's Quadratic Exponential
    scheme [Andersen2007]. Also see [Grzelak, section 9.4.3].

    Args:
        n_paths: Number of simulated paths.
        timeline:: Time steps.
        init_price: Initial states of the price paths, i.e. S(0). 
        init_var: Initial states of the variance paths, i.e. v(0). 
        kappa: Parameter κ - the rate at which v(t) reverts to θ.
        theta: Parameter θ - long-run average variance.
        eps: Parameter ε - volatility of variance.
        rho: Correlation between underlying Brownian motions for S(t) and v(t).
        drift: Drift parameter μ.
        minimum_var: On each step, clamp the value of the variance process to
            the range [minimum_var, +∞) to prevent it from being too close to
            zero. This is necessary when using autograd to compute the
            derivative of generated values w.r.t. v(0).

    Returns:
        Two tensors: 1) simulated paths for price, 2) simulated paths for variance.
        Both tensors have the shape (n_paths, n_steps + 1).
    """

    dt_steps = timeline.diff()                          # шаги по времени Δt
    n_steps = dt_steps.shape[0]                          # число шагов

    init_state_price = init_price * torch.ones(n_paths)  # начальная цена для каждого пути
    # init_var нормализуем к скаляру или вектору длины n_paths, иначе generate_cir
    # вернёт лишнюю ось и индексация var_paths[:, i] сломает broadcasting.
    _iv = torch.as_tensor(init_var, dtype=init_state_price.dtype, device=init_state_price.device)
    if _iv.dim() == 0 or _iv.numel() == 1:
        init_state_var = _iv.reshape(())               # скаляр v(0)
    elif _iv.shape == (n_paths,):
        init_state_var = _iv                            # своя v(0) на каждый путь
    else:
        raise ValueError(
            f"init_var must be a scalar or tensor of shape ({n_paths},), got {tuple(_iv.shape)}"
        )

    if init_state_price.shape != torch.Size((n_paths,)):  # форма цен должна быть (n_paths,)
        raise ValueError('Shape of `init_state_price` must be (n_paths,)')

    var_paths = generate_cir(n_paths, timeline, init_state_var, kappa, theta, eps, minimum_var)  # траектории дисперсии (CIR)
    log_paths = torch.empty((n_paths, n_steps + 1), dtype=init_state_price.dtype)  # лог-цены
    log_paths[:, 0] = init_state_price.log()            # начальная лог-цена

    gamma2 = 0.5                                         # вес дисперсии на конце интервала (схема Андерсена)

    for i in range(0, n_steps):                          # цикл по шагам времени
        dt = dt_steps[i]                                # текущий Δt
        # условие регулярности [Andersen 2007, section 4.3.2] — подбор gamma2
        if rho > 0:  # при rho<=0 всегда выполняется
            L = rho*dt*(kappa/eps - 0.5*rho)
            R = 2*kappa/(eps*eps*(1 - torch.exp(-kappa*dt))) - rho/eps
            if R<=0 or L==0 or (L<0 and R>=0):
                # в этих случаях условие регулярности выполняется автоматически
                pass
            elif L > 0:
                gamma2 = min(0.5, R / L * 0.9)  # запас 10% для устойчивости
        gamma1 = 1.0 - gamma2                            # вес дисперсии в начале интервала
    
        k0 = -rho * kappa * theta * dt / eps            # константы дискретизации лог-цены:
        k1 = gamma1 * dt * (kappa * rho / eps - 0.5) - rho / eps
        k2 = gamma2 * dt * (kappa * rho / eps - 0.5) + rho / eps
        k3 = gamma1 * dt * (1 - rho * rho)
        k4 = gamma2 * dt * (1 - rho * rho)

        v_i = var_paths[..., i]                          # дисперсия в начале шага
        v_next = var_paths[..., i + 1]                   # дисперсия в конце шага
        next_vals = drift*dt + log_paths[:, i] + k0 + k1*v_i + k2*v_next + \
            torch.sqrt(k3*v_i + k4*v_next) * torch.randn_like(v_i)  # шаг лог-цены с коррелированным шумом
        log_paths[:, i+1] = next_vals                    # сохранить шаг
    return log_paths.exp(), var_paths                    # вернуть цены (exp лог-цен) и дисперсии
