"""pyquant.black_scholes: модель Блэка-Шоулза (цены, греки, implied vol).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (низкоуровневая количественная библиотека, numba).
НАЗНАЧЕНИЕ:  калькулятор Блэка-Шоулза (jitclass BSCalc): цены ванильных опционов,
             греки (delta, gamma, vega, vanna, volga), инверсия implied vol из
             премии (гибрид Ньютона/бисекции), а также страйк из дельты.
ИМПОРТИРУЕТ: numpy, numba; всё из .utils и .common (value-объекты, normal_cdf/pdf).
ЭКСПОРТИРУЕТ: BSCalc.
КЕМ ИСПОЛЬЗУЕТСЯ: services.heston_pricing.bs_implied_vol (инверсия IV),
             pyquant.vol_surface, бенчмарк Black-Scholes в калибровке.
=============================================================================
"""

import numpy as np                                       # численные массивы
import numba as nb                                       # JIT-компиляция

from .utils import *                                     # normal_cdf/pdf, сплайны
from .common import *                                    # value-объекты (Forward, Strike, …)


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64),
    ("K", nb.float64),
    ("is_call", nb.boolean),
    ("tol", nb.float64),
    ("sigma_lower", nb.float64),
    ("sigma_upper", nb.float64),
    ("grad_eps", nb.float64),
    ("delta_tol", nb.float64),
    ("strike_lower", nb.float64),
    ("strike_upper", nb.float64),
    ("delta_grad_eps", nb.float64)
])
class BSCalc:                                            # калькулятор Блэка-Шоулза с настройками решателей
    def __init__(self):
        self.tol = 10**-6                              # допуск решателя implied vol
        self.sigma_lower = 10**-3                      # нижняя граница σ
        self.sigma_upper = 3                           # верхняя граница σ
        self.grad_eps = 1e-6                           # порог градиента (vega) для Ньютона
        self.delta_tol = 10**-12                       # допуск решателя страйка из дельты
        self.strike_lower = 0.1                        # нижняя граница страйка (доля от спота)
        self.strike_upper = 10.                        # верхняя граница страйка (доля от спота)
        self.delta_grad_eps = 1e-4                     # порог градиента для решателя страйка
        
    def strike_from_delta(self, forward: Forward, delta: Delta, implied_vol: ImpliedVol) -> Strike:  # найти страйк по заданной дельте
        K_l = self.strike_lower*forward.S              # нижняя граница поиска
        K_r = self.strike_upper*forward.S              # верхняя граница поиска
        option_type = OptionType(delta.pv >= 0.)       # знак дельты → call/put
        
        def g(K):                                      # невязка: delta(K) - целевая дельта
            return self._delta(forward, Strike(K), option_type, implied_vol) - delta.pv

        def g_prime(K):                                # производная dDelta/dK
            return self._dDelta_dK(forward, Strike(K), implied_vol)
        
        if g(K_l) * g(K_r) > 0:                        # нет смены знака на интервале…
            raise ValueError('No solution within strikes interval')
        
        K = (K_l+K_r) / 2                              # старт — середина
        epsilon = g(K)                                 # текущая невязка
        grad = g_prime(K)                              # текущий градиент
        while abs(epsilon) > self.delta_tol:           # пока не сошлось:
            if abs(grad) > self.delta_grad_eps:        #   шаг Ньютона (если градиент адекватен)
                K -= epsilon / grad
                if K > K_r or K < K_l:                 #     вышли за границы → бисекция
                    K = (K_l + K_r) / 2
                    if g(K_l)*g(K) > 0:
                        K_l = K
                    else:
                        K_r = K
                    K = (K_l + K_r) / 2
            else:                                      #   иначе чистая бисекция
                if g(K_l)*epsilon > 0:
                    K_l = K
                else:
                    K_r = K
                K = (K_l + K_r) / 2

            epsilon = g(K)                             #   пересчёт невязки
            grad = g_prime(K)                          #   пересчёт градиента
            
        return Strike(K)                               # найденный страйк
        
    def implied_vol(self, forward: Forward, strike: Strike, premium: Premium) -> ImpliedVol:  # инверсия премии в implied vol
        pv = premium.pv                                # премия

        if pv < 0.:                                    # отрицательная премия недопустима
            raise ValueError('Negative PV provided to implied vol solver')
     
        if pv == 0.0:                                  # нулевая премия → ставим 1 цент
            print('WARNING: trivial PV provided, setting PV = 1 cent')
            pv = 0.01

        fv = forward.forward_rate().fv                 # форвардная цена

        def g(sigma):                                  # невязка: рыночная − модельная премия
            return pv - self._premium(forward, strike, OptionType(strike.K >= fv), ImpliedVol(sigma))

        def g_prime(sigma):                            # производная по σ (−vega)
            return -self._vega(forward, strike, ImpliedVol(sigma))
        
        sigma_l = self.sigma_lower                     # нижняя граница σ
        sigma_r = self.sigma_upper                     # верхняя граница σ
       
        if g(sigma_l) * g(sigma_r) > 0:                # нет смены знака → нет решения
            raise ValueError('No solution within implied vol interval')
        
        sigma = (sigma_l + sigma_r) / 2                # старт — середина
        epsilon = g(sigma)                             # текущая невязка
        grad = g_prime(sigma)                          # текущий градиент
        while abs(epsilon) > self.tol:                 # пока не сошлось:
            if abs(grad) > self.grad_eps:              #   шаг Ньютона (если vega адекватна)
                sigma -= epsilon / grad
                if sigma > sigma_r or sigma < sigma_l:  #     вышли за границы → бисекция
                    sigma = (sigma_l + sigma_r) / 2
                    if g(sigma_l)*g(sigma) > 0:
                        sigma_l = sigma
                    else:
                        sigma_r = sigma
                    sigma = (sigma_l + sigma_r) / 2
            else:                                      #   иначе чистая бисекция
                if g(sigma_l)*epsilon > 0:
                    sigma_l = sigma
                else:
                    sigma_r = sigma
                sigma = (sigma_l + sigma_r) / 2
            
            epsilon = g(sigma)                         #   пересчёт невязки
            grad = g_prime(sigma)                      #   пересчёт градиента

        return ImpliedVol(sigma)                       # найденная implied vol
    
    def implied_vols(self, forward: Forward, strikes: Strikes, premiums: Premiums) -> ImpliedVols:  # векторная инверсия IV
        if not strikes.data.shape == premiums.data.shape:  # размерности совпадают
            raise ValueError('Inconsistent data between strikes and premiums')
        n = len(strikes.data)                          # число точек
        fv = forward.forward_rate().fv                 # форвардная цена
        ivols = np.zeros(n, dtype=np.float64)          # результат
        for index in range(n):                         # по каждому страйку…
            K = strikes.data[index]
            PV = premiums.data[index]
            ivols[index] = self.implied_vol(           #   инверсия отдельной премии
                forward,
                Strike(K), 
                Premium(PV)).sigma
        return ImpliedVols(ivols)
    
    def _premium(self, forward: Forward, strike: Strike, option_type: OptionType, implied_vol: ImpliedVol) -> nb.float64:  # цена БШ (скаляр)
        pm = 1 if option_type.is_call else -1          # знак: +1 call / -1 put
        d1 = self._d1(forward, strike, implied_vol)    # d1
        d2 = self._d2(d1, forward, implied_vol)        # d2
        return pm * forward.S * normal_cdf(pm * d1) - pm * strike.K * \
            np.exp(-forward.r * forward.T) * normal_cdf(pm * d2)  # формула Блэка-Шоулза
    
    def premium(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Premium:  # премия с номиналом
        assert forward.T == vanilla.T                  # сроки согласованы
        return Premium(vanilla.N * self._premium(forward, vanilla.strike(), vanilla.option_type(), implied_vol))
    
    def premiums(self, forward: Forward, vanillas: SingleMaturityVanillas, implied_vols: ImpliedVols) -> Premiums:  # векторные премии
        assert forward.T == vanillas.T                 # сроки согласованы
        ivs = implied_vols.data                        # волатильности
        Ks = vanillas.Ks                               # страйки
        assert ivs.shape == Ks.shape
        res_premiums = np.zeros_like(ivs)              # результат
        is_calls = vanillas.is_call                    # типы
        for i in range(len(ivs)):                       # по каждой ванили…
            res_premiums[i] = self._premium(forward, Strike(Ks[i]), OptionType(is_calls[i]), ImpliedVol(ivs[i]))
        return Premiums(vanillas.Ns * res_premiums)    # × номиналы
    
    def _delta(self, forward: Forward, strike: Strike, option_type: OptionType, implied_vol: ImpliedVol) -> nb.float64:  # дельта (скаляр)
        d1 = self._d1(forward, strike, implied_vol)
        call_delta = forward.discount_ratio().D * normal_cdf(d1)  # дельта колла
        return call_delta if option_type.is_call else call_delta - 1.0  # пут = колл − 1
        
    def delta(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Delta:  # дельта с номиналом
        assert forward.T == vanilla.T
        return Delta(vanilla.N*\
            self._delta(forward, vanilla.strike(), vanilla.option_type(), implied_vol)
        )

    def deltas(self, forward: Forward, vanillas: SingleMaturityVanillas, implied_vols: ImpliedVols) -> Deltas:  # векторные дельты
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_deltas = np.zeros_like(ivs)
        is_call = vanillas.is_call
        for i in range(len(ivs)):                       # по каждой ванили…
            res_deltas[i] = self._delta(forward, Strike(Ks[i]), OptionType(is_call[i]), ImpliedVol(ivs[i]))
        return Deltas(vanillas.Ns * res_deltas)
    
    def _gamma(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:  # гамма (скаляр)
        d1 = self._d1(forward, strike, implied_vol) 
        return forward.discount_ratio().D * normal_pdf(d1) / (forward.S * implied_vol.sigma * np.sqrt(forward.T))
    
    def gamma(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Gamma:  # гамма с номиналом
        assert forward.T == vanilla.T
        return Gamma(vanilla.N*\
            self._gamma(forward, vanilla.strike(), implied_vol)
        )

    def gammas(self, forward: Forward, vanillas: SingleMaturityVanillas, implied_vols: ImpliedVols) -> Gammas:  # векторные гаммы
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_gammas = np.zeros_like(ivs)
        for i in range(len(ivs)):                       # по каждой ванили…
            res_gammas[i] = self._gamma(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Gammas(vanillas.Ns * res_gammas)
    
    def _vega(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:  # вега (скаляр)
        return forward.discount_ratio().D * forward.S * np.sqrt(forward.T) * normal_pdf(self._d1(forward, strike, implied_vol))
    
    
    def vega(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Vega:  # вега с номиналом
        assert forward.T == vanilla.T
        return Vega(vanilla.N*\
            self._vega(forward, vanilla.strike(), implied_vol)
        )
    
    def vegas(self, forward: Forward, vanillas: SingleMaturityVanillas, implied_vols: ImpliedVols) -> Vegas:  # векторные веги
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_vegas = np.zeros_like(ivs)
        for i in range(len(ivs)):                       # по каждой ванили…
            res_vegas[i] = self._vega(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Vegas(vanillas.Ns * res_vegas)
    
    
    def _vanna(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:  # ванна (скаляр)
        d2 = self._d2(self._d1(forward, strike, implied_vol), forward, implied_vol)
        return self._vega(forward, strike, implied_vol) * d2 / (implied_vol.sigma * forward.S)
    
    def vanna(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Vanna:  # ванна с номиналом
        assert forward.T == vanilla.T
        return Vanna(vanilla.N*\
            self._vanna(forward, vanilla.strike(), implied_vol)
        )
    
    def vannas(self, forward: Forward, vanillas: SingleMaturityVanillas, implied_vols: ImpliedVols) -> Vannas:  # векторные ванны
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_vannas = np.zeros_like(ivs)
        for i in range(len(ivs)):                       # по каждой ванили…
            res_vannas[i] = self._vanna(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Vannas(vanillas.Ns * res_vannas)
    
    def _volga(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:  # волга (скаляр)
        d1 = self._d1(forward, strike, implied_vol)
        d2 = self._d2(d1, forward, implied_vol)
        return self._vega(forward, strike, implied_vol) * d1 * d2 / implied_vol.sigma
        
    
    def volga(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Volga:  # волга с номиналом
        assert forward.T == vanilla.T
        return Volga(vanilla.N*\
            self._volga(forward, vanilla.strike(), implied_vol)
        )
    
    def volgas(self, forward: Forward, vanillas: SingleMaturityVanillas, implied_vols: ImpliedVols) -> Volgas:  # векторные волги
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_volgas = np.zeros_like(ivs)
        for i in range(len(ivs)):                       # по каждой ванили…
            res_volgas[i] = self._volga(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Volgas(vanillas.Ns * res_volgas)
    
    def _d1(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:  # параметр d1 формулы БШ
        d1 = (np.log(forward.S / strike.K) + (forward.r + implied_vol.sigma**2 / 2) * forward.T) / (implied_vol.sigma * np.sqrt(forward.T))
        return d1
    
    def _d2(self, d1: nb.float64, forward: Forward, implied_vol: ImpliedVol) -> nb.float64:  # параметр d2 = d1 − σ√T
        return d1 - implied_vol.sigma * np.sqrt(forward.T)
    
    def _dDelta_dK(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:  # производная дельты по страйку
        d1 = self._d1(forward, strike, implied_vol)
        DoF = forward.discount_ratio().D               # отношение дисконтов
        return - DoF * normal_pdf(d1) / (strike.K * np.sqrt(forward.T) * implied_vol.sigma)
    