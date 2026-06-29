"""pyquant.common: типобезопасные value-объекты и кривые ставок/форвардов.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (низкоуровневая количественная библиотека, numba).
НАЗНАЧЕНИЕ:  определяет numba-jitclass «обёртки» над скалярными/векторными
             величинами (Spot, Strike, Premium, Greeks и т.д.) — это даёт
             типобезопасность и читаемость в численном ядре. Также содержит
             кривые доходностей/дисконтирования (сплайны) и форвардную кривую,
             используемые при ценообразовании и калибровке.
ИМПОРТИРУЕТ: numpy, numba; всё из .utils (сплайны, searchsorted, is_sorted).
ЭКСПОРТИРУЕТ (ключевое): Spot, Strike, Strikes, Premium, Premiums, Notional,
             TimeToMaturity, TimesToMaturity, Forward, ForwardRate(s),
             OptionType(s), Vanilla, StrikesMaturitiesGrid, греки (Delta, Gamma,
             Vega, Vanna, Volga и пр.), ForwardYieldCurve, DiscountCurve,
             ForwardCurve, forward_curve_from_forward_rates.
ПАТТЕРН:     большинство классов — тривиальные обёртки «хранит одно поле»; это
             нужно numba для строгой типизации сигнатур движков pyquant.
КЕМ ИСПОЛЬЗУЕТСЯ: pyquant.black_scholes, pyquant.heston, pyquant.vol_surface;
             через services.heston_pricing — весь остальной проект.
=============================================================================
"""

import numpy as np                                       # численные массивы
import numba as nb                                       # JIT-компиляция (jitclass/njit)

from .utils import *                                     # сплайны, searchsorted, is_sorted, константы

@nb.experimental.jitclass([
    ("v", nb.float64)
])
class CalibrationError:                                  # обёртка: ошибка калибровки (скаляр)
    def __init__(self, value: nb.float64):
        self.v = value


@nb.experimental.jitclass([
    ("w", nb.float64[:])
])
class CalibrationWeights:                               # веса наблюдений при калибровке (вектор ≥ 0)
    def __init__(self, w: nb.float64):
        if not np.all(w>=0):                            # все веса неотрицательны
            raise ValueError('Weights must be non-negative')
        if not w.sum() > 0:                            # хотя бы один вес ненулевой
            raise ValueError('At least one weight must be non-trivial')
        self.w = w


@nb.experimental.jitclass([
    ("v", nb.boolean)
])
class StickyStrike:                                     # флаг режима «sticky strike» для поверхности волатильности
    def __init__(self, v: nb.boolean = False):
        self.v = v  


@nb.experimental.jitclass([
    ("sigma", nb.float64)
])
class ImpliedVol:                                       # подразумеваемая волатильность (скаляр > 0)
    def __init__(self, sigma: nb.float64):
        if not sigma > 0:                              # IV должна быть положительной
            raise ValueError('Non-positive implied vol')
        self.sigma = sigma
                
@nb.experimental.jitclass([
    ("sigma", nb.float64)
])
class VolatilityQuote:                                  # рыночная котировка волатильности (скаляр)
    def __init__(self, sigma: nb.float64):
        self.sigma = sigma


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class ImpliedVols:                                      # вектор подразумеваемых волатильностей (> 0)
    def __init__(self, sigmas: nb.float64[:]):
        if not np.all(sigmas > 0.):                    # все IV положительны
            raise ValueError('Not all implied vols are positive')
        self.data = sigmas
   

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class VolatilityQuotes:                                 # вектор рыночных котировок волатильности
    def __init__(self, sigmas: nb.float64[:]):
        self.data = sigmas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Premium:                                          # премия (цена) опциона, скаляр
    def __init__(self, pv: nb.float64):
        self.pv = pv

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Premiums:                                         # вектор премий опционов
    def __init__(self, pvs: nb.float64[:]):
        self.data = pvs

@nb.experimental.jitclass([
    ("N", nb.float64)
])
class Notional:                                         # номинал (объём) позиции, скаляр
    def __init__(self, N: nb.float64):
        self.N = N

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Notionals:                                        # вектор номиналов
    def __init__(self, notionals: nb.float64[:]):
        self.data = notionals
        

@nb.experimental.jitclass([
    ("S", nb.float64)
])
class Spot:                                             # спот-цена базового актива, скаляр
    def __init__(self, spot: nb.float64):
        self.S = spot


@nb.experimental.jitclass([
    ("r_d", nb.float64)
])
class DiscountYield:                                    # ставка дисконтирования, скаляр
    def __init__(self, rate: nb.float64):
        self.r_d = rate


@nb.experimental.jitclass([
    ("D", nb.float64)
])
class DiscountFactor:                                   # фактор дисконтирования exp(-rT), скаляр
    def __init__(self, D: nb.float64):
        self.D = D


@nb.experimental.jitclass([
    ("r", nb.float64)
])
class ForwardYield:                                     # форвардная доходность (carry), скаляр
    def __init__(self, rate: nb.float64):
        self.r = rate


@nb.experimental.jitclass([
    ("D", nb.float64)
])
class ForwardDiscount:                                  # форвардный дисконт exp(-rT), скаляр
    def __init__(self, d: nb.float64):
        self.D = d


@nb.experimental.jitclass([
    ("D", nb.float64)
])
class DiscountRatio:                                    # отношение дисконт/форвард-дисконт, скаляр
    def __init__(self, d: nb.float64):
        self.D = d

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class DiscountYields:                                   # вектор ставок дисконтирования
    def __init__(self, rates: nb.float64[:]):
        self.data = rates


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class ForwardYields:                                    # вектор форвардных доходностей
    def __init__(self, rates: nb.float64[:]):
        self.data = rates


@nb.experimental.jitclass([
    ("T", nb.float64)
])
class TimeToMaturity:                                   # срок до экспирации (годы), скаляр > 0
    def __init__(self, time_to_maturity: nb.float64):
        if not time_to_maturity > 0:                   # срок должен быть положительным
            raise ValueError('Non-positive time to maturity')
        self.T = time_to_maturity


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class TimesToMaturity:                                  # вектор сроков до экспирации (отсортированный)
    def __init__(self, T: nb.float64[:]):
        if not np.all(T >= 0) and is_sorted(T):        # сроки неотрицательны и отсортированы
            raise ValueError('Not all times to maturity are positive and sorted')
        self.data = T      


@nb.experimental.jitclass([
    ("fv", nb.float64)
])
class ForwardRate:                                      # форвардная цена актива, скаляр
    def __init__(self, forward: nb.float64):
        self.fv = forward


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class ForwardRates:                                     # вектор форвардных цен
    def __init__(self, forwards: nb.float64[:]):
        self.data = forwards


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class DiscountFactors:                                  # вектор факторов дисконтирования
    def __init__(self, data: nb.float64[:]):
        self.data = data


@nb.experimental.jitclass([
    ("r_d", nb.float64),
    ("T", nb.float64)
])
class Discount:                                         # пара (ставка, срок) с производным дисконтом
    def __init__(self, rate: DiscountYield, time_to_maturity: TimesToMaturity):
        self.r_d = rate.r_d                            # ставка дисконтирования
        self.T = time_to_maturity.T                    # срок

    def discount_yield(self) -> DiscountYield:
        return DiscountYield(self.r_d)                 # вернуть ставку как объект

    def time_to_maturity(self) -> TimeToMaturity:
        return TimeToMaturity(self.T)                  # вернуть срок как объект

    def discount_factor(self) -> DiscountFactor:
        return DiscountFactor(np.exp(-self.r_d * self.T))  # фактор дисконтирования exp(-rT)


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64),
    ("r_d", nb.float64),
    ("T", nb.float64)
])
class Forward:                                          # форвард: спот + ставки + срок, с производными величинами
    def __init__(self, spot: Spot, forward_yield: ForwardYield, discount_yield: DiscountYield, time_to_maturity: TimeToMaturity):
        self.S = spot.S                                # спот
        self.r = forward_yield.r                       # форвардная доходность
        self.r_d = discount_yield.r_d                  # ставка дисконтирования
        self.T = time_to_maturity.T                    # срок

    def spot(self) -> Spot:
        return Spot(self.S)                            # спот как объект

    def forward_yield(self) -> ForwardYield:
        return ForwardYield(self.r)                    # форвардная доходность как объект

    def discount_yield(self) -> DiscountYield:
        return DiscountYield(self.r_d)                 # ставка дисконтирования как объект

    def discount_factor(self) -> DiscountFactor:
        return DiscountFactor(np.exp(-self.r_d * self.T))  # фактор дисконтирования

    def time_to_maturity(self) -> TimeToMaturity:
        return TimeToMaturity(self.T)                  # срок как объект

    def forward_rate(self) -> ForwardRate:
        return ForwardRate(self.S * np.exp(self.r * self.T))  # форвардная цена = S·exp(rT)

    def forward_discount(self) -> ForwardDiscount:
        return ForwardDiscount(np.exp(-self.r * self.T))  # форвардный дисконт

    def discount_ratio(self) -> DiscountRatio:
        return DiscountRatio(self.discount_factor().D / self.forward_discount().D)  # отношение дисконтов


@nb.njit
def forward_from_forward_rate(                          # построить Forward из спота и форвардной цены
    spot: Spot,
    forward_rate: ForwardRate,
    time_to_maturity: TimeToMaturity
) -> Forward:
    # assumes discount and forward yield are the same   # предполагаем равенство ставок
    r = - np.log(spot.S / forward_rate.fv)/ time_to_maturity.T  # неявная ставка из F=S·exp(rT)
    return Forward(
        spot, ForwardYield(r), DiscountYield(r), time_to_maturity
        )



@nb.experimental.jitclass()
class ForwardYieldCurve:                                # кривая форвардных доходностей (сплайн по rT)
    _spline: PchipSpline1D

    def __init__(self, forward_yields: ForwardYields, times_to_maturity: TimesToMaturity):
        if not forward_yields.data.shape == times_to_maturity.data.shape:  # размерности совпадают
            raise ValueError('Inconsistent data between yields and times to maturity')
        if not is_sorted(times_to_maturity.data) and np.all(times_to_maturity.data > 0):  # сроки валидны
            raise ValueError('Times to maturity are invalid')

        self._spline = PchipSpline1D(                  # сплайн интегральной доходности r·T (через 0)
           XAxis(np.append(np.array([0.]), times_to_maturity.data)),
           YAxis(np.append(np.array([0.]), times_to_maturity.data*forward_yields.data)) 
        )

    def forward_yield(self, time_to_maturity: TimeToMaturity) -> ForwardYield:
        assert time_to_maturity.T > 0.                 # срок положителен
        return ForwardYield(self._spline.apply(time_to_maturity.T) / time_to_maturity.T)  # r = (rT)/T

    def forward_yields(self, times_to_maturity: TimeToMaturity) -> ForwardYields:
        Ts = times_to_maturity.data                    # сроки
        res = np.zeros_like(Ts)                        # результат
        for i in range(len(Ts)):                       # по каждому сроку…
            assert Ts[i] > 0.
            res[i] = self._spline.apply(Ts[i]) / Ts[i]  #   доходность из сплайна
        return ForwardYields(res)
    

@nb.experimental.jitclass()
class DiscountCurve:                                    # кривая дисконтирования (сплайн по r_d·T)
    _spline: PchipSpline1D

    def __init__(self, discount_yields: DiscountYields, times_to_maturity: TimesToMaturity):
        if not discount_yields.data.shape == times_to_maturity.data.shape:  # размерности совпадают
            raise ValueError('Inconsistent data between discount yields and times to maturity')
        if not is_sorted(times_to_maturity.data) and np.all(times_to_maturity.data > 0):  # сроки валидны
            raise ValueError('Times to maturity are invalid')

        self._spline = PchipSpline1D(                  # сплайн интегральной ставки r_d·T (через 0)
           XAxis(np.append(np.array([0.]), times_to_maturity.data)),
           YAxis(np.append(np.array([0.]), times_to_maturity.data*discount_yields.data)) 
        )

    def discount_yield(self, time_to_maturity: TimeToMaturity) -> DiscountYield:
        assert time_to_maturity.T > 0.                 # срок положителен
        return DiscountYield(self._spline.apply(time_to_maturity.T) / time_to_maturity.T)  # r_d из сплайна

    def discount_factor(self, time_to_maturity: TimeToMaturity) -> DiscountFactor:
        return DiscountFactor(np.exp(-self.discount_yield(time_to_maturity).r_d * time_to_maturity.T))  # exp(-r_d·T)

    def discount_yields(self, times_to_maturity: TimeToMaturity) -> DiscountYields:
        Ts = times_to_maturity.data                    # сроки
        res = np.zeros_like(Ts)                        # результат
        for i in range(len(Ts)):                       # по каждому сроку…
            assert Ts[i] > 0.
            res[i] = self._spline.apply(Ts[i]) / Ts[i]  #   ставка из сплайна
        return DiscountYields(res)

    def discount_factors(self, times_to_maturity: TimeToMaturity) -> DiscountFactors:
        return DiscountFactors(                        # векторные факторы дисконтирования
            np.exp(-self.discount_yields(times_to_maturity).data * times_to_maturity.data)
        )


@nb.experimental.jitclass([
    ("S", nb.float64)
])
class ForwardCurve:                                     # форвардная кривая: спот + кривая доходностей + кривая дисконта
    _curve: ForwardYieldCurve
    _curve_d: DiscountCurve

    def __init__(self, spot: Spot, forward_yield_curve: ForwardYieldCurve, discount_curve: DiscountCurve):
        self.S = spot.S                                # спот
        self._curve = forward_yield_curve              # кривая форвардных доходностей
        self._curve_d = discount_curve                 # кривая дисконтирования

    def spot(self) -> Spot:
        return Spot(self.S)                            # спот как объект

    def forward(self, time_to_maturity: TimeToMaturity) -> Forward:
        return Forward(                                # форвард на заданный срок
            Spot(self.S), 
            self._curve.forward_yield(time_to_maturity), 
            self._curve_d.discount_yield(time_to_maturity),
            time_to_maturity
        )

    def forward_rates(self, times_to_maturity: TimesToMaturity) -> ForwardRates:
        return ForwardRates(                           # форвардные цены = S·exp(r·T)
            self.S * np.exp(times_to_maturity.data * self._curve.forward_yields(times_to_maturity).data)
        )

    def forward_yields(self, times_to_maturity: TimesToMaturity) -> ForwardYields:
        return self._curve.forward_yields(times_to_maturity)  # форвардные доходности по срокам

    def discount_yields(self, times_to_maturity: TimesToMaturity) -> DiscountYields:
        return self._curve_d.discount_yields(times_to_maturity)  # ставки дисконтирования по срокам

    def discount_factors(self, times_to_maturity: TimesToMaturity) -> DiscountFactors:
        return self._curve_d.discount_factors(times_to_maturity)  # факторы дисконтирования по срокам


@nb.njit
def forward_curve_from_forward_rates(                   # построить ForwardCurve из спота и форвардных цен
    spot: Spot,
    forward_rates: ForwardRates,
    times_to_maturity: TimesToMaturity
) -> ForwardCurve:
    rs = - np.log(spot.S / forward_rates.data) / times_to_maturity.data  # неявные ставки из F=S·exp(rT)
    return ForwardCurve(
        spot,
        ForwardYieldCurve(                             # кривая форвардных доходностей
            ForwardYields(rs),
            times_to_maturity), 
        DiscountCurve(                                 # кривая дисконтирования (та же ставка)
            DiscountYields(rs),
            times_to_maturity)
    )   
    



@nb.experimental.jitclass([
    ("K", nb.float64)  
])
class Strike:                                           # страйк опциона, скаляр
    def __init__(self, strike: nb.float64):
        self.K = strike

        
@nb.experimental.jitclass([
    ("data",  nb.float64[:])  
])
class Strikes:                                          # вектор страйков
    def __init__(self, strikes:  nb.float64[:]):
        self.data = strikes
            

@nb.experimental.jitclass([
    ("is_call", nb.boolean)
])
class OptionType:                                       # тип опциона: call (True) / put (False)
    def __init__(self, is_call: nb.boolean):
        self.is_call = is_call


@nb.njit()
def call_option(): 
    return OptionType(True)                             # удобный конструктор колла


@nb.njit()
def put_option(): 
    return OptionType(False)                            # удобный конструктор пута
        
        
@nb.experimental.jitclass([
    ("data", nb.boolean[:])
])
class OptionTypes:                                      # вектор типов опционов (call/put)
    def __init__(self, is_call: nb.boolean[:]):
        self.data = is_call
      
    
@nb.experimental.jitclass([
    ("is_call", nb.boolean),
    ("K", nb.float64),
    ("N", nb.float64),
    ("T", nb.float64)
])
class Vanilla:                                          # ванильный опцион: тип + страйк + номинал + срок
    def __init__(self, option_type: OptionType, strike: Strike, notional: Notional, time_to_maturity: TimeToMaturity):
        self.is_call = option_type.is_call             # call/put
        self.K = strike.K                              # страйк
        self.N = notional.N                            # номинал
        self.T = time_to_maturity.T                    # срок

    def option_type(self) -> OptionType:
        return OptionType(self.is_call)                # тип как объект

    def strike(self) -> Strike:
        return Strike(self.K)                          # страйк как объект

    def time_to_maturity(self) -> TimeToMaturity:
        return TimeToMaturity(self.T)                  # срок как объект
        
        
@nb.experimental.jitclass([
    ("is_call", nb.boolean[:]),
    ("Ks", nb.float64[:]),
    ("Ns", nb.float64[:]),
    ("T", nb.float64)
])
class SingleMaturityVanillas:                           # набор ванил одного срока (векторные страйки/типы/номиналы)
    def __init__(self, option_types: OptionTypes, strikes: Strikes, notionals: Notionals, time_to_maturity: TimeToMaturity):
        if not option_types.data.shape == strikes.data.shape:  # типы и страйки согласованы
            raise ValueError('Inconsistent data between strikes and option types')
        if not notionals.data.shape == strikes.data.shape:  # номиналы и страйки согласованы
            raise ValueError('Inconsistent data between strikes and notionals')
        self.is_call = option_types.data               # типы
        self.Ks = strikes.data                         # страйки
        self.Ns = notionals.data                       # номиналы
        self.T = time_to_maturity.T                    # общий срок

    def strikes(self) -> Strikes:
        return Strikes(self.Ks)                        # страйки как объект

    def time_to_maturity(self):
        return TimeToMaturity(self.T)                  # срок как объект
        

@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Delta:                                            # грек: дельта (∂P/∂S), скаляр
    def __init__(self, delta: nb.float64):
        self.pv = delta

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Deltas:                                            # вектор дельт
    def __init__(self, deltas: nb.float64[:]):
        self.data = deltas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Gamma:                                            # грек: гамма (∂²P/∂S²), скаляр
    def __init__(self, gamma: nb.float64):
        self.pv = gamma

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Gammas:                                            # вектор гамм
    def __init__(self, gammas: nb.float64[:]):
        self.data = gammas

        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Vega:                                             # грек: вега (∂P/∂σ), скаляр
    def __init__(self, vega: nb.float64):
        self.pv = vega

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Vegas:                                             # вектор вег
    def __init__(self, vegas: nb.float64):
        self.data = vegas

@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Rega:                                             # грек: rega (чувствительность к ставке/параметру), скаляр
    def __init__(self, rega: nb.float64):
        self.pv = rega

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Regas:                                             # вектор rega
    def __init__(self, regas: nb.float64):
        self.data = regas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Sega:                                             # грек: sega (чувствительность 2-го порядка), скаляр
    def __init__(self, sega: nb.float64):
        self.pv = sega

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Segas:                                             # вектор sega
    def __init__(self, segas: nb.float64):
        self.data = segas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Vanna:                                            # грек: ванна (∂²P/∂S∂σ), скаляр
    def __init__(self, vanna: nb.float64):
        self.pv = vanna

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Vannas:                                            # вектор ванн
    def __init__(self, vannas: nb.float64[:]):
        self.data = vannas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Volga:                                            # грек: волга (∂²P/∂σ²), скаляр
    def __init__(self, volga: nb.float64):
        self.pv = volga

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Volgas:                                            # вектор волг
    def __init__(self, volgas: nb.float64[:]):
        self.data = volgas

@nb.experimental.jitclass([
    ("S", nb.float64),
    ("Ts", nb.float64[:]),
    ("Ks", nb.float64[:])
])
class StrikesMaturitiesGrid:                            # сетка точек (страйк, срок) при общем споте
    def __init__(
        self,
        spot: Spot,
        times_to_maturity: TimesToMaturity,
        strikes: Strikes):

        assert times_to_maturity.data.shape == strikes.data.shape  # длины страйков и сроков совпадают
        self.S = spot.S                                # спот
        self.Ts = times_to_maturity.data               # сроки
        self.Ks = strikes.data                         # страйки

    def spot(self) -> Spot:
        return Spot(self.S)                            # спот как объект

   