"""pyquant.vol_surface: представления улыбки/поверхности волатильности.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (низкоуровневая количественная библиотека, numba).
НАЗНАЧЕНИЕ:  структуры для работы с улыбкой и поверхностью волатильности в двух
             пространствах: дельта-котировки (ATM/RR/BF на 25Δ и 10Δ) и
             страйк-пространство (страйки + implied vol). Конвертация между ними,
             построение по срокам через PCHIP-сплайны и фильтрация OTM-котировок.
ИМПОРТИРУЕТ: numpy, numba; всё из .common, .black_scholes, .utils.
ЭКСПОРТИРУЕТ: Straddle, RiskReversal, Butterfly, VolSmileChainSpace,
             VolSmileDeltaSpace, Straddles, RiskReversals, Butterflies,
             VolSurfaceDeltaSpace, VolSurfaceChainSpace.
КЕМ ИСПОЛЬЗУЕТСЯ: pyquant.heston (VolSurfaceChainSpace при калибровке);
             services.heston_pricing.calibrate_iv_surface.
=============================================================================
"""

import numpy as np                                       # численные массивы
import numba as nb                                       # JIT-компиляция
from typing import Tuple                                 # аннотации

from .common import *                                    # value-объекты, форвардные кривые
from .black_scholes import *                             # BSCalc (инверсия IV, страйк из дельты)
from .utils import *                                     # PchipSpline1D, is_sorted, оси


@nb.experimental.jitclass([
    ("sigma", nb.float64),
    ("T", nb.float64)
])
class Straddle:                                          # ATM-стрэддл: одна IV на срок (центр улыбки)
    def __init__(self, implied_vol: ImpliedVol, time_to_maturity: TimeToMaturity):
        self.sigma = implied_vol.sigma                 # ATM-волатильность
        self.T = time_to_maturity.T                    # срок

@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64),
    ("T", nb.float64)
    
])
class RiskReversal:                                     # риск-реверсал: наклон улыбки (call_IV - put_IV) на дельте
    def __init__(self, delta: Delta, vol_quote: VolatilityQuote, time_to_maturity: TimeToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):       # дельта в [0,1]
            raise ValueError('Delta expected within [0,1]')
        self.delta = delta.pv                          # уровень дельты (0.25 / 0.1)
        self.sigma = vol_quote.sigma                   # котировка RR
        self.T = time_to_maturity.T                    # срок
        
@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64),
    ("T", nb.float64)
])
class Butterfly:                                        # бабочка: кривизна улыбки (выпуклость) на дельте
    def __init__(self, delta: Delta, vol_quote: VolatilityQuote, time_to_maturity: TimeToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):       # дельта в [0,1]
            raise ValueError('Delta expected within [0,1]')
        self.delta = delta.pv                          # уровень дельты (0.25 / 0.1)
        self.sigma = vol_quote.sigma                   # котировка BF
        self.T = time_to_maturity.T                    # срок


@nb.experimental.jitclass([
    ("T", nb.float64),
    ("S", nb.float64),
    ("r", nb.float64),  
    ("r_d", nb.float64),
    ("f", nb.float64),
    ("sigmas", nb.float64[:]),
    ("Ks", nb.float64[:]),
])
class VolSmileChainSpace:                               # улыбка в страйк-пространстве (один срок: страйки + IV)
    def __init__(self, forward: Forward, strikes: Strikes, implied_vols: ImpliedVols):
        if not strikes.data.shape == implied_vols.data.shape:  # формы согласованы
            raise ValueError('Inconsistent data between strikes and implied vols')
        if not is_sorted(strikes.data):                # страйки должны быть упорядочены
            print(strikes.data)
            raise ValueError(f'Strikes are not in order')


        self.T = forward.T                             # срок
        self.S = forward.S                             # спот
        self.r = forward.r                             # форвардная ставка
        self.r_d = forward.r_d                         # ставка дисконтирования
        self.f = forward.forward_rate().fv             # форвард-цена

        self.sigmas = implied_vols.data                # вектор implied vol
        self.Ks = strikes.data                         # вектор страйков
        
    def strikes(self) -> Strikes:                       # страйки как value-объект
        return Strikes(self.Ks)
    
    def implied_vols(self) -> ImpliedVols:              # IV как value-объект
        return ImpliedVols(self.sigmas)

    def time_to_maturity(self) -> TimeToMaturity:       # срок как value-объект
        return TimeToMaturity(self.T)

    def vanillas(self) -> SingleMaturityVanillas:        # набор ванилл (call если K>=f, иначе put)
        return SingleMaturityVanillas(
            OptionTypes(self.Ks >= self.f),
            self.strikes(),
            Notionals(np.ones_like(self.Ks)),
            self.time_to_maturity())
    
    def forward(self) -> Forward:                       # восстановить форвард из полей
        return Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T))


@nb.experimental.jitclass([
    ("T", nb.float64),
    ("S", nb.float64),
    ("r", nb.float64),
    ("r_d", nb.float64),
    ("f", nb.float64),
    ("ATM", nb.float64),
    ("RR25", nb.float64),
    ("BF25", nb.float64),
    ("RR10", nb.float64),
    ("BF10", nb.float64),
    ("atm_blip", nb.float64),
    ("rr25_blip", nb.float64),
    ("bf25_blip", nb.float64),
    ("rr10_blip", nb.float64),
    ("bf10_blip", nb.float64),
    ("strike_lower", nb.float64),
    ("strike_upper", nb.float64),
    ("delta_tol", nb.float64),
    ("delta_grad_eps", nb.float64)
]) 
class VolSmileDeltaSpace:                               # улыбка в дельта-пространстве (ATM/RR/BF на 25Δ и 10Δ)
    bs_calc: BSCalc                                     # калькулятор Блэка-Шоулза (страйк из дельты)
    def __init__(
        self, 
        forward: Forward,
        ATM: Straddle,                                 # ATM-стрэддл
        RR25: RiskReversal,                            # риск-реверсал 25Δ
        BF25: Butterfly,                               # бабочка 25Δ
        RR10: RiskReversal,                            # риск-реверсал 10Δ
        BF10: Butterfly                                # бабочка 10Δ
    ):
        self.T = forward.T                             # срок
        self.S = forward.S                             # спот
        self.r = forward.r                             # форвардная ставка
        self.r_d = forward.r_d                         # ставка дисконтирования
        self.f = forward.forward_rate().fv             # форвард-цена

        if not ATM.T == self.T:                        # сроки всех котировок должны совпадать
            raise ValueError('Inconsistent time_to_maturity for ATM')
        self.ATM = ATM.sigma                           # ATM-волатильность

        if not RR25.delta == 0.25:
            raise ValueError('Inconsistent delta for 25RR')
        if not RR25.T == self.T:
            raise ValueError('Inconsistent time_to_maturity for 25RR')
        self.RR25 = RR25.sigma 

        if not BF25.delta == 0.25:
            raise ValueError('Inconsistent delta for 25BF')
        if not BF25.T == self.T:
            raise ValueError('Inconsistent time_to_maturity for 25BF')
        self.BF25 = BF25.sigma

        if not RR10.delta == 0.1:
            raise ValueError('Inconsistent delta for 10RR')
        if not RR10.T == self.T:
            raise ValueError('Inconsistent delta for 10RR')
        self.RR10 = RR10.sigma

        if not BF10.delta == 0.1:
            raise ValueError('Inconsistent delta for 10BF')
        if not BF10.T == self.T:
            raise ValueError('Inconsistent time to maturity for 10BF')
        self.BF10 = BF10.sigma

        self.atm_blip = 0.0025                         # шаг «блипа» ATM (для расчёта вега-чувствительностей)
        self.rr25_blip = 0.001                         # шаг блипа RR25
        self.bf25_blip = 0.001                         # шаг блипа BF25
        self.rr10_blip = 0.0016                        # шаг блипа RR10
        self.bf10_blip = 0.00256                       # шаг блипа BF10
        
        self.strike_lower = 0.1                        # нижняя граница поиска страйка (доля форварда)
        self.strike_upper = 10.                        # верхняя граница поиска страйка
        self.delta_tol = 10**-12                       # допуск сходимости по дельте
        self.delta_grad_eps = 1e-4                     # шаг конечной разности для градиента дельты
        
        self.bs_calc = BSCalc()                        # инициализируем BS-калькулятор и переносим в него настройки:
        self.bs_calc.strike_lower = self.strike_lower
        self.bs_calc.strike_upper = self.strike_upper
        self.bs_calc.delta_tol = self.delta_tol
        self.bs_calc.delta_grad_eps = self.delta_grad_eps

    def _implied_vols(self, RR: nb.float64, BB: nb.float64) -> Tuple[nb.float64]:
        # из RR/BF восстановить put-IV и call-IV: IV = ATM + BF ∓ RR/2
        return -RR/2 + (BB + self.ATM), RR/2 + (BB + self.ATM)
    
    def _get_strike(self, sigma: nb.float64, delta: nb.float64) -> nb.float64:
        # страйк, соответствующий заданным дельте и IV (через BS-инверсию)
        return self.bs_calc.strike_from_delta(
            Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T)), 
            Delta(delta), 
            ImpliedVol(sigma)
        ).K

    def to_chain_space(self) -> VolSmileChainSpace:      # конвертация дельта-улыбки в страйк-улыбку (5 точек)
        ivs = np.zeros(5, dtype=np.float64)            # IV для 10ΔP, 25ΔP, ATM, 25ΔC, 10ΔC
        strikes = np.zeros(5, dtype=np.float64)        # соответствующие страйки
           
        ivs[2] = self.ATM                              # центр — ATM
        ivs[1], ivs[3] = self._implied_vols(self.RR25, self.BF25)  # 25Δ put/call
        ivs[0], ivs[4] = self._implied_vols(self.RR10, self.BF10)  # 10Δ put/call

        strikes[0] = self._get_strike(ivs[0], -0.1)    # страйк 10Δ put
        strikes[1] = self._get_strike(ivs[1], -0.25)   # страйк 25Δ put
        strikes[2] = self.f                            # ATM-страйк = форвард
        strikes[3] = self._get_strike(ivs[3], 0.25)    # страйк 25Δ call
        strikes[4] = self._get_strike(ivs[4], 0.1)     # страйк 10Δ call
        
        return VolSmileChainSpace(                     # собрать страйк-улыбку
            Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T)),
            Strikes(strikes),
            ImpliedVols(ivs)
        ) 
    
    def forward(self) -> Forward:                       # восстановить форвард
        return Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T))
    
    def blip_ATM(self):                                 # сдвинуть ATM на блип (для конечно-разностных греков по улыбке)
        self.ATM += self.atm_blip
        return self

    def blip_25RR(self):                                # сдвинуть RR25 на блип
        self.RR25 += self.rr25_blip
        return self

    def blip_25BF(self):                                # сдвинуть BF25 на блип
        self.BF25 += self.bf25_blip
        return self

    def blip_10RR(self):                                # сдвинуть RR10 на блип
        self.RR10 += self.rr10_blip
        return self

    def blip_10BF(self):                                # сдвинуть BF10 на блип
        self.BF10 += self.bf10_blip
        return self


@nb.experimental.jitclass([
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
])
class Straddles:                                        # векторная версия Straddle: ATM-вол по нескольким срокам
    def __init__(self, implied_vols: ImpliedVols, times_to_maturity: TimesToMaturity):
        if not implied_vols.data.shape == times_to_maturity.data.shape:  # формы согласованы
            raise ValueError('Inconsistent data between implied vols and times to maturity')
        if not is_sorted(times_to_maturity.data):      # сроки упорядочены
            raise ValueError('Times to maturity are not in order')
        self.sigma = implied_vols.data                 # ATM-волатильности по срокам
        self.T = times_to_maturity.data                # сроки


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
    
])
class RiskReversals:                                    # векторная версия RiskReversal по нескольким срокам
    def __init__(self, delta: Delta, volatility_quotes: VolatilityQuotes, times_to_maturity: TimesToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):       # дельта в [0,1]
            raise ValueError('Delta expected within [0,1]')
        if not volatility_quotes.data.shape == times_to_maturity.data.shape:  # формы согласованы
            raise ValueError('Inconsistent data between quotes and times to maturity')
        if not is_sorted(times_to_maturity.data):      # сроки упорядочены
            raise ValueError('Times to maturity are not in order')
        self.delta = delta.pv                          # уровень дельты
        self.sigma = volatility_quotes.data            # котировки RR по срокам
        self.T = times_to_maturity.data                # сроки


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
])
class Butterflies:                                      # векторная версия Butterfly по нескольким срокам
    def __init__(self, delta: Delta, volatility_quotes: VolatilityQuotes, times_to_maturity: TimesToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):       # дельта в [0,1]
            raise ValueError('Delta expected within [0,1]')
        if not volatility_quotes.data.shape == times_to_maturity.data.shape:  # формы согласованы
            raise ValueError('Inconsistent data between quotes and times to maturity')
        if not is_sorted(times_to_maturity.data):      # сроки упорядочены
            raise ValueError('Times to maturity are not in order')
        self.delta = delta.pv                          # уровень дельты
        self.sigma = volatility_quotes.data            # котировки BF по срокам
        self.T = times_to_maturity.data                # сроки


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("max_T", nb.float64),
    ("min_T", nb.float64)
])
# Поверхность волатильности в дельта-пространстве: PCHIP-сплайны ATM/RR/BF по сроку
class VolSurfaceDeltaSpace:
    FWD: ForwardCurve                                  # форвардная кривая
    ATM: PchipSpline1D                                 # сплайн ATM-дисперсии (T·σ²) по сроку
    RR25: PchipSpline1D                                # сплайн RR25·T по сроку
    BF25: PchipSpline1D                                # сплайн BF25·T по сроку
    RR10: PchipSpline1D                                # сплайн RR10·T по сроку
    BF10: PchipSpline1D                                # сплайн BF10·T по сроку

    def __init__(
        self, 
        forward_curve: ForwardCurve, 
        straddles: Straddles,
        risk_reversals_25: RiskReversals,
        butterflies_25: Butterflies,
        risk_reversals_10: RiskReversals,
        butterflies_10: Butterflies
    ): 
        self.S = forward_curve.S                        # спот

        self.FWD = forward_curve                        # форвардная кривая

        self.ATM = PchipSpline1D(                       # интерполируем суммарную дисперсию T·σ² (так гарантируется монотонность)
            XAxis(straddles.T),
            YAxis(straddles.T*straddles.sigma*straddles.sigma)
        )

        self.RR25 = PchipSpline1D(                      # интерполируем RR25, масштабированное на T
            XAxis(risk_reversals_25.T),
            YAxis(straddles.T*risk_reversals_25.sigma)
        )

        self.BF25 = PchipSpline1D(                      # интерполируем BF25·T
            XAxis(butterflies_25.T),
            YAxis(straddles.T*butterflies_25.sigma)
        )

        self.RR10 = PchipSpline1D(                      # интерполируем RR10·T
            XAxis(risk_reversals_10.T),
            YAxis(straddles.T*risk_reversals_10.sigma)
        )

        self.BF10 = PchipSpline1D(                      # интерполируем BF10·T
            XAxis(butterflies_10.T),
            YAxis(straddles.T*butterflies_10.sigma)
        )

        self.max_T = np.min(np.array([                  # верхняя граница интерполяции = минимум из последних сроков
            straddles.T[-1],
            risk_reversals_25.T[-1],
            butterflies_25.T[-1],
            risk_reversals_10.T[-1],
            butterflies_10.T[-1]
        ]))

        self.min_T = np.max(np.array([                  # нижняя граница = максимум из первых сроков
            straddles.T[0],
            risk_reversals_25.T[0],
            butterflies_25.T[0],
            risk_reversals_10.T[0],
            butterflies_10.T[0]
        ]))
        assert self.max_T > self.min_T                  # валидный диапазон сроков
        assert self.min_T > 0                           # сроки положительны
        
    def get_vol_smile(self, time_to_maturity: TimeToMaturity) -> VolSmileDeltaSpace:  # улыбка на заданный срок
        T = time_to_maturity.T                          # запрашиваемый срок
        if not (T >= self.min_T and T <= self.max_T):  # вне диапазона — предупреждаем о возможном календарном арбитраже
            print('TimeToMaturity outside available bounds, calendar arbitrage possible!')
    
        return VolSmileDeltaSpace(                      # собрать дельта-улыбку из сплайнов (обратное масштабирование на T)
            self.FWD.forward(time_to_maturity),
            Straddle(
                ImpliedVol( np.sqrt(self.ATM.apply(T) / T) ),  # σ_ATM = sqrt(дисперсия/T)
                time_to_maturity
            ),
            RiskReversal(
                Delta(.25),
                VolatilityQuote(self.RR25.apply(T) / T),
                time_to_maturity
            ),
            Butterfly(
                Delta(.25),
                VolatilityQuote(self.BF25.apply(T) / T),
                time_to_maturity
            ),
            RiskReversal(
                Delta(.1),
                VolatilityQuote(self.RR10.apply(T) / T),
                time_to_maturity
            ),
            Butterfly(
                Delta(.1),
                VolatilityQuote(self.BF10.apply(T) / T),
                time_to_maturity
            )
        )

    def forward_curve(self) -> ForwardCurve:            # доступ к форвардной кривой
        return self.forward_curve
    
    def atm_quotes(self, times_to_maturity: TimesToMaturity) -> ImpliedVols:  # ATM-вол по набору сроков
        Ts = times_to_maturity.data
        atm = np.zeros_like(Ts)
        for i in range(len(Ts)):
            atm[i] = np.sqrt(self.ATM.apply(Ts[i]) / Ts[i])  # восстановить σ из T·σ²
        return ImpliedVols(atm)
    
    def rr25_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:  # RR25 по набору сроков
        Ts = times_to_maturity.data
        rr = np.zeros_like(Ts)
        for i in range(len(Ts)):
            rr[i] = self.RR25.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(rr)
    
    def bf25_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:  # BF25 по набору сроков
        Ts = times_to_maturity.data
        bf = np.zeros_like(Ts)
        for i in range(len(Ts)):
            bf[i] = self.BF25.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(bf)
    
    def rr10_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:  # RR10 по набору сроков
        Ts = times_to_maturity.data
        rr = np.zeros_like(Ts)
        for i in range(len(Ts)):
            rr[i] = self.RR10.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(rr)

    def bf10_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:  # BF10 по набору сроков
        Ts = times_to_maturity.data
        bf = np.zeros_like(Ts)
        for i in range(len(Ts)):
            bf[i] = self.BF10.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(bf)


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("Ts", nb.float64[:]),
    ("Ks", nb.float64[:]),
    ("pvs", nb.float64[:]),
    ("sigmas", nb.float64[:]),
    ("compute_implied_vol", nb.boolean)
])
# Поверхность волатильности в страйк-пространстве: цепочка (T, K, premium, IV) с фильтром OTM
class VolSurfaceChainSpace:
    bs_calc: BSCalc                                    # калькулятор BS (инверсия implied vol)
    FWD: ForwardCurve                                  # форвардная кривая
  
    def __init__(
        self,
        forward_curve: ForwardCurve, 
        times_to_maturity: TimesToMaturity,
        strikes: Strikes,
        option_types: OptionTypes,
        premiums: Premiums,
        compute_implied_vol: bool = True               # вычислять ли IV (медленно) или оставить -1
    ):
        if not times_to_maturity.data.shape == strikes.data.shape == premiums.data.shape == option_types.data.shape:  # формы согласованы
            raise ValueError('Inconsistent data shape between times to maturity, strikes, premiums and option types')
        if not np.all(premiums.data > 0):              # премии положительны
            raise ValueError('Invalid premiums data')
        if not is_sorted(times_to_maturity.data):      # сроки упорядочены (нужно для группировки по сроку)
            raise ValueError('Invalid TTM data')
        
        self.bs_calc = BSCalc()                         # BS-калькулятор
        
        self.S = forward_curve.S                        # спот
        self.FWD = forward_curve                        # форвардная кривая

        self.compute_implied_vol = compute_implied_vol  # флаг расчёта IV
        self._process(times_to_maturity.data.flatten(), strikes.data.flatten(), option_types.data.flatten(), premiums.data.flatten())  # обработать котировки

    def _process(self, Ts: nb.float64[:], Ks: nb.float64[:], Cs: nb.float64[:], PVs: nb.float64[:]):
        # отбираем только OTM-опционы и считаем для них implied vol
        lTs = []                                        # накопители результатов:
        lKs = []                                        #   страйки
        lPVs = []                                       #   премии
        lIVs = []                                       #   implied vol
        n = len(Ts)
        
        lT = Ts[0]                                       # текущий срок (для кэширования форварда)
        assert lT > 0

        F = self.FWD.forward(TimeToMaturity(lT))        # форвард на текущем сроке
        f = F.forward_rate().fv                          # форвард-цена

        for i in range(n):
            T = Ts[i]
            assert T >= lT                              # сроки неубывают (данные отсортированы)
            if T > lT:                                  # перешли на новый срок — пересчитать форвард
                F = self.FWD.forward(TimeToMaturity(T))
                f = F.forward_rate().fv
                lT = T

            K = Ks[i]                                   # страйк
            is_call = Cs[i]                             # тип опциона

            if (f > K and is_call) or (K >= f and not is_call):  # пропускаем ITM (call с K<f или put с K>=f)
                continue
            else:                                       # оставляем OTM-котировку
                pv = PVs[i]
                if self.compute_implied_vol:
                    iv = self.bs_calc.implied_vol(F, Strike(K), Premium(pv)).sigma  # инверсия премии в IV
                else:
                    iv = -1                             # IV не считаем (заглушка)

                lTs.append(T)
                lKs.append(K)
                lPVs.append(pv)
                lIVs.append(iv)
        
        self.Ts = np.array(lTs)                          # сохраняем отфильтрованные данные:
        self.Ks = np.array(lKs)
        self.pvs = np.array(lPVs)
        self.sigmas = np.array(lIVs)

    def times_to_maturities(self) -> TimesToMaturity:    # уникальные сроки поверхности
        return TimesToMaturity(np.unique(self.Ts))

    def get_vol_smile(self, time_to_maturity: TimeToMaturity) -> VolSmileChainSpace:  # срез улыбки на одном сроке
        F = self.FWD.forward(time_to_maturity)
        T = time_to_maturity.T
        n = len(self.Ts)

        lKs = []
        lsigmas = []

        for i in range(n):                              # данные отсортированы по T → линейный проход
            cT = self.Ts[i]
            if cT < T:
                continue                                # ещё не дошли до нужного срока
            elif cT == T:
                lKs.append(self.Ks[i])                  # собираем точки нужного срока
                lsigmas.append(self.sigmas[i])
            else:
                break                                   # сроки пошли больше — можно выходить

        Ks = np.array(lKs)
        sigmas = np.array(lsigmas)
        idx = np.argsort(Ks)                            # упорядочить по страйку

        return VolSmileChainSpace(F , Strikes(Ks[idx]), ImpliedVols(sigmas[idx]))
    
    def forward_curve(self) -> ForwardCurve:            # доступ к форвардной кривой
        return self.FWD
    
    def strikes_maturities_grid(self) -> Tuple[StrikesMaturitiesGrid, OptionTypes]:  # сетка (страйк×срок) + типы опционов
        forwards = self.FWD.forward_rates(TimesToMaturity(self.Ts))
        return StrikesMaturitiesGrid(
            self.FWD.spot(),
            TimesToMaturity(self.Ts),
            Strikes(self.Ks)
        ), OptionTypes(forwards.data <= self.Ks)        # call там, где форвард ≤ страйк
        

 