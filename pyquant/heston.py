"""pyquant.heston: полу-аналитический движок модели Хестона (цены + калибровка).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (низкоуровневая количественная библиотека, numba).
НАЗНАЧЕНИЕ:  ценообразование ванильных опционов в модели Хестона через
             характеристическую функцию и квадратуру Гаусса-Лежандра (32 узла),
             а также калибровка параметров (v0, kappa, theta, eps, rho) методом
             Левенберга-Марквардта с АНАЛИТИЧЕСКИМ якобианом (быстро и устойчиво).
ИМПОРТИРУЕТ: numba, numpy; всё из .utils, .common, .vol_surface, .black_scholes.
КОНСТАНТЫ:
  - _U64 / _W64       : узлы и веса 32-точечной квадратуры Гаусса-Лежандра.
  - _LOWER/_UPPER_BOUND, _P, _Q : пределы и масштаб интегрирования по частоте.
  - _ZERO/_ONE/_TWO/_I/_PI : комплексные/действительные числовые константы.
ЭКСПОРТИРУЕТ: Variance, VarReversion, AverageVar, VolOfVar, Correlation,
             FlatForwardYield, HestonParams, HestonCalc.
КЕМ ИСПОЛЬЗУЕТСЯ: services.heston_pricing (обёртка над движком) → весь проект.
ПРИМЕЧАНИЕ:  методы _hes_int_MN / _hes_int_jac реализуют комплексный интегранд
             характеристической функции и её аналитический якобиан; математика
             плотная — комментарии поясняют структуру (что вычисляется), а не
             каждую промежуточную алгебраическую переменную.
=============================================================================
"""

import numba as nb                                       # JIT-компиляция
from numba.experimental import jitclass                  # декоратор jitclass
from typing import Final, Tuple                          # аннотации (Final-константы)

from .utils import *                                     # normal_cdf/pdf, np_clip, сплайны
from .common import *                                    # value-объекты (Spot, Strike, греки …)
from .vol_surface import *                               # VolSurfaceChainSpace для калибровки
from .black_scholes import *                             # BSCalc (инверсия IV на поверхности)


################# Helper classes and variables for HestonCalc #################
# noinspection DuplicatedCode
@nb.experimental.jitclass([
    ("M1", nb.float64[:]),
    ("N1", nb.float64[:]),
    ("M2", nb.float64[:]),
    ("N2", nb.float64[:]),
])
class _TagMn(object):                                   # контейнер интегрантов M1/N1/M2/N2 (две вероятности P1,P2)
    M1: nb.float64[:]
    N1: nb.float64[:]
    M2: nb.float64[:]
    N2: nb.float64[:]

    def __init__(
        self, M1: nb.float64[:], N1: nb.float64[:], M2: nb.float64[:], N2: nb.float64[:]
    ):
        self.M1 = M1
        self.M2 = M2
        self.N1 = N1
        self.N2 = N2


@nb.experimental.jitclass([
    ("pa1s", nb.float64[:]),
    ("pa2s", nb.float64[:]),
    ("pb1s", nb.float64[:]),
    ("pb2s", nb.float64[:]),
    ("pc1s", nb.float64[:]),
    ("pc2s", nb.float64[:]),
    ("prho1s", nb.float64[:]),
    ("prho2s", nb.float64[:]),
    ("pv01s", nb.float64[:]),
    ("pv02s", nb.float64[:]),
])
class _TagMNJac(object):                                # контейнер интегрантов якобиана по параметрам (a,b,c,rho,v0)
    pa1s: nb.float64[:]
    pa2s: nb.float64[:]
    pb1s: nb.float64[:]
    pb2s: nb.float64[:]
    pc1s: nb.float64[:]
    pc2s: nb.float64[:]
    prho1s: nb.float64[:]
    prho2s: nb.float64[:]
    pv01s: nb.float64[:]
    pv02s: nb.float64[:]

    def __init__(
        self,
        pa1s: nb.float64[:],
        pa2s: nb.float64[:],
        pb1s: nb.float64[:],
        pb2s: nb.float64[:],
        pc1s: nb.float64[:],
        pc2s: nb.float64[:],
        prho1s: nb.float64[:],
        prho2s: nb.float64[:],
        pv01s: nb.float64[:],
        pv02s: nb.float64[:],
    ):
        self.pa1s = pa1s
        self.pa2s = pa2s
        self.pb1s = pb1s
        self.pb2s = pb2s
        self.pc1s = pc1s
        self.pc2s = pc2s
        self.prho1s = prho1s
        self.prho2s = prho2s
        self.pv01s = pv01s
        self.pv02s = pv02s


_ZERO = np.complex128(complex(0.0, 0.0))                 # комплексный 0
_ONE = np.complex128(complex(1.0, 0.0))                  # комплексная 1
_TWO = np.complex128(complex(2.0, 0.0))                  # комплексная 2
_I = np.complex128(complex(0.0, 1.0))                    # мнимая единица i
_PI: Final[np.float64] = np.pi                           # число π
_LOWER_BOUND: Final[np.float64] = np.float64(0.0)        # нижний предел интегрирования по частоте
_UPPER_BOUND: Final[np.int32] = np.int32(200)            # верхний предел интегрирования
_Q: Final[np.float64] = np.float64(0.5 * (_UPPER_BOUND - _LOWER_BOUND))  # полудлина интервала (масштаб)
_P: Final[np.float64] = np.float64(0.5 * (_UPPER_BOUND + _LOWER_BOUND))  # центр интервала (сдвиг)

# points in which quadratures are computed
_U64 = np.array(                                         # 32 узла квадратуры Гаусса-Лежандра (на [0,1])
    [
        0.0243502926634244325089558,
        0.0729931217877990394495429,
        0.1214628192961205544703765,
        0.1696444204239928180373136,
        0.2174236437400070841496487,
        0.2646871622087674163739642,
        0.3113228719902109561575127,
        0.3572201583376681159504426,
        0.4022701579639916036957668,
        0.4463660172534640879849477,
        0.4894031457070529574785263,
        0.5312794640198945456580139,
        0.5718956462026340342838781,
        0.6111553551723932502488530,
        0.6489654712546573398577612,
        0.6852363130542332425635584,
        0.7198818501716108268489402,
        0.7528199072605318966118638,
        0.7839723589433414076102205,
        0.8132653151227975597419233,
        0.8406292962525803627516915,
        0.8659993981540928197607834,
        0.8893154459951141058534040,
        0.9105221370785028057563807,
        0.9295691721319395758214902,
        0.9464113748584028160624815,
        0.9610087996520537189186141,
        0.9733268277899109637418535,
        0.9833362538846259569312993,
        0.9910133714767443207393824,
        0.9963401167719552793469245,
        0.9993050417357721394569056,
    ],
    dtype=np.float64,
)

# Gaussian quadrature weights from 0 to 1 (because we integrate from zero)
_W64 = np.array(                                         # 32 веса квадратуры Гаусса-Лежандра
    [
        0.0486909570091397203833654,
        0.0485754674415034269347991,
        0.0483447622348029571697695,
        0.0479993885964583077281262,
        0.0475401657148303086622822,
        0.0469681828162100173253263,
        0.0462847965813144172959532,
        0.0454916279274181444797710,
        0.0445905581637565630601347,
        0.0435837245293234533768279,
        0.0424735151236535890073398,
        0.0412625632426235286101563,
        0.0399537411327203413866569,
        0.0385501531786156291289625,
        0.0370551285402400460404151,
        0.0354722132568823838106931,
        0.0338051618371416093915655,
        0.0320579283548515535854675,
        0.0302346570724024788679741,
        0.0283396726142594832275113,
        0.0263774697150546586716918,
        0.0243527025687108733381776,
        0.0222701738083832541592983,
        0.0201348231535302093723403,
        0.0179517157756973430850453,
        0.0157260304760247193219660,
        0.0134630478967186425980608,
        0.0111681394601311288185905,
        0.0088467598263639477230309,
        0.0065044579689783628561174,
        0.0041470332605624676352875,
        0.0017832807216964329472961,
    ],
    dtype=np.float64,
)

##############################################################################


@nb.experimental.jitclass([
    ("v0", nb.float64)
])
class Variance:                                         # параметр Хестона: начальная дисперсия v0
    def __init__(self, v0: nb.float64):
        self.v0 = v0


@nb.experimental.jitclass([
    ("kappa", nb.float64)
])
class VarReversion:                                     # параметр Хестона: скорость возврата к среднему kappa
    def __init__(self, kappa: nb.float64):
        self.kappa = kappa


@nb.experimental.jitclass([
    ("theta", nb.float64)
])
class AverageVar:                                       # параметр Хестона: долгосрочная дисперсия theta
    def __init__(self, theta: nb.float64):
        self.theta = theta


@nb.experimental.jitclass([
    ("eps", nb.float64)
])
class VolOfVar:                                         # параметр Хестона: волатильность волатильности eps
    def __init__(self, eps: nb.float64):
        self.eps = eps


@nb.experimental.jitclass([
    ("rho", nb.float64)
])
class Correlation:                                      # параметр Хестона: корреляция спот-вол rho
    def __init__(self, rho: nb.float64):
        self.rho = rho


@nb.experimental.jitclass([
    ("r", nb.float64)
])
class FlatForwardYield:                                 # плоская форвардная доходность r (ставка)
    def __init__(self, r: nb.float64):
        self.r = r


@jitclass([
    ("v0", nb.float64),
    ("kappa", nb.float64),
    ("theta", nb.float64),
    ("eps", nb.float64),
    ("rho", nb.float64),
    ("r", nb.float64)
])
class HestonParams:                                     # полный набор параметров Хестона (v0,kappa,theta,eps,rho,r)
    def __init__(
        self,
        variance: Variance,
        var_reversion: VarReversion,
        average_var: AverageVar,
        vol_var: VolOfVar,
        correlation: Correlation, 
        flat_forward_yield: FlatForwardYield
    ):
        self.v0 = variance.v0                          # начальная дисперсия
        self.kappa = var_reversion.kappa               # скорость возврата
        self.theta = average_var.theta                 # долгосрочная дисперсия
        self.eps = vol_var.eps                         # вол-оф-вол
        self.rho = correlation.rho                     # корреляция
        self.r = flat_forward_yield.r                  # ставка

    def array(self) -> nb.float64[:]:
        return np.array([self.v0, self.kappa, self.theta, self.eps, self.rho])  # параметры как вектор (для оптимизатора)
    
    def flat_forward_yield(self) -> FlatForwardYield:
        return FlatForwardYield(self.r)                # ставка как объект


# noinspection DuplicatedCode
@jitclass([
    ("cached_params", nb.float64[:]),
    ("num_iter", nb.int64),
    ("tol", nb.float64)
])
class HestonCalc:                                       # движок Хестона: ценообразование + калибровка (LM)
    bs_calc: BSCalc

    def __init__(self):
        self.cached_params = np.array([1.,1.,1.,1.,0.])  # кэш параметров (тёплый старт калибровки)
        self.num_iter = 50                             # макс. итераций Левенберга-Марквардта
        self.tol = 1e-8                                # допуск сходимости

        self.bs_calc = BSCalc()                        # вложенный калькулятор БШ (для инверсии IV)
      
    def update_cached_params(self, params: HestonParams):
        self.cached_params = params.array()            # обновить тёплый старт

    def calibrate(
        self,
        market_chain: VolSurfaceChainSpace,
        flat_forward_yield: FlatForwardYield,
        calibration_weights: CalibrationWeights
    ) -> Tuple[HestonParams, CalibrationError]:        # калибровка параметров по рыночной цепочке (МНК премий)
        w = calibration_weights.w                      # веса наблюдений
        if not w.shape == market_chain.pvs.shape:      # формы согласованы
            raise ValueError('Inconsistent data shape between `calibration_weights` and `market_chain`')
        weights = w / w.sum()                          # нормированные веса

        n_points = len(market_chain.pvs)               # число точек цепочки
        grid, is_call = market_chain.strikes_maturities_grid()  # сетка (страйк,срок) и типы
        
        PARAMS_TO_CALIBRATE: nb.int64 = 5              # калибруем 5 параметров
        if not n_points - PARAMS_TO_CALIBRATE >= 0:    # точек должно быть ≥ 5
            raise ValueError('Need at least 5 points to calibrate Heston model')

        def clip_params(params: np.ndarray) -> np.ndarray:  # проекция параметров в допустимую область
            small = 1e-4
            v0, kappa, theta, eps, rho = params[0], params[1], params[2], params[3], params[4]
            v0 = np_clip(v0, small, 10.0)              #   v0 ∈ [ε, 10]
            kappa = np_clip(kappa, small, 500.0)       #   kappa ∈ [ε, 500]
            theta = np_clip(theta, small, 500.0)       #   theta ∈ [ε, 500]
            eps = np_clip(eps, small, 150.0)           #   eps ∈ [ε, 150]
            rho = np_clip(rho, -1.0 + small, 1.0 - small)  #   rho ∈ (-1, 1)
            clipped_params = np.array([v0, kappa, theta, eps, rho])
            return clipped_params

        def get_residuals(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # невязки и якобиан в точке params
            heston_params = HestonParams(             #   собрать параметры
                Variance(params[0]),
                VarReversion(params[1]),
                AverageVar(params[2]),
                VolOfVar(params[3]),
                Correlation(params[4]),
                flat_forward_yield
            )
            premiums = self._grid_premiums(heston_params, grid, is_call)  #   модельные премии
            residuals = (premiums - market_chain.pvs) * weights  #   взвешенные невязки
            jacobian = self._jac_hes(heston_params, market_chain) @ np.diag(weights)  #   взвешенный якобиан
            return residuals, jacobian

        def levenberg_marquardt(f, proj, x0):          # минимизатор Левенберга-Марквардта
            x = x0.copy()                              #   текущая точка

            mu = 1e-2                                  #   демпфирующий множитель
            nu1 = 2.0                                  #   коэффициент уменьшения mu при успехе
            nu2 = 2.0                                  #   коэффициент увеличения mu при неудаче

            res, J = f(x)                              #   невязки и якобиан
            F = res.T @ res                            #   текущая сумма квадратов

            result_x = x                               #   лучшее решение
            result_error = F / n_points                #   средняя ошибка

            for i in range(self.num_iter):             #   итерации LM:
                if result_error < self.tol:            #     сошлось…
                    break
                multipl = J @ J.T                      #     приближение Гессиана J·Jᵀ
                I = np.diag(np.diag(multipl)) + 1e-5 * np.eye(len(x))  #     демпфирующая диагональ
                dx = np.linalg.solve(mu * I + multipl, J @ res)  #     шаг (решение СЛАУ)
                x_ = proj(x - dx)                      #     новый кандидат (с проекцией)
                res_, J_ = f(x_)                       #     невязки/якобиан кандидата
                F_ = res_.T @ res_                     #     его сумма квадратов
                if F_ < F:                             #     улучшение…
                    x, F, res, J = x_, F_, res_, J_    #       принимаем шаг
                    mu /= nu1                          #       уменьшаем демпфирование
                    result_error = F / n_points
                else:                                  #     ухудшение…
                    i -= 1
                    mu *= nu2                          #       усиливаем демпфирование
                    continue
                result_x = x                           #     обновляем лучшее

            return result_x, result_error              #   параметры и ошибка

        calc_params, calibration_error \
            = levenberg_marquardt(get_residuals, clip_params, self.cached_params)  # запуск калибровки

        return HestonParams(                           # результат: параметры + ошибка
            Variance(calc_params[0]),
            VarReversion(calc_params[1]),
            AverageVar(calc_params[2]),
            VolOfVar(calc_params[3]),
            Correlation(calc_params[4]),
            flat_forward_yield
        ), CalibrationError(calibration_error)

    def surface_grid_ivs(self, params: HestonParams, grid: StrikesMaturitiesGrid) -> ImpliedVols:  # IV-поверхность из параметров
        Fs = grid.S * np.exp(params.r * grid.Ts)       # форвардные цены по срокам
        is_call = OptionTypes(Fs <= grid.Ks)           # OTM-выбор типа (call если страйк ≥ форвард)
        pvs = self._grid_premiums(params, grid, is_call)  # модельные премии
    
        ivs = np.zeros_like(pvs)                       # результат — IV
        r = ForwardYield(params.r)                     # ставка
        r_d = DiscountYield(params.r)                  # ставка дисконтирования
        S = grid.spot()                                # спот

        for i in range(len(pvs)):                       # по каждой точке сетки…
            T = TimeToMaturity(grid.Ts[i])             #   срок
            F = Forward(S,r,r_d,T)                      #   форвард
            ivs[i] = self.bs_calc.implied_vol(F, Strike(grid.Ks[i]), Premium(pvs[i])).sigma  #   инверсия премии в IV
    
        return ImpliedVols(ivs)

    def _hes_int_jac(
        self,
        heston_params: HestonParams,
        grid: StrikesMaturitiesGrid,
        market_pointer: int,
    ) -> _TagMNJac:
        """Calculates real-valued integrands for Jacobian."""
        # Вычисляет действительные интегранты частных производных характеристической
        # функции по параметрам (kappa→a, theta→b, eps→c, rho, v0) в узлах квадратуры.
        # Ниже — плотная комплексная алгебра двух P-функций (индексы M/N — две половины
        # симметричной квадратуры, индексы 1/2 — две вероятности P1, P2).
        PQ_M, PQ_N = _P + _Q * _U64, _P - _Q * _U64
        imPQ_M = _I * PQ_M
        imPQ_N = _I * PQ_N
        _imPQ_M = _I * (PQ_M - _I)
        _imPQ_N = _I * (PQ_N - _I)

        Ks = grid.Ks
        Ts = grid.Ts

        h_M = np.divide(np.power(Ks[market_pointer], -imPQ_M), imPQ_M)
        h_N = np.divide(np.power(Ks[market_pointer], -imPQ_N), imPQ_N)

        x0 = (
            np.log(grid.S)
            + heston_params.r * Ts[market_pointer]
        )
        tmp = heston_params.eps * heston_params.rho
        kes_M1 = heston_params.kappa - np.multiply(tmp, _imPQ_M)
        kes_N1 = heston_params.kappa - np.multiply(tmp, _imPQ_N)
        kes_M2 = kes_M1 + tmp
        kes_N2 = kes_N1 + tmp

        _msqr = np.power(PQ_M - _I, 2)
        _nsqr = np.power(PQ_N - _I, 2)
        msqr = np.power(PQ_M - _ZERO * _I, 2)
        nsqr = np.power(PQ_N - _ZERO * _I, 2)

        m_M1 = imPQ_M + _ONE + _msqr
        m_N1 = imPQ_N + _ONE + _nsqr
        m_M2 = imPQ_M + msqr
        m_N2 = imPQ_N + nsqr

        csqr = np.power(heston_params.eps, 2)
        d_M1 = np.sqrt(np.power(kes_M1, 2) + m_M1 * csqr)
        d_N1 = np.sqrt(np.power(kes_N1, 2) + m_N1 * csqr)
        d_M2 = np.sqrt(np.power(kes_M2, 2) + m_M2 * csqr)
        d_N2 = np.sqrt(np.power(kes_N2, 2) + m_N2 * csqr)

        abrt = (
            heston_params.kappa
            * heston_params.theta
            * heston_params.rho
            * Ts[market_pointer]
        )
        tmp1 = -abrt / heston_params.eps
        tmp2 = np.exp(tmp1)

        g_M2 = np.exp(tmp1 * imPQ_M)
        g_N2 = np.exp(tmp1 * imPQ_N)
        g_M1 = g_M2 * tmp2
        g_N1 = g_N2 * tmp2

        halft = 0.5 * Ts[market_pointer]
        alpha = d_M1 * halft
        calp_M1 = np.cosh(alpha)
        salp_M1 = np.sinh(alpha)

        alpha = d_N1 * halft
        calp_N1 = np.cosh(alpha)
        salp_N1 = np.sinh(alpha)

        alpha = d_M2 * halft
        calp_M2 = np.cosh(alpha)
        salp_M2 = np.sinh(alpha)

        alpha = d_N2 * halft
        calp_N2 = np.cosh(alpha)
        salp_N2 = np.sinh(alpha)

        # A2 = d*calp + kes*salp;
        A2_M1 = d_M1 * calp_M1 + kes_M1 * salp_M1
        A2_N1 = d_N1 * calp_N1 + kes_N1 * salp_N1
        A2_M2 = d_M2 * calp_M2 + kes_M2 * salp_M2
        A2_N2 = d_N2 * calp_N2 + kes_N2 * salp_N2

        # A1 = m*salp;
        A1_M1 = m_M1 * salp_M1
        A1_N1 = m_N1 * salp_N1
        A1_M2 = m_M2 * salp_M2
        A1_N2 = m_N2 * salp_N2

        # A = A1/A2;
        A_M1 = A1_M1 / A2_M1
        A_N1 = A1_N1 / A2_N1
        A_M2 = A1_M2 / A2_M2
        A_N2 = A1_N2 / A2_N2

        # B = d*exp(a*T/2)/A2;
        tmp = np.exp(heston_params.kappa * halft)
        # exp(a*T/2)
        B_M1 = d_M1 * tmp / A2_M1
        B_N1 = d_N1 * tmp / A2_N1
        B_M2 = d_M2 * tmp / A2_M2
        B_N2 = d_N2 * tmp / A2_N2

        # characteristic function: y1 = exp(i*x0*u1) * exp(-v0*A) * g * exp(2*a*b/pow(c,2)*D)
        tmp3 = 2 * heston_params.kappa * heston_params.theta / csqr
        D_M1 = (
            np.log(d_M1)
            + (heston_params.kappa - d_M1) * halft
            - np.log(
            (d_M1 + kes_M1) * 0.5
            + (d_M1 - kes_M1)
            * 0.5
            * np.exp(-d_M1 * Ts[market_pointer])
        )
        )
        D_M2 = (
            np.log(d_M2)
            + (heston_params.kappa - d_M2) * halft
            - np.log(
            (d_M2 + kes_M2) * 0.5
            + (d_M1 - kes_M2)
            * 0.5
            * np.exp(-d_M2 * Ts[market_pointer])
        )
        )
        D_N1 = (
            np.log(d_N1)
            + (heston_params.kappa - d_N1) * halft
            - np.log(
            (d_N1 + kes_N1) * 0.5
            + (d_N1 - kes_N1)
            * 0.5
            * np.exp(-d_N1 * Ts[market_pointer])
        )
        )
        D_N2 = (
            np.log(d_N2)
            + (heston_params.kappa - d_N2) * halft
            - np.log(
            (d_N2 + kes_N2) * 0.5
            + (d_N2 - kes_N2)
            * 0.5
            * np.exp(-d_N2 * Ts[market_pointer])
        )
        )

        y1M1 = np.exp(x0 * _imPQ_M - heston_params.v0 * A_M1 + tmp3 * D_M1) * g_M1
        y1N1 = np.exp(x0 * _imPQ_N - heston_params.v0 * A_N1 + tmp3 * D_N1) * g_N1
        y1M2 = np.exp(x0 * imPQ_M - heston_params.v0 * A_M2 + tmp3 * D_M2) * g_M2
        y1N2 = np.exp(x0 * imPQ_N - heston_params.v0 * A_N2 + tmp3 * D_N2) * g_N2

        # H = kes*calp + d*salp;
        H_M1 = kes_M1 * calp_M1 + d_M1 * salp_M1
        H_M2 = kes_M2 * calp_M2 + d_M2 * salp_M2
        H_N1 = kes_N1 * calp_N1 + d_N1 * salp_N1
        H_N2 = kes_N2 * calp_N2 + d_N2 * salp_N2

        # lnB = log(B);
        lnB_M1, lnB_M2, lnB_N1, lnB_N2 = D_M1, D_M2, D_N1, D_N2

        # partial b: y3 = y1*(2*a*lnB/pow(c,2)-a*rho*T*u1*i/c);
        tmp4 = tmp3 / heston_params.theta
        tmp5 = tmp1 / heston_params.theta

        y3M1 = tmp4 * lnB_M1 + tmp5 * _imPQ_M
        y3M2 = tmp4 * lnB_M2 + tmp5 * imPQ_M
        y3N1 = tmp4 * lnB_N1 + tmp5 * _imPQ_N
        y3N2 = tmp4 * lnB_N2 + tmp5 * imPQ_N

        # partial rho:
        tmp1 = tmp1 / heston_params.rho  # //-a*b*T/c;

        # for M1
        ctmp = heston_params.eps * _imPQ_M / d_M1
        pd_prho_M1 = -kes_M1 * ctmp
        pA1_prho_M1 = m_M1 * calp_M1 * halft * pd_prho_M1
        pA2_prho_M1 = -ctmp * H_M1 * (_ONE + kes_M1 * halft)
        pA_prho_M1 = (pA1_prho_M1 - A_M1 * pA2_prho_M1) / A2_M1
        ctmp = pd_prho_M1 - pA2_prho_M1 * d_M1 / A2_M1
        pB_prho_M1 = tmp / A2_M1 * ctmp
        y4M1 = -heston_params.v0 * pA_prho_M1 + tmp3 * ctmp / d_M1 + tmp1 * _imPQ_M

        # for N1
        ctmp = heston_params.eps * _imPQ_N / d_N1
        pd_prho_N1 = -kes_N1 * ctmp
        pA1_prho_N1 = m_N1 * calp_N1 * halft * pd_prho_N1
        pA2_prho_N1 = -ctmp * H_N1 * (_ONE + kes_N1 * halft)
        pA_prho_N1 = (pA1_prho_N1 - A_N1 * pA2_prho_N1) / A2_N1
        ctmp = pd_prho_N1 - pA2_prho_N1 * d_N1 / A2_N1
        pB_prho_N1 = tmp / A2_N1 * ctmp
        y4N1 = -heston_params.v0 * pA_prho_N1 + tmp3 * ctmp / d_N1 + tmp1 * _imPQ_N

        # for M2
        ctmp = heston_params.eps * imPQ_M / d_M2
        pd_prho_M2 = -kes_M2 * ctmp
        pA1_prho_M2 = m_M2 * calp_M2 * halft * pd_prho_M2
        pA2_prho_M2 = -ctmp * H_M2 * (_ONE + kes_M2 * halft) / d_M2
        pA_prho_M2 = (pA1_prho_M2 - A_M2 * pA2_prho_M2) / A2_M2
        ctmp = pd_prho_M2 - pA2_prho_M2 * d_M2 / A2_M2
        pB_prho_M2 = tmp / A2_M2 * ctmp
        y4M2 = -heston_params.v0 * pA_prho_M2 + tmp3 * ctmp / d_M2 + tmp1 * imPQ_M

        # for N2
        ctmp = heston_params.eps * imPQ_N / d_N2
        pd_prho_N2 = -kes_N2 * ctmp
        pA1_prho_N2 = m_N2 * calp_N2 * halft * pd_prho_N2
        pA2_prho_N2 = -ctmp * H_N2 * (_ONE + kes_N2 * halft)
        pA_prho_N2 = (pA1_prho_N2 - A_N2 * pA2_prho_N2) / A2_N2
        ctmp = pd_prho_N2 - pA2_prho_N2 * d_N2 / A2_N2
        pB_prho_N2 = tmp / A2_N2 * ctmp
        y4N2 = -heston_params.v0 * pA_prho_N2 + tmp3 * ctmp / d_N2 + tmp1 * imPQ_N

        # partial a:
        tmp1 = (
            heston_params.theta
            * heston_params.rho
            * Ts[market_pointer]
            / heston_params.eps
        )
        tmp2 = tmp3 / heston_params.kappa  # 2*b/csqr;
        ctmp = -_ONE / (heston_params.eps * _imPQ_M)

        pB_pa = ctmp * pB_prho_M1 + B_M1 * halft
        y5M1 = (
            -heston_params.v0 * pA_prho_M1 * ctmp
            + tmp2 * lnB_M1
            + heston_params.kappa * tmp2 * pB_pa / B_M1
            - tmp1 * _imPQ_M
        )

        ctmp = -_ONE / (heston_params.eps * imPQ_M)
        pB_pa = ctmp * pB_prho_M2 + B_M2 * halft
        y5M2 = (
            -heston_params.v0 * pA_prho_M2 * ctmp
            + tmp2 * lnB_M2
            + heston_params.kappa * tmp2 * pB_pa / B_M2
            - tmp1 * imPQ_M
        )

        ctmp = -_ONE / (heston_params.eps * _imPQ_N)
        pB_pa = ctmp * pB_prho_N1 + B_N1 * halft
        y5N1 = (
            -heston_params.v0 * pA_prho_N1 * ctmp
            + tmp2 * lnB_N1
            + heston_params.kappa * tmp2 * pB_pa / B_N1
            - tmp1 * _imPQ_N
        )
        # NOTE: here is a ZeroDivisionError if wrong P, Q
        ctmp = -_ONE / (heston_params.eps * imPQ_N)
        pB_pa = ctmp * pB_prho_N2 + B_N2 * halft

        y5N2 = (
            -heston_params.v0 * pA_prho_N2 * ctmp
            + tmp2 * lnB_N2
            + heston_params.kappa * tmp2 * pB_pa / B_N2
            - tmp1 * imPQ_N
        )

        # partial c:
        tmp = heston_params.rho / heston_params.eps
        tmp1 = 4 * heston_params.kappa * heston_params.theta / np.power(heston_params.eps, 3)
        tmp2 = abrt / csqr
        # M1
        pd_pc = (tmp - _ONE / kes_M1) * pd_prho_M1 + heston_params.eps * _msqr / d_M1
        pA1_pc = m_M1 * calp_M1 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_M1
            - _ONE
            / _imPQ_M
            * (_TWO / (Ts[market_pointer] * kes_M1) + _ONE)
            * pA1_prho_M1
            + heston_params.eps * halft * A1_M1
        )
        pA_pc = pA1_pc / A2_M1 - A_M1 / A2_M1 * pA2_pc

        y6M1 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_M1
            + tmp3 / d_M1 * (pd_pc - d_M1 / A2_M1 * pA2_pc)
            + tmp2 * _imPQ_M
        )

        # M2
        pd_pc = (tmp - _ONE / kes_M2) * pd_prho_M2 + heston_params.eps * msqr / d_M2
        pA1_pc = m_M2 * calp_M2 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_M2
            - _ONE
            / imPQ_M
            * (_TWO / (Ts[market_pointer] * kes_M2) + _ONE)
            * pA1_prho_M2
            + heston_params.eps * halft * A1_M2
        )
        pA_pc = pA1_pc / A2_M2 - A_M2 / A2_M2 * pA2_pc
        y6M2 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_M2
            + tmp3 / d_M2 * (pd_pc - d_M2 / A2_M2 * pA2_pc)
            + tmp2 * imPQ_M
        )

        # N1
        pd_pc = (tmp - _ONE / kes_N1) * pd_prho_N1 + heston_params.eps * _nsqr / d_N1
        pA1_pc = m_N1 * calp_N1 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_N1
            - _ONE
            / (_imPQ_N)
            * (_TWO / (Ts[market_pointer] * kes_N1) + _ONE)
            * pA1_prho_N1
            + heston_params.eps * halft * A1_N1
        )
        pA_pc = pA1_pc / A2_N1 - A_N1 / A2_N1 * pA2_pc
        y6N1 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_N1
            + tmp3 / d_N1 * (pd_pc - d_N1 / A2_N1 * pA2_pc)
            + tmp2 * _imPQ_N
        )

        # N2
        pd_pc = (tmp - _ONE / kes_N2) * pd_prho_N2 + heston_params.eps * nsqr / d_N2
        pA1_pc = m_N2 * calp_N2 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_N2
            - _ONE
            / (imPQ_N)
            * (_TWO / (Ts[market_pointer] * kes_N2) + _ONE)
            * pA1_prho_N2
            + heston_params.eps * halft * A1_N2
        )
        pA_pc = pA1_pc / A2_N2 - A_N2 / A2_N2 * pA2_pc
        y6N2 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_N2
            + tmp3 / d_N2 * (pd_pc - d_N2 / A2_N2 * pA2_pc)
            + tmp2 * imPQ_N
        )

        hM1 = h_M * y1M1
        hM2 = h_M * y1M2
        hN1 = h_N * y1N1
        hN2 = h_N * y1N2

        jacobian = _TagMNJac(
            np.real(hM1 * y5M1 + hN1 * y5N1),
            np.real(hM2 * y5M2 + hN2 * y5N2),
            np.real(hM1 * y3M1 + hN1 * y3N1),
            np.real(hM2 * y3M2 + hN2 * y3N2),
            np.real(hM1 * y6M1 + hN1 * y6N1),
            np.real(hM2 * y6M2 + hN2 * y6N2),
            np.real(hM1 * y4M1 + hN1 * y4N1),
            np.real(hM2 * y4M2 + hN2 * y4N2),
            np.real(-hM1 * A_M1 - hN1 * A_N1),
            np.real(-hM2 * A_M2 - hN2 * A_N2),
        )
        return jacobian

    def _grid_premiums(
        self,
        heston_params: HestonParams,
        grid: StrikesMaturitiesGrid,
        option_types: OptionTypes
    ) -> nb.float64[:]:
        """Calculates the premium of vanilla option under the Heston model."""
        # Цена опциона = квадратура характеристической функции по 32 узлам Гаусса-Лежандра.

        Ks = grid.Ks                                   # страйки
        Ts = grid.Ts                                   # сроки
        is_call = option_types.data                    # типы (call/put)
        assert Ks.shape == is_call.shape

        x = np.zeros_like(Ks)                          # результат — премии

        for l in range(len(Ks)):                        # по каждой точке сетки…
            K = Ks[l]                                  #   страйк
            T = Ts[l]                                  #   срок
            disc = np.exp(-heston_params.r * T)        #   фактор дисконтирования
            tmp = 0.5 * (grid.S - K * disc)            #   слагаемое паритета (S − K·disc)/2
            # tmp = 0.5 * (market_parameters.S - K) * disc
            disc = disc / _PI                          #   масштаб 1/π для квадратуры
            y1, y2 = nb.float64(0.0), nb.float64(0.0)  #   накопители интегралов P1, P2

            MN: _TagMn = self._hes_int_MN(heston_params, grid, np.int32(l))  # интегранты в узлах

            y1 = y1 + np.multiply(_W64, (MN.M1 + MN.N1)).sum()  #   взвешенная сумма (P1)
            y2 = y2 + np.multiply(_W64, (MN.M2 + MN.N2)).sum()  #   взвешенная сумма (P2)
            Qv1, Qv2 = np.float64(0.0), nb.float64(0.0)
            Qv1 = _Q * y1                              #   масштабирование интеграла P1
            Qv2 = _Q * y2                              #   масштабирование интеграла P2
            pv = np.float64(0.0)                       #   премия
            delta = 0.5 + Qv1 / _PI                    #   вероятность (диагностика)
            # print(delta)

            if is_call[l]:                             #   call vs put:
                # calls
                # p1 = market_parameters.S*(0.5 + Qv1/pi)
                # p2 = K*np.exp(-market_parameters.r * T)*(0.5 + Qv2/pi)
                # orig = p1 - p2
                pv = disc * (Qv1 - K * Qv2) + tmp      #   премия колла
                # print(pv, orig)
            else:
                # puts
                # p1 = market_parameters.S*(- 0.5 + Qv1/pi)
                # p2 = K*np.exp(-market_parameters.r * T)*(- 0.5 + Qv2/pi)
                # orig = p1 - p2
                pv = disc * (Qv1 - K * Qv2) - tmp      #   премия пута
            x[l] = pv if pv >= 0.01 else 0.            #   отсекаем мизерные/отрицательные премии
        return x                                       # массив премий

    def _hes_int_MN(
        self,
        heston_params: HestonParams,
        grid: StrikesMaturitiesGrid,
        market_pointer: int,
    ) -> _TagMn:
        # Действительные интегранты характеристической функции Хестона в узлах
        # квадратуры (M/N — две половины симметричной квадратуры; 1/2 — P1, P2).
        Ks = grid.Ks                                   # страйки
        Ts = grid.Ts                                   # сроки

        csqr = np.power(heston_params.eps, 2)          # eps² (часто используется)

        PQ_M, PQ_N = _P + _Q * _U64, _P - _Q * _U64
        imPQ_M = _I * PQ_M
        imPQ_N = _I * PQ_N
        _imPQ_M = _I * (PQ_M - _I)
        _imPQ_N = _I * (PQ_N - _I)

        h_M = np.divide(np.power(Ks[market_pointer], -imPQ_M), imPQ_M)
        h_N = np.divide(np.power(Ks[market_pointer], -imPQ_N), imPQ_N)

        x0 = (
        np.log(grid.S)
        + heston_params.r * Ts[market_pointer]
        )
        tmp = heston_params.eps * heston_params.rho

        kes_M1 = heston_params.kappa - np.multiply(tmp, _imPQ_M)
        kes_N1 = heston_params.kappa - np.multiply(tmp, _imPQ_N)
        kes_M2 = kes_M1 + tmp
        kes_N2 = kes_N1 + tmp

        m_M1 = imPQ_M + _ONE + np.power(PQ_M - _I, 2)
        m_N1 = imPQ_N + _ONE + np.power(PQ_N - _I, 2)
        m_M2 = imPQ_M + np.power(PQ_M - _ZERO * _I, 2)
        m_N2 = imPQ_N + np.power(PQ_N - _ZERO * _I, 2)

        d_M1 = np.sqrt(np.power(kes_M1, 2) + m_M1 * csqr)
        d_N1 = np.sqrt(np.power(kes_N1, 2) + m_N1 * csqr)
        d_M2 = np.sqrt(np.power(kes_M2, 2) + m_M2 * csqr)
        d_N2 = np.sqrt(np.power(kes_N2, 2) + m_N2 * csqr)

        tmp1 = (
        -heston_params.kappa
        * heston_params.theta
        * heston_params.rho
        * Ts[market_pointer]
        / heston_params.eps
        )

        tmp = np.exp(tmp1)
        g_M2 = np.exp(tmp1 * imPQ_M)
        g_N2 = np.exp(tmp1 * imPQ_N)
        g_M1 = g_M2 * tmp
        g_N1 = g_N2 * tmp

        tmp = 0.5 * Ts[market_pointer]
        alpha = d_M1 * tmp
        calp_M1 = np.cosh(alpha)
        salp_M1 = np.sinh(alpha)

        alpha = d_N1 * tmp
        calp_N1 = np.cosh(alpha)
        salp_N1 = np.sinh(alpha)

        alpha = d_M2 * tmp
        calp_M2 = np.cosh(alpha)
        salp_M2 = np.sinh(alpha)

        alpha = d_N2 * tmp
        calp_N2 = np.cosh(alpha)
        salp_N2 = np.sinh(alpha)

        A2_M1 = np.multiply(d_M1, calp_M1) + np.multiply(kes_M1, salp_M1)
        A2_N1 = np.multiply(d_N1, calp_N1) + np.multiply(kes_N1, salp_N1)
        A2_M2 = np.multiply(d_M2, calp_M2) + np.multiply(kes_M2, salp_M2)
        A2_N2 = np.multiply(d_N2, calp_N2) + np.multiply(kes_N2, salp_N2)

        A1_M1 = np.multiply(m_M1, salp_M1)
        A1_N1 = np.multiply(m_N1, salp_N1)
        A1_M2 = np.multiply(m_M2, salp_M2)
        A1_N2 = np.multiply(m_N2, salp_N2)

        A_M1 = np.divide(A1_M1, A2_M1)
        A_N1 = np.divide(A1_N1, A2_N1)
        A_M2 = np.divide(A1_M2, A2_M2)
        A_N2 = np.divide(A1_N2, A2_N2)

        tmp = 2 * heston_params.kappa * heston_params.theta / csqr
        halft = 0.5 * Ts[market_pointer]

        D_M1 = (
        np.log(d_M1)
        + (heston_params.kappa - d_M1) * halft
        - np.log(
        (d_M1 + kes_M1) * 0.5
        + (d_M1 - kes_M1)
        * 0.5
        * np.exp(-d_M1 * Ts[market_pointer])
        )
        )
        D_M2 = (
        np.log(d_M2)
        + (heston_params.kappa - d_M2) * halft
        - np.log(
        (d_M2 + kes_M2) * 0.5
        + (d_M1 - kes_M2)
        * 0.5
        * np.exp(-d_M2 * Ts[market_pointer])
        )
        )
        D_N1 = (
        np.log(d_N1)
        + (heston_params.kappa - d_N1) * halft
        - np.log(
        (d_N1 + kes_N1) * 0.5
        + (d_N1 - kes_N1)
        * 0.5
        * np.exp(-d_N1 * Ts[market_pointer])
        )
        )
        D_N2 = (
        np.log(d_N2)
        + (heston_params.kappa - d_N2) * halft
        - np.log(
        (d_N2 + kes_N2) * 0.5
        + (d_N2 - kes_N2)
        * 0.5
        * np.exp(-d_N2 * Ts[market_pointer])
        )
        )

        MNbas = _TagMn(
        np.real(
        h_M * np.exp(x0 * _imPQ_M - heston_params.v0 * A_M1 + tmp * D_M1) * g_M1
        ),
        np.real(
        h_N * np.exp(x0 * _imPQ_N - heston_params.v0 * A_N1 + tmp * D_N1) * g_N1
        ),
        np.real(
        h_M * np.exp(x0 * imPQ_M - heston_params.v0 * A_M2 + tmp * D_M2) * g_M2
        ),
        np.real(
        h_N * np.exp(x0 * imPQ_N - heston_params.v0 * A_N2 + tmp * D_N2) * g_N2
        ),
        )
        return MNbas

    def _jac_hes(
        self,
        heston_params: HestonParams,
        grid: StrikesMaturitiesGrid,
    ) -> nb.float64[:,:]:
        """Computes Jacobian w.r.t. HestonParams.

        **IMPORTANT**: *the order of rows is changed compared to the original
        implementation, so that they correspond to the new ordering of Heston
        parameters: v0, kappa, theta, eps, rho*

        Returns the array with shape (M, N_POINTS), where M = 5 (the number of
        Heston parameters, and N_POINTS is the number of market points).
        """
        # Собирает якобиан премий по параметрам: для каждой точки интегрирует
        # частные производные (через _hes_int_jac) квадратурой Гаусса-Лежандра.
        Ks = grid.Ks                                   # страйки
        Ts = grid.Ts                                   # сроки

        n = len(Ks)                                    # число точек
        r = heston_params.r                            # ставка

        da, db, dc, drho, dv0 = (
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
        )
        jacs = np.zeros((5, n), dtype=np.float64)      # якобиан 5×N (параметры × точки)
        for l in range(n):                              # по каждой точке…
            K = Ks[l]                                  #   страйк
            T = Ts[l]                                  #   срок
            discpi = np.exp(-r * T) / _PI              #   дисконт/π
            pa1, pa2, pb1, pb2, pc1, pc2, prho1, prho2, pv01, pv02 = (
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
            )
            jacint: _TagMNJac = self._hes_int_jac(heston_params, grid, l)  # интегранты производных в узлах
            pa1 += np.multiply(_W64, jacint.pa1s).sum()  # квадратура ∂P1/∂kappa
            pa2 += np.multiply(_W64, jacint.pa2s).sum()  # квадратура ∂P2/∂kappa

            pb1 += np.multiply(_W64, jacint.pb1s).sum()  # квадратура ∂P1/∂theta
            pb2 += np.multiply(_W64, jacint.pb2s).sum()  # квадратура ∂P2/∂theta

            pc1 += np.multiply(_W64, jacint.pc1s).sum()  # квадратура ∂P1/∂eps
            pc2 += np.multiply(_W64, jacint.pc2s).sum()  # квадратура ∂P2/∂eps

            prho1 += np.multiply(_W64, jacint.prho1s).sum()  # квадратура ∂P1/∂rho
            prho2 += np.multiply(_W64, jacint.prho2s).sum()  # квадратура ∂P2/∂rho

            pv01 += np.multiply(_W64, jacint.pv01s).sum()  # квадратура ∂P1/∂v0
            pv02 += np.multiply(_W64, jacint.pv02s).sum()  # квадратура ∂P2/∂v0

            # (initial) Variance (v0)                   # производная премии по v0:
            Qv1 = _Q * pv01
            Qv2 = _Q * pv02
            dv0 = discpi * (Qv1 - K * Qv2)
            jacs[0][l] = dv0

            # VarReversion (kappa)                       # производная премии по kappa:
            Qv1 = _Q * pa1
            Qv2 = _Q * pa2
            da = discpi * (Qv1 - K * Qv2)
            jacs[1][l] = da

            # AverageVar (theta)                         # производная премии по theta:
            Qv1 = _Q * pb1
            Qv2 = _Q * pb2
            db = discpi * (Qv1 - K * Qv2)
            jacs[2][l] = db

            # VolOfVar (eps)                             # производная премии по eps:
            Qv1 = _Q * pc1
            Qv2 = _Q * pc2
            dc = discpi * (Qv1 - K * Qv2)
            jacs[3][l] = dc

            # Correlation (rho)                          # производная премии по rho:
            Qv1 = _Q * prho1
            Qv2 = _Q * prho2
            drho = discpi * (Qv1 - K * Qv2)
            jacs[4][l] = drho

        return jacs                                    # якобиан 5×N
