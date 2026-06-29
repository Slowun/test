"""pyquant.utils: численные примитивы (нормальное распределение, сплайны, поиск).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        pyquant (низкоуровневая количественная библиотека, numba).
НАЗНАЧЕНИЕ:  базовые numba-функции и структуры, на которых строится остальная
             библиотека: аппроксимация нормального CDF/PDF, клиппинг, проверка
             сортировки, бинарный поиск, кубический и PCHIP-сплайны (для кривых
             ставок и поверхностей волатильности).
ИМПОРТИРУЕТ: numpy, numba, enum.Enum.
КОНСТАНТЫ:   YEAR_NANOS — наносекунд в году (365 дней).
ЭКСПОРТИРУЕТ: InstrumentId, InstrumentType, normal_cdf(_vec), normal_pdf,
             np_clip, is_sorted, mass_weights, searchsorted, XAxis, YAxis,
             CubicSpline1D, PchipSpline1D.
КЕМ ИСПОЛЬЗУЕТСЯ: pyquant.common (через `from .utils import *`) и весь pyquant.
=============================================================================
"""

import numpy as np                                       # численные массивы
import numba as nb                                       # JIT-компиляция
from enum import Enum                                    # перечисления инструментов

# One year in nanoseconds (365 * 24 * 3600 * 1_000_000_000)
YEAR_NANOS: int = 31536000000000000                      # наносекунд в году (для перевода времени)


class InstrumentId(Enum):                                # идентификаторы базовых активов
    BTC = 0
    ETH = 1

class InstrumentType(Enum):                              # типы инструментов (фьючерс/опцион/своп/…)
    FUTURE = 1
    OPTION = 2
    FUTURE_COMBO = 3
    OPTION_COMBO = 4
    CALL_OPTION = 5
    PUT_OPTION = 6
    SWAP = 7


@nb.njit(nb.float64(nb.float64))
def normal_cdf(x: nb.float64) -> nb.float64:             # аппроксимация функции стандартного нормального CDF (Abramowitz-Stegun)
    t = 1 / (1 + 0.2316419 * np.absolute(x))            # вспомогательная переменная
    summ = (                                            # полиномиальная аппроксимация хвоста:
        0.319381530 * t
        - 0.356563782 * t**2
        + 1.781477937 * t**3
        - 1.821255978 * t**4
        + 1.330274429 * t**5
    )
    if x >= 0:                                          # для x ≥ 0…
        return 1 - summ * np.exp(-np.absolute(x) ** 2 / 2) / np.sqrt(2 * np.pi)
    else:                                              # для x < 0 (симметрия)…
        return summ * np.exp(-np.absolute(x) ** 2 / 2) / np.sqrt(2 * np.pi)

@nb.vectorize([nb.float64(nb.float64)], nopython=True)
def normal_cdf_vec(x: nb.float64[:]) -> nb.float64[:]:  # векторная версия normal_cdf
    return normal_cdf(x)

@nb.njit()
def normal_pdf(x: nb.float64) -> nb.float64:            # плотность стандартного нормального распределения
    probability = 1.0 / np.sqrt(2 * np.pi)             # нормировочный множитель
    probability *= np.exp(-0.5 * x**2)                 # экспоненциальное ядро
    return probability


@nb.njit()
def np_clip(a: nb.float64, a_min: nb.float64, a_max: nb.float64) -> nb.float64:  # скалярный clip (njit-совместимый)
    if a < a_min:                                      # ниже минимума…
        out = a_min
    elif a > a_max:                                    # выше максимума…
        out = a_max
    else:                                              # внутри коридора…
        out = a
    return out 

@nb.njit
def is_sorted(a: nb.float64[:]) -> nb.boolean:          # проверка, что массив неубывающий
    for i in range(a.size-1):
         if a[i+1] < a[i] :                            # нашли нарушение порядка…
               return False
    return True

@nb.njit
def mass_weights(t: nb.float64, Ts: nb.float64[:], tol: nb.float64 = 1e-6) -> nb.float64[:]:  # веса интерполяции по сетке сроков
    n = len(Ts)                                        # число узлов
    w = np.zeros_like(Ts)                              # веса
    flag = False                                       # признак найденного интервала
    for i in range(n):                                 # ищем ближайшие узлы к t:
        wi = t - Ts[i]                                 #   расстояние до узла
        flag = t - Ts[i] <= 0.                         #   узел не раньше t
        if flag:
            if abs(wi) <= tol:                         #   попали точно в узел…
                w[i] = 1.
                break
            w[i] = 1/(abs(wi) + 1e-12)                 #   вес ~ обратное расстояние
            if i-1 >= 0:                               #   и для предыдущего узла…
                w[i-1] = 1/(abs(t - Ts[i-1]) + 1e-12)
            break
    if np.all(w<=0):                                   # t правее всех узлов…
        w[-1] = 1.                                     #   берём последний узел
    return w / w.sum()                                 # нормированные веса


@nb.njit(cache = True, fastmath = True)
def searchsorted(a: nb.float64[:], b: nb.float64) -> nb.int64:  # индекс вставки b в отсортированный a
    idx = 0                                            # результат
    pa, pb = 0, 0                                      # указатели
    while pb < 1:                                      # одно «срабатывание»:
        if pa < len(a) and a[pa] < b:                 #   элемент меньше b…
            pa += 1                                    #     сдвигаемся
        else:                                          #   нашли позицию…
            idx = pa
            pb += 1
    return idx


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class XAxis:                                            # ось X сплайна (узлы)
    def __init__(self, x: nb.float64[:]):
        self.data = x


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class YAxis:                                            # ось Y сплайна (значения)
    def __init__(self, y: nb.float64[:]):
        self.data = y


@nb.experimental.jitclass([
    ("_x0", nb.float64[:]),
    ("_a", nb.float64[:]),
    ("_b", nb.float64[:]),
    ("_c", nb.float64[:]),
    ("_d", nb.float64[:]),
])
class CubicSpline1D:                                    # натуральный кубический сплайн 1D
    def __init__(self, x: XAxis, y: YAxis):
        self._x0 = x.data                              # узлы
        self._calc_spline_params(x.data, y.data)       # вычисляем коэффициенты сплайна

    def _calc_spline_params(self, x: nb.float64[:], y: nb.float64[:]):  # решение трёхдиагональной системы
        n = x.size - 1                                 # число интервалов
        a = y.copy()                                   # коэффициенты a (значения)
        h = x[1:] - x[:-1]                             # длины интервалов
        alpha = 3 * ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1])  # правая часть
        c = np.zeros(n+1)                              # коэффициенты c
        ell, mu, z = np.ones(n+1), np.zeros(n), np.zeros(n+1)  # рабочие массивы прогонки
        for i in range(1, n):                          # прямой ход прогонки:
            ell[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / ell[i]
            z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / ell[i]
        for i in range(n-1, -1, -1):                   # обратный ход:
            c[i] = z[i] - mu[i] * c[i+1]
        b = (a[1:] - a[:-1]) / h + (c[:-1] + 2 * c[1:]) * h / 3  # коэффициенты b
        d = np.diff(c) / (3 * h)                        # коэффициенты d

        self._a = a[1:]                                # сохраняем коэффициенты
        self._b = b
        self._c = c[1:]
        self._d = d

    def _func_spline(self, x: nb.float64, ix: nb.int64) -> nb.float64:  # значение полинома на интервале ix
        dx = x - self._x0[1:][ix]                      # смещение от узла
        return self._a[ix] + (self._b[ix] + (self._c[ix] + self._d[ix] * dx) * dx) * dx  # схема Горнера

    def apply(self, x: nb.float64) -> nb.float64:       # интерполяция в точке x
        ix = searchsorted(self._x0[1 : -1], x)         # индекс интервала
        return self._func_spline(x, ix)


@nb.experimental.jitclass([
    ("_x0", nb.float64[:]),
    ("_a", nb.float64[:]),
    ("_b", nb.float64[:]),
    ("_c", nb.float64[:]),
    ("_d", nb.float64[:]),
])
class PchipSpline1D:                                    # монотонный сплайн PCHIP (без переколебаний)
    def __init__(self, x: XAxis, y: YAxis):
        self._x0 = x.data                              # узлы
        self._calc_pchip_params(x.data, y.data)        # коэффициенты PCHIP

    def _calc_pchip_params(self, x: nb.float64[:], y: nb.float64[:]):  # вычисление кубических коэффициентов
        n = x.size - 1                                 # число интервалов
        
        # Calculate differences                         # разности узлов и наклонов:
        h = x[1:] - x[:-1]                             # длины интервалов
        delta = (y[1:] - y[:-1]) / h                   # секущие наклоны
        
        # Calculate slopes using PCHIP algorithm        # наклоны в узлах (монотонные):
        slopes = self._pchip_slopes(x, y, delta)
        
        # Calculate cubic coefficients                  # кубические коэффициенты a,b,c,d:
        a = y[:-1].copy()
        b = slopes[:-1].copy()
        c = (3 * delta - 2 * slopes[:-1] - slopes[1:]) / h
        d = (slopes[:-1] + slopes[1:] - 2 * delta) / (h * h)
        
        self._a = a                                    # сохраняем коэффициенты
        self._b = b
        self._c = c
        self._d = d

    def _pchip_slopes(self, x: nb.float64[:], y: nb.float64[:], delta: nb.float64[:]) -> nb.float64[:]:
        """Calculate PCHIP slopes ensuring monotonicity."""
        n = x.size                                     # число узлов
        slopes = np.zeros(n)                           # наклоны
        
        # For interior points, use PCHIP slope selection  # внутренние точки:
        for i in range(1, n - 1):
            h1 = x[i] - x[i-1]                         #   левый интервал
            h2 = x[i+1] - x[i]                         #   правый интервал
            d1 = delta[i-1]                            #   левый наклон
            d2 = delta[i]                              #   правый наклон
            
            # PCHIP slope selection                     #   выбор наклона по PCHIP:
            if d1 * d2 <= 0:                            #     смена знака → плато (наклон 0)
                slopes[i] = 0.0
            else:                                      #     взвешенное гармоническое среднее
                w1 = 2 * h2 + h1
                w2 = 2 * h1 + h2
                slopes[i] = (w1 + w2) / (w1 / d1 + w2 / d2)
        
        # Handle endpoints                              # обработка концов:
        if n > 1:
            # Left endpoint                              #   левый край:
            if n == 2:
                slopes[0] = delta[0]
            else:
                h1 = x[1] - x[0]
                h2 = x[2] - x[1]
                d1 = delta[0]
                d2 = delta[1]
                
                if d1 * d2 <= 0:
                    slopes[0] = 0.0
                else:
                    slopes[0] = (3 * h1 + 2 * h2) / (h1 / d1 + 2 * h2 / d2)
            
            # Right endpoint                             #   правый край:
            if n == 2:
                slopes[n-1] = delta[n-2]
            else:
                h1 = x[n-1] - x[n-2]
                h2 = x[n-2] - x[n-3]
                d1 = delta[n-2]
                d2 = delta[n-3]
                
                if d1 * d2 <= 0:
                    slopes[n-1] = 0.0
                else:
                    slopes[n-1] = (3 * h1 + 2 * h2) / (h1 / d1 + 2 * h2 / d2)
        
        return slopes                                  # наклоны в узлах

    def _func_pchip(self, x: nb.float64, ix: nb.int64) -> nb.float64:  # значение полинома на интервале ix
        dx = x - self._x0[ix]                          # смещение от узла
        return self._a[ix] + (self._b[ix] + (self._c[ix] + self._d[ix] * dx) * dx) * dx  # схема Горнера
    
    def apply(self, x: nb.float64) -> nb.float64:       # интерполяция в точке x
        ix = searchsorted(self._x0[1 : -1], x)         # индекс интервала
        return self._func_pchip(x, ix)
