"""deribit_vol_surface: загрузка реальной поверхности волатильности Deribit из CSV.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        корневая утилита (демонстрационный/тестовый хелпер pyquant).
НАЗНАЧЕНИЕ:  прочитать CSV-снимок цепочки опционов Deribit (мульти-индексный
             заголовок «тип/страйк») и собрать из него VolSurfaceChainSpace —
             поверхность волатильности в страйк-пространстве. Используется как
             пример/проверка движка pyquant на рыночных данных.
ИМПОРТИРУЕТ: pandas; всё из pyquant.common и pyquant.vol_surface.
ЭКСПОРТИРУЕТ: get_vol_surface(test_file) -> VolSurfaceChainSpace.
КОНСТАНТЫ:   YEAR_NANOS (из pyquant.common) — наносекунд в году (для перевода в годы).
=============================================================================
"""

import pandas as pd                                      # чтение CSV и работа с таблицей
from pyquant.common import *                             # Spot, ForwardRates, кривые, YEAR_NANOS
from pyquant.vol_surface import *                        # VolSurfaceChainSpace и сопутствующие

def get_vol_surface(test_file):
    df = pd.read_csv(test_file, header=[0,1], index_col=0)  # CSV с двухуровневым заголовком (тип, страйк)
    
    Ts = pd.DatetimeIndex(df.index).astype(int).values.squeeze()  # индекс времени → наносекунды (int)
    Ts = (Ts - Ts[0]) / YEAR_NANOS                       # перевести в годы от первого момента
    
    types_str, strikes_str = zip(*df.columns[2:].values)  # разобрать колонки (пропустив swap/futures) на тип и страйк
    Ks = np.array([float(x) for x in strikes_str])       # страйки как числа
    option_types = np.array([True if x == 'call' else False for x in types_str])  # call=True, put=False
    
    spot = Spot(df['swap'].values[0].item())             # спот = цена swap из первой строки
    df.iloc[:, 2:] *= spot.S                             # перевести премии из единиц базового в денежные (×спот)
    
    Fs = df['futures'].values.squeeze()                  # фьючерсные (форвардные) цены по срокам
    Fidx = np.nonzero(Fs)[0]                             # индексы непустых фьючерсов
    fwd_curve = forward_curve_from_forward_rates(spot, ForwardRates(Fs[Fidx]), TimesToMaturity(Ts[Fidx]))  # форвардная кривая
    
    n_T = len(Ts) - 1                                    # число сроков (без нулевого момента)
    buf_T = Ts[1:].repeat(len(Ks))                       # развернуть сроки под каждый страйк
    buf_K = np.tile(Ks, n_T)                             # повторить страйки для каждого срока
    buf_C = np.tile(option_types, n_T)                   # повторить типы опционов для каждого срока
    buf_pv = df.values[1:,2:].flatten()                  # все премии в один вектор
    pv_idx = np.nonzero(buf_pv)                          # оставить только ненулевые котировки

    return VolSurfaceChainSpace(                          # собрать поверхность волатильности из отфильтрованных данных
        fwd_curve, 
        TimesToMaturity(buf_T[pv_idx]),
        Strikes(buf_K[pv_idx]),
        OptionTypes(buf_C[pv_idx]),
        Premiums(buf_pv[pv_idx])
    )