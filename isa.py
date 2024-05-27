import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Стандартные данные воздуха на уровне моря
T0 = 288.15     # [K]
p0 = 101325.0   # [Pa]
rho0 = 1.225    # [kg/m3]

# Standard acceleration due to gravity:
g = 9.80665     # [kg*m/s2]

# Specific gas constant for air:
R = 287.058     # [J/(kg*K)]

# Lapse rates and atmospheric zones altitudes:
# Тропосфера (0-11)km
h_ts = 0        # [m]
a_ts = -0.0065  # [K/m]
# Тропопауза (11-20)km
h_tp = 11000    # [m]
a_tp = 0        # [K/m] (isothermal)
# Стратосфера (20-32)km
h_ss1 = 20000   # [m]
a_ss1 = 0.001   # [K/m]
# (32-47)km
h_ss2 = 32000   # [m]
a_ss2 = 0.0028  # [K/m]
# Стратопауза  (47-51)km
h_sp = 47000    # [m]
a_sp = 0        # [K/m] (isothermal)
# Мезосфера (51-71)km
h_ms1 = 51000   # [m]
a_ms1 = -0.0028 # [K/m]
#  (71-85)km
h_ms2 = 71000   # [m]
a_ms2 = -0.002  # [K/m]
# ============
h_fin = 150000   # [m]


def temperature(altitude):
    t = 0

    T_1 = T0 + a_ts * (h_tp - h_ts)
    T_2 = T_1
    T_3 = T_2 + a_ss1 * (h_ss2 - h_ss1)
    T_4 = T_3 + a_ss2 * (h_sp - h_ss2)
    T_5 = T_4
    T_6 = T_5 + a_ms1 * (h_ms2 - h_ms1)

    if altitude >= h_ts and altitude < h_tp:
        t = T0 + a_ts * (altitude - h_ts)
    elif altitude >= h_tp and altitude < h_ss1:
        t = T_1
    elif altitude >= h_ss1 and altitude < h_ss2:
        t = T_2 + a_ss1 * (altitude - h_ss1)
    elif altitude >= h_ss2 and altitude < h_sp:
        t = T_3 + a_ss2 * (altitude - h_ss2)
    elif altitude >= h_sp and altitude < h_ms1:
        t = T_4
    elif altitude >= h_ms1 and altitude < h_ms2:
        t = T_5 + a_ms1 * (altitude - h_ms1)
    elif altitude >= h_ms2 and altitude <= h_fin:
        t = T_6 + a_ms2 * (altitude - h_ms2)

    return t


def pressure(altitude):
    p = 0

    T_1 = T0 + a_ts * (h_tp - h_ts)
    T_2 = T_1
    T_3 = T_2 + a_ss1 * (h_ss2 - h_ss1)
    T_4 = T_3 + a_ss2 * (h_sp - h_ss2)
    T_5 = T_4
    T_6 = T_5 + a_ms1 * (h_ms2 - h_ms1)

    p_1 = p0*(T_1/T0)**(-g/(a_ts*R))
    p_2 = p_1 * np.exp(-(g/(R*T_2)) * (h_ss1 - h_tp))
    p_3 = p_2*(T_3/T_2)**(-g/(a_ss1*R))
    p_4 = p_3*(T_4/T_3)**(-g/(a_ss2*R))
    p_5 = p_4 * np.exp(-(g/(R*T_5)) * (h_ms1 - h_sp))
    p_6 = p_5*(T_6/T_5)**(-g/(a_ms1*R))

    if altitude >= h_ts and altitude < h_tp:
        p = p0*(temperature(altitude)/T0)**(-g/(a_ts*R))
    elif altitude >= h_tp and altitude < h_ss1:
        p = p_1 * np.exp(-(g/(R*temperature(altitude))) * (altitude - h_tp))
    elif altitude >= h_ss1 and altitude < h_ss2:
        p = p_2*(temperature(altitude)/T_2)**(-g/(a_ss1*R))
    elif altitude >= h_ss2 and altitude < h_sp:
        p = p_3*(temperature(altitude)/T_3)**(-g/(a_ss2*R))
    elif altitude >= h_sp and altitude < h_ms1:
        p = p_4 * np.exp(-(g/(R*temperature(altitude))) * (altitude - h_sp))
    elif altitude >= h_ms1 and altitude < h_ms2:
        p = p_5*(temperature(altitude)/T_5)**(-g/(a_ms1*R))
    elif altitude >= h_ms2 and altitude <= h_fin:
        p = p_6*(temperature(altitude)/T_6)**(-g/(a_ms2*R))

    return p


def density(altitude):
    rho = 0

    T_1 = T0 + a_ts * (h_tp - h_ts)
    T_2 = T_1
    T_3 = T_2 + a_ss1 * (h_ss2 - h_ss1)
    T_4 = T_3 + a_ss2 * (h_sp - h_ss2)
    T_5 = T_4
    T_6 = T_5 + a_ms1 * (h_ms2 - h_ms1)

    rho_1 = rho0*(T_1/T0)**(-g/(a_ts*R) - 1)
    rho_2 = rho_1 * np.exp(-(g/(R*T_2)) * (h_ss1 - h_tp))
    rho_3 = rho_2*(T_3/T_2)**(-g/(a_ss1*R) - 1)
    rho_4 = rho_3*(T_4/T_3)**(-g/(a_ss2*R) - 1)
    rho_5 = rho_4 * np.exp(-(g/(R*T_5)) * (h_ms1 - h_sp))
    rho_6 = rho_5*(T_6/T_5)**(-g/(a_ms1*R) - 1)


    if altitude >= h_ts and altitude < h_tp:
        rho = rho0*(temperature(altitude)/T0)**(-g/(a_ts*R) - 1)
    elif altitude >= h_tp and altitude < h_ss1:
        rho = rho_1 * np.exp(-(g/(R*temperature(altitude))) * (altitude - h_tp))
    elif altitude >= h_ss1 and altitude < h_ss2:
        rho = rho_2*(temperature(altitude)/T_2)**(-g/(a_ss1*R) - 1)
    elif altitude >= h_ss2 and altitude < h_sp:
        rho = rho_3*(temperature(altitude)/T_3)**(-g/(a_ss2*R) - 1)
    elif altitude >= h_sp and altitude < h_ms1:
        rho = rho_4 * np.exp(-(g/(R*temperature(altitude))) * (altitude - h_sp))
    elif altitude >= h_ms1 and altitude < h_ms2:
        rho = rho_5*(temperature(altitude)/T_5)**(-g/(a_ms1*R) - 1)
    elif altitude >= h_ms2 and altitude <= h_fin:
        rho = rho_6*(temperature(altitude)/T_6)**(-g/(a_ms2*R) - 1)

    return rho


# # Генерация данных для графиков
# altitudes = np.linspace(0, 90000, 500)
# temperatures = [temperature(h) for h in altitudes]
# pressures = [pressure(h) for h in altitudes]
# densities = [density(h) for h in altitudes]
#
# # Создание графика согласно ГОСТам
# plt.figure(figsize=(8, 6))  # Установка размеров графика
#
# # Построение графика температуры
# plt.plot(temperatures, altitudes / 1000, 'b-', linewidth=2)  # Использование синего цвета с толщиной линии 2
# plt.xlabel('Температура (K)', fontsize=12)  # Установка подписи оси x с размером шрифта 12
# plt.ylabel('Высота (км)', fontsize=12)  # Установка подписи оси y с размером шрифта 12
# plt.title('Температура в зависимости от высоты', fontsize=14)  # Установка заголовка с размером шрифта 14
# plt.grid(True)  # Включение сетки на графике
# plt.xticks(fontsize=10)  # Установка размера шрифта для делений на оси x
# plt.yticks(fontsize=10)  # Установка размера шрифта для делений на оси y
# plt.legend(['Температура'], loc='best', fontsize=10)  # Добавление легенды с размером шрифта 10 и лучшим положением
#
# plt.show()
#
#
# # Создание графика согласно ГОСТам
# plt.figure(figsize=(8, 6))  # Установка размеров графика
#
# # Построение графика давления
# plt.plot(pressures, altitudes / 1000, 'r-', linewidth=2)  # Использование красного цвета с толщиной линии 2
# plt.xlabel('Давление (Па)', fontsize=12)  # Установка подписи оси x с размером шрифта 12
# plt.ylabel('Высота (км)', fontsize=12)  # Установка подписи оси y с размером шрифта 12
# plt.title('Давление в зависимости от высоты', fontsize=14)  # Установка заголовка с размером шрифта 14
# plt.grid(True)  # Включение сетки на графике
# plt.xticks(fontsize=10)  # Установка размера шрифта для делений на оси x
# plt.yticks(fontsize=10)  # Установка размера шрифта для делений на оси y
# plt.legend(['Давление'], loc='best', fontsize=10)  # Добавление легенды с размером шрифта 10 и лучшим положением
#
# plt.show()
#
# # Создание графика согласно ГОСТам
# plt.figure(figsize=(8, 6))  # Установка размеров графика
#
# # Построение графика плотности
# plt.plot(densities, altitudes / 1000, 'g-', linewidth=2)  # Использование зеленого цвета с толщиной линии 2
# plt.xlabel('Плотность (кг/м^3)', fontsize=12)  # Установка подписи оси x с размером шрифта 12
# plt.ylabel('Высота (км)', fontsize=12)  # Установка подписи оси y с размером шрифта 12
# plt.title('Плотность в зависимости от высоты', fontsize=14)  # Установка заголовка с размером шрифта 14
# plt.grid(True)  # Включение сетки на графике
# plt.xticks(fontsize=10)  # Установка размера шрифта для делений на оси x
# plt.yticks(fontsize=10)  # Установка размера шрифта для делений на оси y
# plt.legend(['Плотность'], loc='best', fontsize=10)  # Добавление легенды с размером шрифта 10 и лучшим положением
#
# plt.show()
