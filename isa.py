import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

g = 9.8         # Ускорение свободного падения, м/с²
R = 287.058     # Газовая постоянная дял воздуха, Дж/(кг*К)

# Стандартные данные воздуха на уровне моря
T0 = 288.15     # К
p0 = 101325.0   # Па
rho0 = 1.225    # кг/м³

def temperature_gradient():
    T_1 = T0 + (-0.0065) * (11000 - 0)
    T_2 = T_1
    T_3 = T_2 + 0.001 * (32000 - 20000)
    T_4 = T_3 + 0.0028 * (47000 - 32000)
    T_5 = T_4
    T_6 = T_5 + (-0.0028) * (71000 - 51000)
    return T_1, T_2, T_3, T_4, T_5, T_6

def pressure_gradient():
    T_1, T_2, T_3, T_4, T_5, T_6 = temperature_gradient()

    p_1 = p0 * (T_1 / T0) ** (-g / (-0.0065 * R))
    p_2 = p_1 * np.exp(-(g / (R * T_2)) * (20000 - 11000))
    p_3 = p_2 * (T_3 / T_2) ** (-g / (0.001 * R))
    p_4 = p_3 * (T_4 / T_3) ** (-g / (0.0028 * R))
    p_5 = p_4 * np.exp(-(g / (R * T_5)) * (51000 - 47000))
    p_6 = p_5 * (T_6 / T_5) ** (-g / (-0.0028 * R))

    return p_1, p_2, p_3, p_4, p_5, p_6

def density_gradient():
    T_1, T_2, T_3, T_4, T_5, T_6 = temperature_gradient()

    rho_1 = rho0*(T_1/T0)**(-g/(-0.0065*R) - 1)
    rho_2 = rho_1 * np.exp(-(g/(R*T_2)) * (20000 - 11000))
    rho_3 = rho_2*(T_3/T_2)**(-g/(0.001*R) - 1)
    rho_4 = rho_3*(T_4/T_3)**(-g/(0.0028*R) - 1)
    rho_5 = rho_4 * np.exp(-(g/(R*T_5)) * (51000 - 47000))
    rho_6 = rho_5*(T_6/T_5)**(-g/(-0.0028*R) - 1)

    return rho_1, rho_2, rho_3, rho_4, rho_5, rho_6

def temperature(altitude):
    t = 0
    T_1, T_2, T_3, T_4, T_5, T_6 = temperature_gradient()

    if 0 <= altitude < 11000:
        t = T0 + -0.0065 * (altitude - 0)
    elif 11000 <= altitude < 20000:
        t = T_1
    elif 20000 <= altitude < 32000:
        t = T_2 + 0.001 * (altitude - 20000)
    elif 32000 <= altitude < 47000:
        t = T_3 + 0.0028 * (altitude - 32000)
    elif 47000 <= altitude < 51000:
        t = T_4
    elif 51000 <= altitude < 71000:
        t = T_5 + (-0.0028) * (altitude - 51000)
    elif 71000 <= altitude <= 85000:
        t = T_6 + (-0.002) * (altitude - 71000)
    elif 85000 < altitude <= 95000:
        t = 180
    elif altitude > 95000:
        t = 180 + (0.0032) * (altitude - 95000)

    return t

def pressure(altitude):
    p = 0

    T_1, T_2, T_3, T_4, T_5, T_6 = temperature_gradient()
    p_1, p_2, p_3, p_4, p_5, p_6 = pressure_gradient()

    if 0 <= altitude < 11000:
        p = p0*(temperature(altitude)/T0)**(-g/(-0.0065*R))
    elif 11000 <= altitude < 20000:
        p = p_1 * np.exp(-(g/(R*temperature(altitude))) * (altitude - 11000))
    elif 20000 <= altitude < 32000:
        p = p_2*(temperature(altitude)/T_2)**(-g/(0.001*R))
    elif 32000 <= altitude < 47000:
        p = p_3*(temperature(altitude)/T_3)**(-g/(0.0028*R))
    elif 47000 <= altitude < 51000:
        p = p_4 * np.exp(-(g/(R*temperature(altitude))) * (altitude - 47000))
    elif 51000 <= altitude < 71000:
        p = p_5*(temperature(altitude)/T_5)**(-g/(-0.0028*R))
    elif 71000 <= altitude <= 95000:
        p = p_6*(temperature(altitude)/T_6)**(-g/(-0.002*R))
    elif altitude > 95000:
        p = 0

    return p

def density(altitude):
    rho = 0

    T_1, T_2, T_3, T_4, T_5, T_6 = temperature_gradient()
    p_1, p_2, p_3, p_4, p_5, p_6 = pressure_gradient()
    rho_1, rho_2, rho_3, rho_4, rho_5, rho_6 = density_gradient()


    if 0 <= altitude < 11000:
        rho = rho0*(temperature(altitude)/T0)**(-g/(-0.0065*R) - 1)
    elif 11000 <= altitude < 20000:
        rho = rho_1 * np.exp(-(g/(R*temperature(altitude))) * (altitude - 11000))
    elif 20000 <= altitude < 32000:
        rho = rho_2*(temperature(altitude)/T_2)**(-g/(0.001*R) - 1)
    elif 32000 <= altitude < 47000:
        rho = rho_3*(temperature(altitude)/T_3)**(-g/(0.0028*R) - 1)
    elif 47000 <= altitude < 51000:
        rho = rho_4 * np.exp(-(g/(R*temperature(altitude))) * (altitude - 47000))
    elif 51000 <= altitude < 71000:
        rho = rho_5*(temperature(altitude)/T_5)**(-g/(-0.0028*R) - 1)
    elif 71000 <= altitude <= 95000:
        rho = rho_6*(temperature(altitude)/T_6)**(-g/(-0.002*R) - 1)
    elif altitude > 95000:
        rho = 0

    return rho

# def plot_graph(data, title, color, label):
#     plt.figure(figsize=(10, 8))
#     plt.plot(data, np.array(altitudes) / 1000, color + '-', linewidth=2.5)
#     plt.xlabel('{}'.format(label), fontsize=14)
#     plt.ylabel('Высота (км)', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.grid(True)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.legend([label], loc='best', fontsize=12)
#
# # Генерация данных
# altitudes = np.linspace(0, 140000, 1000)
# temperatures = [temperature(h) for h in altitudes]
# pressures = [pressure(h) for h in altitudes]
# densities = [density(h) for h in altitudes]
#
# plt.figure(0)
# plot_graph(temperatures, 'Температура в зависимости от высоты', 'b', 'Температура (К)')
#
# plt.figure(1)
# plot_graph(pressures, 'Давление в зависимости от высоты', 'r', 'Давление (Па)')
#
# plt.figure(2)
# plot_graph(densities, 'Плотность в зависимости от высоты', 'g', 'Плотность (кг/м³)')
#
# plt.show()
