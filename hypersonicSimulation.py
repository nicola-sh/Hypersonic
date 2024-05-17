import numpy as np
import pandas as pd
import plotly.graph_objects as go
from isa import density

# region Константы и параметры ЛА
# Константы
g = 9.8                 # Ускорение свободного падения, м/с²
Radius = 6371000        # Радиус Земли, м
a = 331                 # Скорость звука, м/с
pi = 3.141592653589793  # Число Пи

# Параметры ЛА
length = 12.5           # Длина ЛА, м
diameter = 0.5          # Диаметр ЛА, м
radius = diameter / 2   # Радиус ЛА, м

Ae = pi * radius ** 2   # Выходное сечение ЛА, м²
Am = Ae                 # Сечение миделя ЛА, м²
A0 = 0.54 * Ae          # Сечение воздухозаборника ЛА, м²
Afin = 0.2              # Площадь крыльев ЛА, м²

mha = 990               # Масса только ЛА, кг
mf = 450                # Начальная масса топлива, кг

# print("Сечение воздухозаборника ЛА: " + str(A0) + " м²")
# print("Сечение миделя ЛА: " + str(Am) + " м²")
# print("Выходное сечение ЛА: " + str(Ae) + " м²")
# endregion data


def simulation(t, t_end, dt, x, y, v, m, theta, alpha, engine, ballistic, glide, type3):
    data = {
        "x": [],
        "y": [],
        "v": [],
        "theta": [],
        "alpha": [],
        "m": [],

        "weight": [],
        "acceleration": [],

        "Thrust": [],
        "Drag": [],
        "Lift": [],

        "nxa": [],
        "nya": [],
        "costheta": [],

        "air_mass_flow_rate": [],
        "fuel_mass_flow_rate": [],
        "specificImpulse": [],

        "t": []
    }

    def air_mass_flow_rate(altitude, v):
        return density(altitude) * A0 * v

    def fuel_mass_flow_rate(altitude, v):
        # fuel_ratio = 3.5      # Стехеометрический коэффициент для керосин/кислород
        fuel_ratio = 14.7       # Стехеометрический коэффициент для керосин/воздух

        mf = air_mass_flow_rate(altitude, v) / fuel_ratio

        if mf > 3:
            mf = 3

        return mf

    def thrust(altitude, v, m, engine):

        thrust_ref = 140000         # Сила тяги, Н
        # g_fuel_ref = 5.05         # Расход топлива, кг/с   1кг/с = 30кН
        g_fuel_ref = 10.05          # Расход топлива, кг/с   1кг/с = 60кН

        mach_ratio = (v / a) / 7

        if m > mha and engine and v > 2.5 * a and altitude < 99000:
            thrust = ((fuel_mass_flow_rate(altitude, v) * thrust_ref) / g_fuel_ref) * mach_ratio
        else:
            thrust = 0

        return thrust

    def drag(altitude, v):
        if ballistic:
            drag_coefficient = 0
        else:
            drag_coefficient = 0.3
        return .5 * A0 * drag_coefficient * density(altitude) * v ** 2

    def lift(altitude, v):
        if ballistic:
            lift_coefficient = 0
        else:
            lift_coefficient = 0.3
        return .5 * Afin * lift_coefficient * density(altitude) * v ** 2

    def specific_impulse(y, v, m, engine):
        T = thrust(y, v, m, engine)
        g_fuel = fuel_mass_flow_rate(y, v)

        return T / g_fuel if T > 0 and g_fuel > 0 else 0

    def calculate_derivatives(t, x, y, v, theta, m, engine, alpha):
        P = thrust(y, v, m, engine)
        Drag = drag(y, v)
        Lift = lift(y, v)
        g_fuel = fuel_mass_flow_rate(y, v)

        dxdt = v * np.cos(theta) * Radius / (Radius + y)
        dydt = v * np.sin(theta)
        dvdt = (P * np.cos(alpha) - Drag - (m * g * np.sin(theta))) / m
        dthetadt = (P * np.sin(alpha) + Lift - m * g * np.cos(theta) +
                    (m * v ** 2 * np.cos(theta)) / (Radius + y)) / (m * v)
        dmdt = -g_fuel

        return dxdt, dydt, dvdt, dthetadt, dmdt

    def runge_kutta_step(t, dt, x, y, v, theta, m, engine, alpha):
        k1_x, k1_y, k1_v, k1_theta, k1_m = calculate_derivatives(t, x, y, v, theta, m, engine, alpha)
        k2_x, k2_y, k2_v, k2_theta, k2_m = calculate_derivatives(t + dt / 2,
                                                                 x + k1_x * dt / 2,
                                                                 y + k1_y * dt / 2,
                                                                 v + k1_v * dt / 2,
                                                                 theta + k1_theta * dt / 2,
                                                                 m + k1_m * dt / 2,
                                                                 engine, alpha)
        k3_x, k3_y, k3_v, k3_theta, k3_m = calculate_derivatives(t + dt / 2,
                                                                 x + k2_x * dt / 2,
                                                                 y + k2_y * dt / 2,
                                                                 v + k2_v * dt / 2,
                                                                 theta + k2_theta * dt / 2,
                                                                 m + k2_m * dt / 2,
                                                                 engine, alpha)
        k4_x, k4_y, k4_v, k4_theta, k4_m = calculate_derivatives(t + dt,
                                                                 x + k3_x * dt,
                                                                 y + k3_y * dt,
                                                                 v + k3_v * dt,
                                                                 theta + k3_theta * dt,
                                                                 m + k3_m * dt,
                                                                 engine, alpha)

        x = x + dt * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y = y + dt * (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        v = v + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        theta = theta + dt * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6

        if m > mha:
            m = m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
        else:
            m = mha

        return x, y, v, theta, m

    while y > 0 and t < t_end:

        #region Расчет других параметров
        weight = m * g
        acceleration = thrust(y, v, m, engine) / m

        # nxa = (thrust(y, v, m, engine) - drag(y, v)) / m * g
        # nya = (thrust(y, v, m, engine) + lift(y, v)) / m * g
        #
        # costheta = np.cos(theta)
        #endregion Расчет других параметров

        if ballistic:
            if t < 100:
                engine = True
            else:
                engine = False

        if glide:
            if 0.3 < t < 50:
                engine = True
            elif 170 < t and theta < 0:
                alpha = np.deg2rad(2)
                engine = True
            else:
                engine = False

        # if weight - drag(y, v) <= 0 and t > 50 and y < 25000:
        #     engine = True
        #     if 160 < t and np.degrees(theta) < 0:
        #         theta += np.degrees(0.01)

        x, y, v, theta, m = runge_kutta_step(t, dt, x, y, v, theta, m, engine, alpha)

        #region Добавляем данные каждой итерации в массив data
        data["x"].append(x)
        data["y"].append(y)
        data["v"].append(v)
        data["theta"].append(np.rad2deg(theta))
        data["alpha"].append(np.rad2deg(alpha))
        data["m"].append(m)
        data["weight"].append(weight)
        data["acceleration"].append(acceleration)
        data["Thrust"].append(thrust(y, v, m, engine))
        data["Drag"].append(drag(y, v))
        data["Lift"].append(lift(y, v))
        # data["nxa"].append(nxa)
        # data["nya"].append(nya)
        # data["costheta"].append(costheta)
        data["specificImpulse"].append(specific_impulse(y, v, m, engine))
        data["air_mass_flow_rate"].append(air_mass_flow_rate(y, v))
        data["fuel_mass_flow_rate"].append(fuel_mass_flow_rate(y, v))
        data["t"].append(t)
        #endregion

        t += dt  # Обновляем время

    last_x = data["x"][-1]
    print("Максимальная дальность данной симуляции:", last_x)

    return data


def plot_flight_data(data):
    fig = go.Figure()

    # Определение параметров для добавления на график
    traces = [
        ("t", "x", "Положение ЛА по оси X, м от времени"),
        ("t", "y", "Положение ЛА по оси Y, м от времени"),
        ("t", "v", "Скорость ЛА, м/с от времени"),
        ("t", "m", "Масса ЛА, кг от времени"),
        ("t", "weight", "Ускорение, м/с² от времени"),
        ("t", "acceleration", "Вес ЛА, Н от времени"),
        ("t", "theta", "Угол наклона траектории, градус от времени"),
        ("t", "alpha", "Угол атаки, градус от времени"),
        ("t", "Thrust", "Сила тяги, Н от времени"),
        ("t", "Drag", "Сила сопротивления, Н от времени"),
        ("t", "Lift", "Подъемная сила, Н от времени"),
        ("t", "air_mass_flow_rate", "Массовый расход воздуха ПВРД, кг/с от времени"),
        ("t", "fuel_mass_flow_rate", "Массовый расход топлива ПВРД, кг/с от времени"),
        ("t", "specificImpulse", "Удельный импульс, от времени"),
        # ("t", "nxa", "nxa"),
        # ("t", "nya", "nya"),
        # ("t", "costheta", "costheta"),
        ("x", "y", "Траектория полета ЛА")
    ]

    # Добавление данных на график с использованием цикла
    for x_key, y_key, name in traces:
        fig.add_trace(go.Scatter(x=data[x_key], y=data[y_key], mode='lines',
                                 name=name, line=dict(width=3),
                                 hovertemplate="Время: %{x}<br>Значение: %{y}"))

    # Настройка макета графика
    fig.update_layout(
        title="График данных полета",
        xaxis=dict(
            title="Время, с",
            tickfont=dict(size=18, family='Times New Roman'),
            title_font=dict(size=18, family='Times New Roman')
        ),
        yaxis=dict(
            title="Значение",
            tickfont=dict(size=18, family='Times New Roman'),
            title_font=dict(size=18, family='Times New Roman')
        ),
        height=1000,
        width=1700,
        showlegend=True,
        legend=dict(font=dict(size=18, family='Times New Roman'))
    )

    fig.show()


thetas = np.radians(np.arange(0, 55, 5))  # Угол тета от 30 до 55 градусов с шагом 5 градусов
alphas = np.radians(np.arange(0, 1, 1))  # Угол альфа от -5 до 5 градусов с шагом 5 градусов
max_x = float('-inf')
best_simulation = None
all_simulations_data = []

for theta in thetas:
    for alpha in alphas:
        simulation_data = simulation(0.0, 999, 0.01, 0, 2000, 3 * a, 1440, theta, alpha, False, True, False, False)
        last_x = simulation_data["x"][-1]  # Получение последнего значения x из данных симуляции
        simulation_info = {"Theta": np.rad2deg(theta), "Alpha": np.rad2deg(alpha), "Last_X": last_x}
        all_simulations_data.append(simulation_info)

        if last_x > max_x:
            max_x = last_x
            best_simulation = simulation_data #содержит информацию о симуляции с самой дальней траектории

# Создание DataFrame из списка данных всех симуляций
df = pd.DataFrame(all_simulations_data)
print(df)

plot_flight_data(best_simulation)  # Построение графика для лучшей симуляции

# ballistic = simulation(0.0, 999, 0.01, 0, 2000, 3 * a, 1440, np.radians(47), np.radians(0), False, True, False, False)
# plot_flight_data(ballistic)

# glide = simulation(0.0, 999, 0.01, 0, 2000, 3 * a, 1440, np.radians(30), np.radians(0), False, False, True, False)
# plot_flight_data(glide)

# type3 = simulation(0.0, 999, 0.01, 0, 2000, 3 * a, 1440, np.radians(30), np.radians(0), False, False, False, True)
# plot_flight_data(glide)
