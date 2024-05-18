import numpy as np
import pandas as pd
import plotly.graph_objects as go
from isa import density
from datetime import datetime
import time
from multiprocessing import Pool

# region Константы
g = 9.8                         # Ускорение свободного падения, м/с²
Radius = 6371000                # Радиус Земли, м
a = 331                         # Скорость звука, м/с
pi = 3.141592653589793          # Число Пи
# endregion

def simulation(t, t_end, dt,
               x, y, v, theta, alpha, engine,
               m, mha, mf,
               A0, Am, Ae, a_ratio, Afin,
               theta_for_eng_true, theta_for_eng_off, engine_time,
               ballistic, glide, type3):

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

        "air_mass_flow_rate": [],
        "fuel_mass_flow_rate": [],
        "air_fuel_ratio": [],

        "specificImpulse": [],

        "t": []
    }

    def air_mass_flow_rate(altitude, v):
        return density(altitude) * A0 * v

    def fuel_mass_flow_rate(altitude, v):
        fuel_ratio = 14.7       # Стехеометрический коэффициент для керосин/воздух

        mf = air_mass_flow_rate(altitude, v) / fuel_ratio

        return mf

    def thrust(altitude, v, m, engine):
        thrust_ref = 22000              # Сила тяги, Н
        mach_ratio = (v / a) / 12
        if m > mha and engine and v > 2.5 * a:
            thrust = (fuel_mass_flow_rate(altitude, v) * thrust_ref) * mach_ratio
        else:
            thrust = 0

        return thrust

    def drag(altitude, v):
        drag_coefficient = 0.3
        return .5 * A0 * drag_coefficient * density(altitude) * v ** 2

    def lift(altitude, v):
        lift_coefficient = 0.4
        return .5 * Afin * lift_coefficient * density(altitude) * v ** 2

    def specific_impulse(y, v, m, engine):
        T = thrust(y, v, m, engine)
        g_fuel = fuel_mass_flow_rate(y, v) * g

        return T / g_fuel if T > 0 and g_fuel > 0 else 0

    def air_fuel_ratio(y, v):
        air_mass_flow = air_mass_flow_rate(y, v)
        fuel_mass_flow = fuel_mass_flow_rate(y, v)

        if fuel_mass_flow != 0:
            af = air_mass_flow / fuel_mass_flow
            return af
        else:
            return None  # Или какое-то другое значение по умолчанию, которое имеет смысл в вашем контексте

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

    engine_duration = 0  # Переменная для отслеживания времени работы двигателя

    while y > 0 and t < t_end:

        weight = m * g
        acceleration = thrust(y, v, m, engine) / m

        if y < 60000 and v < 12 * a:
            if 0 < t < 20:
                engine = True

            if theta < np.deg2rad(theta_for_eng_true):
                alpha = np.deg2rad(6)
                engine = True
                engine_duration += dt
            elif theta > np.deg2rad(theta_for_eng_off):
                alpha = np.deg2rad(0)

            # Проверяем, достигнуто ли требуемое время работы двигателя
            if engine_duration >= engine_time:
                engine = False
                engine_duration = 0
        else:
            engine = False
            engine_duration = 0

        # if weight - drag(y, v) <= 0 and t > 50 and y < 25000:
        #     engine = True
        #     if 160 < t and np.degrees(theta) < 0:
        #         theta += np.degrees(0.01)

        x, y, v, theta, m = runge_kutta_step(t, dt, x, y, v, theta, m, engine, alpha)

        #region Добавляем данные каждой итерации в массив data
        data["x"].append(x)
        data["y"].append(y)
        data["v"].append(v)
        data["theta"].append(np.rad2deg(theta))  # Угол наклона траектории
        data["alpha"].append(np.rad2deg(alpha))  # Угол атаки
        data["m"].append(m)
        data["weight"].append(weight)
        data["acceleration"].append(acceleration)
        data["Thrust"].append(thrust(y, v, m, engine))
        data["Drag"].append(drag(y, v))
        data["Lift"].append(lift(y, v))
        data["specificImpulse"].append(specific_impulse(y, v, m, engine))
        data["air_mass_flow_rate"].append(air_mass_flow_rate(y, v))
        data["fuel_mass_flow_rate"].append(fuel_mass_flow_rate(y, v))
        data["air_fuel_ratio"].append(air_fuel_ratio(y, v))

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
        ("t", "weight", "Вес ЛА, Н от времени"),
        ("t", "acceleration", "Ускорение, м/с² от времени"),
        ("t", "theta", "Угол наклона траектории, градус от времени"),
        ("t", "alpha", "Угол атаки, градус от времени"),
        ("t", "Thrust", "Сила тяги, Н от времени"),
        ("t", "Drag", "Сила сопротивления, Н от времени"),
        ("t", "Lift", "Подъемная сила, Н от времени"),
        ("t", "air_mass_flow_rate", "Массовый расход воздуха ПВРД, кг/с от времени"),
        ("t", "fuel_mass_flow_rate", "Массовый расход топлива ПВРД, кг/с от времени"),
        ("t", "air_fuel_ratio", "Соотношение расхода воздуха к топливу"),
        ("t", "specificImpulse", "Удельный импульс, от времени"),
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
            title_font=dict(size=18, family='Times New Roman'),
            tickformat = "f"  # Настройка формата чисел на оси
    ),
        yaxis=dict(
            title="Значение",
            tickfont=dict(size=18, family='Times New Roman'),
            title_font=dict(size=18, family='Times New Roman'),
            tickformat = "f"  # Настройка формата чисел на оси
        ),
        height=1000,
        width=1700,
        showlegend=True,
        legend=dict(font=dict(size=18, family='Times New Roman'))
    )

    fig.show()

def save_data_to_excel(data, filename=None):
    # Если имя файла не указано, используйте текущее время
    if filename is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"data_{current_time}.xlsx"

    # Создание DataFrame из данных
    df = pd.DataFrame(data)

    # Сохранение DataFrame в файл Excel
    df.to_excel(filename, index=False)

    print(f"Данные успешно сохранены в файл {filename}")

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []


    # Начальные данные
    thetas = np.deg2rad(np.arange(10, 15, 5))  # Угол наклона траектории
    alphas = np.deg2rad(np.arange(0, 4, 2))  # Угол атаки
    altitudes = np.arange(10000, 14000, 4000)  # Высота
    velocities = np.arange(3, 4, 1)  # Скорость
    theta_for_eng_trues = np.arange(5, 10, 5)
    theta_for_eng_offs = np.arange(5, 10, 5)
    engine_times = np.arange(2, 20, 2)
    max_x = float('-inf')
    best_simulation = None
    all_simulations_data = []

    # Переменная для подсчета количества симуляций
    simulation_count = 0

    for alt in altitudes:
        for vel in velocities:
            for theta in thetas:
                for alpha in alphas:
                    for theta_eng_true in theta_for_eng_trues:
                        for theta_eng_off in theta_for_eng_offs:
                            for eng_time in engine_times:
                                simulation_count += 1  # Инкрементируем счетчик симуляций
                                start_time = time.time()  # Замер времени начала симуляции
                                simulation_data = simulation(0.0, 1900, 0.01,
                                                             0, alt, vel * a, theta, alpha, False,
                                                             1800, 1000, 800,
                                                             1, 0.5, 1, 0.5, 0.5,
                                                             theta_eng_true, theta_eng_off, eng_time,
                                                             True, False, False)
                                end_time = time.time()  # Замер времени завершения симуляции

                                last_x = simulation_data["x"][-1]  # Получение последнего значения x из данных симуляции
                                simulation_info = {"Theta": np.rad2deg(theta), "Alpha": np.rad2deg(alpha),
                                                   "Altitude": alt, "X": last_x, "Time": end_time - start_time}
                                all_simulations_data.append(simulation_info)

                                if last_x > max_x:
                                    max_x = last_x
                                    best_simulation = simulation_data  # содержит информацию о симуляции с самой дальней траектории

    print(f"Количество симуляций: {simulation_count}")
    save_data_to_excel(all_simulations_data)

    if best_simulation is not None:
        plot_flight_data(best_simulation)
    else:
        print("Ни одна из симуляций не завершилась успешно.")

    # # Рикошетирующая
    # glide = simulation(0.0, 999, 0.01, 0, 2000, 3 * a, 1440, np.radians(25), np.radians(0), False, False, True, False)
    # plot_flight_data(glide)

    # type3 = simulation(0.0, 999, 0.01, 0, 2000, 3 * a, 1440, np.radians(30), np.radians(0), False, False, False, True)
    # plot_flight_data(glide)

