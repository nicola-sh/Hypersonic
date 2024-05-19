import numpy as np
import pandas as pd
import plotly.graph_objects as go
from isa import density, temperature
from datetime import datetime
import time
from multiprocessing import Pool
import concurrent.futures
from itertools import product

# region Константы
g = 9.8                         # Ускорение свободного падения, м/с²
Radius = 6371000                # Радиус Земли, м
a = 331                         # Скорость звука, м/с
pi = 3.141592653589793          # Число Пи
throttle = 0
alpha = 0
# endregion

def simulation(t, t_end, dt,
               x, y, v, theta, alpha, engine,
               m, mha, mf, throttle,
               A0, Afin):

    """
    Проводит симуляцию движения объекта в атмосфере.

    :param t: Текущее время симуляции.
    :param t_end: Конечное время симуляции.
    :param dt: Шаг времени симуляции.
    :param x: Позиция объекта по оси X.
    :param y: Позиция объекта по оси Y.
    :param v: Скорость объекта.
    :param theta: Угол наклона объекта относительно горизонтали.
    :param alpha: Угол атаки объекта.
    :param engine: Переменная, указывающая, работает ли двигатель объекта.
    :param m: Масса объекта.
    :param mha: Минимальная масса объекта, при которой двигатель может работать.
    :param mf: Масса топлива.
    :param throttle: Уровень управления тягой.
    :param A0: Площадь поперечного сечения прямоточного ракетного двигателя.
    :param Afin: Площадь поперечного сечения сопла.
    :param ballistic: Параметр, указывающий на режим полета.
    :param glide: Параметр, указывающий на режим полета.
    :param type3: Параметр, указывающий на режим полета.
    """

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

        "specificImpulse": [],
        "throttle": [],

        "wd_ratio": [],

        "t": []
    }

    def control_alpha(target_alpha):
        """
        Изменяет угол атаки постепенно, приближая его к целевому значению.
        :param target_alpha: Целевой угол атаки в радианах.
        :return: Текущий угол атаки в радианах после изменения.
        """
        global alpha
        max_alpha_increase_rate = np.deg2rad(0.05)

        if target_alpha > alpha:
            alpha += min(max_alpha_increase_rate, target_alpha - alpha)
        elif target_alpha < alpha:
            alpha -= min(max_alpha_increase_rate, alpha - target_alpha)

        return alpha

    def control_throttle(throttle_pedal):
        """
        Изменяет тягу постепенно в соответствии с педалью управления тягой.
        :param throttle_pedal: Уровень "нажатия педали" управления тягой.
        :return: Уровень тяги после изменения.
        """
        global throttle
        max_throttle_increase_rate = 0.01

        if throttle_pedal == 0:
            throttle = 0
        elif throttle_pedal > throttle:
            throttle += min(max_throttle_increase_rate, throttle_pedal - throttle)
            throttle = min(1.0, max(0.0, throttle))

        return throttle

    def engine_power(throttle):
        """
        Вычисляет мощность двигателя в зависимости от уровня "нажатия педали".
        :param throttle: Уровень "нажатия педали".
        :return: Мощность двигателя.
        """
        max_thrust = 300000  # Максимальная сила тяги ПВРД
        return throttle * max_thrust

    def air_mass_flow_rate(altitude, v):
        """
        Рассчитывает массовый расход воздуха через прямоточный ракетный двигатель.
        :param altitude: Высота над уровнем моря, м.
        :param v: Скорость объекта, м/с.
        :return: Массовый расход воздуха через прямоточный ракетный двигатель, кг/с.
        """
        return density(altitude) * A0 * v

    def fuel_mass_flow_rate(altitude, v, throttle):
        """
        Рассчитывает массовый расход топлива через прямоточный ракетный двигатель.
        :param altitude: Высота над уровнем моря, м.
        :param v: Скорость объекта, м/с.
        :param throttle: Уровень нажатия на педаль управления тягой.
        :return: Массовый расход топлива через прямоточный ракетный двигатель, кг/с.
        """
        fuel_ratio = 15         # Стехеометрический коэффициент для керосин/воздух
        return air_mass_flow_rate(altitude, v) / fuel_ratio * throttle

    def thrust(altitude, v, m, throttle):
        """
        Рассчитывает силу тяги прямоточного ракетного двигателя.
        :param altitude: Высота над уровнем моря, м.
        :param v: Скорость объекта, м/с.
        :param m: Масса объекта, кг.
        :param throttle: Уровень нажатия на педаль управления тягой.
        :return: Сила тяги ПВРД, Н.
        """
        mach_ratio = (v / a) / 12   # Вероятно надо изменить!
        thrust_ref = engine_power(throttle)
        thrust_xx = (fuel_mass_flow_rate(altitude, v, throttle) * thrust_ref) * mach_ratio
        # вероятно надо добавить сюда переменную engine или как то еще ограничить...
        if m > mha and v > 2 * a:
            if thrust_xx > 300000:
                thrust = 300000
            else:
                thrust = thrust_xx
        else:
            thrust = 0

        return thrust

    def drag(altitude, v):
        """
        Рассчитывает силу сопротивления ПВРД.
        :param altitude: Высота над уровнем моря, м.
        :param v: Скорость объекта, м/с.
        :return: Сила сопротивления ПВРД, Н.
        """
        drag_coefficient = 0.3
        return .5 * A0 * drag_coefficient * density(altitude) * v ** 2

    def lift(altitude, v):
        """
        Рассчитывает подъемную силу ПВРД.
        :param altitude: Высота над уровнем моря, м.
        :param v: Скорость объекта, м/с.
        :return: Подъемная сила ПВРД, Н.
        """
        lift_coefficient = 0.4
        return .5 * Afin * lift_coefficient * density(altitude) * v ** 2

    def specific_impulse(y, v, m):
        """
        Вычисляет удельный импульс.
        :param y: Высота объекта, м.
        :param v: Скорость объекта, м/с.
        :param m: Масса объекта, кг.
        :return: Удельный импульс.
        """
        T = thrust(y, v, m, throttle)
        g_fuel = fuel_mass_flow_rate(y, v, throttle) * g

        return T / g_fuel if T > 0 and g_fuel > 0 else 0

    def calculate_derivatives(t, x, y, v, theta, m, alpha, throttle):
        P = thrust(y, v, m, throttle)
        Drag = drag(y, v)
        Lift = lift(y, v)
        g_fuel = fuel_mass_flow_rate(y, v, throttle)

        dxdt = v * np.cos(theta) * Radius / (Radius + y)
        dydt = v * np.sin(theta)
        dvdt = (P * np.cos(alpha) - Drag - (m * g * np.sin(theta))) / m
        dthetadt = (P * np.sin(alpha) + Lift - m * g * np.cos(theta) +
                    (m * v ** 2 * np.cos(theta)) / (Radius + y)) / (m * v)
        dmdt = -g_fuel

        return dxdt, dydt, dvdt, dthetadt, dmdt

    def runge_kutta_step(t, dt, x, y, v, theta, m, alpha, throttle):
        """
        Выполняет один шаг метода Рунге-Кутты четвертого порядка точности
        для численного решения дифференциальных уравнений.

        Погрешность метода Рунге-Кутты четвертого порядка
        составляет приблизительно O(dt^5)

        :param t: Текущее время.
        :param dt: Величина шага времени.
        :param x: Текущая горизонтальная координата.
        :param y: Текущая вертикальная координата.
        :param v: Текущая скорость.
        :param theta: Текущий угол наклона траектории.
        :param m: Текущая масса.
        :param alpha: Текущий угол атаки.
        :param throttle: Уровень нажатия на педаль управления тягой.
        :return: Новые значения горизонтальной координаты, вертикальной координаты, скорости, угла наклона и массы.
        """
        k1_x, k1_y, k1_v, k1_theta, k1_m = calculate_derivatives(t, x, y, v, theta, m, alpha, throttle)
        k2_x, k2_y, k2_v, k2_theta, k2_m = calculate_derivatives(t + dt / 2,
                                                                 x + k1_x * dt / 2,
                                                                 y + k1_y * dt / 2,
                                                                 v + k1_v * dt / 2,
                                                                 theta + k1_theta * dt / 2,
                                                                 m + k1_m * dt / 2,
                                                                 alpha, throttle)
        k3_x, k3_y, k3_v, k3_theta, k3_m = calculate_derivatives(t + dt / 2,
                                                                 x + k2_x * dt / 2,
                                                                 y + k2_y * dt / 2,
                                                                 v + k2_v * dt / 2,
                                                                 theta + k2_theta * dt / 2,
                                                                 m + k2_m * dt / 2,
                                                                 alpha, throttle)
        k4_x, k4_y, k4_v, k4_theta, k4_m = calculate_derivatives(t + dt,
                                                                 x + k3_x * dt,
                                                                 y + k3_y * dt,
                                                                 v + k3_v * dt,
                                                                 theta + k3_theta * dt,
                                                                 m + k3_m * dt,
                                                                 alpha, throttle)

        x = x + dt * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y = y + dt * (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        v = v + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        theta = theta + dt * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6

        if m > mha:
            m = m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
        else:
            m = mha

        return x, y, v, theta, m

    def engine_control(engine, engine_duration, engine_duration_limit, dt, target_throttle):
        """
        Управляет состоянием работы двигателя и его продолжительностью.

        :param engine: Переменная, указывающая, работает ли двигатель объекта.
        :param engine_duration: Продолжительность работы двигателя.
        :param engine_duration_limit: Максимальная продолжительность работы двигателя.
        :param dt: Шаг времени симуляции.
        :param target_throttle: Целевой уровень управления тягой.
        :return: Обновленные значения переменных engine и engine_duration.
        """
        if engine:
            engine_duration += dt
            if engine_duration > engine_duration_limit:
                engine = False
                engine_duration = 0
        else:
            if dt < 0.1:
                engine = True
            engine_duration = 0
            target_throttle = 0

        return engine, engine_duration, target_throttle

    # Переменные для управления двигателем и углом атаки
    start_engine_time = 8              # Время начала стартовой работы двигателя
    engine_duration_limit = 15          # Максимальная продолжительность работы двигателя
    engine_duration = 0                 # Переменная для отслеживания времени работы двигателя
    eng_dt = 0.1
    max_pitch_angle = np.deg2rad(30)    # Максимальный угол атаки для подъема
    min_pitch_angle = np.deg2rad(10)    # Минимальный угол атаки для равновесия
    check_balance_interval = 1          # каждые 10 секунд

    while y > 0 and t < t_end:
        weight = m * g
        acceleration = thrust(y, v, m, throttle) / m

        d = drag(y, v) / 1000
        w = weight / 1000
        wd_ratio = w - d

        # стартовый запуск двигателя
        if t < start_engine_time and y<30000:
            target_throttle = 0.3
            engine = True
        else:
            engine = False

        if y < 45000 and theta < np.deg2rad(20):
            target_throttle = 0.3
            target_alpha = np.deg2rad(6.2)
            engine = True
        else:
            target_alpha = np.deg2rad(0)
            engine = False


        engine, engine_duration, target_throttle = engine_control(engine, engine_duration, engine_duration_limit, eng_dt, target_throttle)
        throttle = control_throttle(target_throttle)
        alpha = control_alpha(target_alpha)

        x, y, v, theta, m = runge_kutta_step(t, dt, x, y, v, theta, m, alpha, throttle)

        #region Добавляем данные каждой итерации в массив data
        data["x"].append(x/1000)
        data["y"].append(y/1000)
        data["v"].append(v)
        data["theta"].append(np.rad2deg(theta))  # Угол наклона траектории
        data["alpha"].append(np.rad2deg(alpha))  # Угол атаки
        data["m"].append(m)
        data["weight"].append(weight/1000)
        data["acceleration"].append(acceleration)
        data["Thrust"].append(thrust(y, v, m, throttle)/1000)
        data["Drag"].append(drag(y, v)/1000)
        data["Lift"].append(lift(y, v)/1000)
        data["specificImpulse"].append(specific_impulse(y, v, m))
        data["air_mass_flow_rate"].append(air_mass_flow_rate(y, v))
        data["fuel_mass_flow_rate"].append(fuel_mass_flow_rate(y, v, throttle))
        data["throttle"].append(throttle)
        data["wd_ratio"].append(wd_ratio)

        data["t"].append(t)
        #endregion

        t += dt  # Обновляем время

    return data


def plot_flight_data(data):
    fig = go.Figure()

    # Определение параметров для добавления на график
    traces = [
        ("t", "x", "Положение ЛА по оси X, км от времени"),
        ("t", "y", "Положение ЛА по оси Y, км от времени"),
        ("t", "v", "Скорость ЛА, м/с от времени"),
        ("t", "m", "Масса ЛА, кг от времени"),
        ("t", "weight", "Вес ЛА, кН от времени"),
        ("t", "acceleration", "Ускорение, м/с² от времени"),
        ("t", "theta", "Угол наклона траектории, градус от времени"),
        ("t", "alpha", "Угол атаки, градус от времени"),
        ("t", "Thrust", "Сила тяги, кН от времени"),
        ("t", "Drag", "Сила сопротивления, кН от времени"),
        ("t", "Lift", "Подъемная сила, кН от времени"),
        ("t", "air_mass_flow_rate", "Массовый расход воздуха ПВРД, кг/с от времени"),
        ("t", "fuel_mass_flow_rate", "Массовый расход топлива ПВРД, кг/с от времени"),
        ("t", "specificImpulse", "Удельный импульс, от времени"),
        ("t", "throttle", "throttle"),
        ("t", "wd_ratio", "wd_ratio"),
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
    sim = False
    if sim:
        start_time = time.time()  # Замер времени начала симуляции

        # Начальные данные
        thetas = np.deg2rad(np.arange(0, 40, 2))  # Угол наклона траектории
        alphas = np.deg2rad(np.arange(0, 1, 1))  # Угол атаки
        altitudes = np.arange(10000, 14000, 2000)  # Высота
        velocities = np.arange(3, 4, 0.5)  # Скорость
        # theta_for_eng_trues = np.arange(-10, 10, 5)
        # theta_for_eng_offs = np.arange(-10, 10, 5)
        # engine_times = np.arange(10, 20, 5)

        # Рассчет количества симуляций
        simulation_count = len(
            list(product(thetas, alphas, altitudes, velocities)))
            # list(product(thetas, alphas, altitudes, velocities, theta_for_eng_trues, theta_for_eng_offs, engine_times)))

        # Вывод количества симуляций в консоль
        print(f"Количество симуляций: {simulation_count}")

        max_x = float('-inf')
        best_simulation = None
        all_simulations_data = []

        # Переменная для подсчета количества симуляций
        simulation_count = 0

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []

            for alt in altitudes:
                for vel in velocities:
                    for theta in thetas:
                        for alpha in alphas:
                            # for theta_eng_true in theta_for_eng_trues:
                            #     for theta_eng_off in theta_for_eng_offs:
                            #         for eng_time in engine_times:
                                        futures.append(executor.submit(simulation,
                                                                       0.0, 1900, 0.01,
                                                                       0, alt, vel * a, theta, alpha, False,
                                                                       3800, 3000, 800,
                                                                       1, 0.5, 1, 0.5, 0.5))
        # Получение результатов
        for future in concurrent.futures.as_completed(futures):
            simulation_data = future.result()

            last_x = simulation_data["x"][-1]  # Получение последнего значения x из данных симуляции
            simulation_info = {"Theta": np.rad2deg(theta), "Alpha": np.rad2deg(alpha),
                               "Altitude": alt, "X": last_x}
            all_simulations_data.append(simulation_info)

            if last_x > max_x:
                max_x = last_x
                best_simulation = simulation_data  # содержит информацию о симуляции с самой дальней траектории

        end_time = time.time()  # Замер времени завершения симуляции
        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60
        print(f"Количество симуляций: {len(all_simulations_data)}, общее время: {total_time_minutes:.2f} минут")

        save_data_to_excel(all_simulations_data)

        if best_simulation is not None:
            plot_flight_data(best_simulation)
        else:
            print("Ни одна из симуляций не завершилась успешно.")
    else:
        simul2 = simulation(0.0, 1900, 0.01,
                           0, 9000, 4 * a, np.deg2rad(30), np.deg2rad(0), False,
                           1900, 1000, 900, throttle,
                           1, 0.5)
        plot_flight_data(simul2)

