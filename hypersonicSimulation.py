import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from isa import density, temperature
from datetime import datetime
import time
from multiprocessing import Pool
import concurrent.futures
from itertools import product
from joblib import Parallel, delayed
# import gc  # Импорт сборщика мусора

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
               m, mf, throttle,
               A0, Afin):

    mha = m - mf
    planning = False
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

        "tw_ratio": [],
        "wd_ratio": [],
        "lw_ratio_normal": [],
        "nxa": [],
        "nya": [],
        "costheta": [],

        "t": []
    }

    def control_alpha(target_alpha):
        """
        Изменяет угол атаки постепенно, приближая его к целевому значению.
        :param target_alpha: Целевой угол атаки в радианах.
        :return: Текущий угол атаки в радианах после изменения.
        """
        global alpha
        max_alpha_increase_rate = np.deg2rad(0.5)

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
        max_throttle_change_rate = 0.1

        if throttle_pedal == 0:
            throttle = 0
        elif throttle_pedal > throttle:
            throttle += min(max_throttle_change_rate, throttle_pedal - throttle)
        else:
            throttle -= min(max_throttle_change_rate, throttle - throttle_pedal)

        return throttle

    def engine_power(throttle):
        """
        Вычисляет мощность двигателя в зависимости от уровня "нажатия педали".
        :param throttle: Уровень "нажатия педали".
        :return: Мощность двигателя.
        """
        max_thrust = 30000  # Максимальная сила тяги ПВРД
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
        fuel_ratio = 14.7         # Стехеометрический коэффициент для керосин/воздух
        # fuel_ratio = 34         # Стехеометрический коэффициент для водород/воздух
        # fuel_ratio = 8         # Стехеометрический коэффициент для водород/кислород
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
        gffuel = fuel_mass_flow_rate(altitude, v, throttle)
        thrust_xx = (gffuel * thrust_ref) * mach_ratio
        # вероятно надо добавить сюда переменную engine или как то еще ограничить...
        if m > mha and v > 2 * a:
            # thrust = (gffuel * thrust_ref) * mach_ratio
            if thrust_xx > 200000:
                thrust = 200000
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
        if v > 6 * a:
            drag_coefficient = 0.2
        else:
            drag_coefficient = 0.4
        return .5 * A0 * drag_coefficient * density(altitude) * v ** 2

    def lift(altitude, v, alpha):
        """
        Рассчитывает подъемную силу ПВРД.
        :param altitude: Высота над уровнем моря, м.
        :param v: Скорость объекта, м/с.
        :return: Подъемная сила ПВРД, Н.
        """
        # lift_coefficient = 0.65
        if alpha == 0:
            lift_coefficient = 0
        else:
            lift_coefficient = 0.6
        return (.5 * Afin * lift_coefficient * density(altitude) * v ** 2)

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
        Lift = lift(y, v, alpha)
        g_fuel = fuel_mass_flow_rate(y, v, throttle)
        nxa = (P * np.cos(alpha) - Drag) / (m * g)
        nya = (P * np.sin(alpha) + Lift) / (m * g)
        costheta = np.cos(theta)

        dxdt = (v * np.cos(theta) * Radius) / (Radius + y)
        dydt = v * np.sin(theta)
        dvdt = (P * np.cos(alpha) - Drag - (m * g * np.sin(theta))) / m

        v_sq = (m * np.float16(v) ** 2 * np.cos(theta)) / (Radius + y)
        dthetadt = (P * np.sin(alpha) + Lift - m * g * np.cos(theta) + v_sq) / (m * v)
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

    # def engine_control(engine, engine_duration, engine_duration_limit, dt, target_throttle, engine_cooldown,
    #                    engine_cooldown_limit):
    #     """
    #     Управляет состоянием работы двигателя и его продолжительностью.
    #
    #     :param engine: Переменная, указывающая, работает ли двигатель объекта.
    #     :param engine_duration: Продолжительность работы двигателя.
    #     :param engine_duration_limit: Максимальная продолжительность работы двигателя.
    #     :param dt: Шаг времени симуляции.
    #     :param target_throttle: Целевой уровень управления тягой.
    #     :param engine_cooldown: Время до следующего разрешенного включения двигателя.
    #     :param engine_cooldown_limit: Минимальное время между включениями двигателя.
    #     :return: Обновленные значения переменных engine, engine_duration, target_throttle, engine_cooldown.
    #     """
    #     if engine:
    #         engine_duration += dt
    #         if engine_duration > engine_duration_limit:
    #             engine = False
    #             engine_duration = 0
    #             target_throttle = 0
    #             engine_cooldown = engine_cooldown_limit
    #     else:
    #         if engine_cooldown > 0:
    #             engine_cooldown -= dt
    #         else:
    #             engine = True
    #             engine_duration = 0
    #             engine_cooldown = 0
    #             target_throttle = 0
    #
    #     return engine, engine_duration, target_throttle, engine_cooldown

    engine_duration = 0                 # Переменная для отслеживания времени работы двигателя
    engine_duration_limit = 10          # Максимальная продолжительность работы двигателя
    engine_cooldown = 0
    engine_cooldown_limit = 10           # Время ожидания перед повторным включением двигателя
    eng_dt = 0.5        # изм с 0.001 до 0.1

    while y > 0 and t < t_end:

        tyaga = thrust(y, v, m, throttle)
        d = drag(y, v)
        l = lift(y, v, alpha)
        weight = m * g

        w = weight
        acceleration = tyaga / m

        tw_ratio = tyaga / w

        wd_ratio = d / w
        lw_ratio = l / w # нормальная переггрузка когда = 1 => горизонт полет

        nxa = (tyaga * np.cos(alpha) - d) / weight
        nya = (tyaga * np.sin(alpha) + l) / weight
        costheta = np.cos(theta)

        if t < 20:
            # Начальная баллистическая траектория
            target_throttle = 0.8
            target_alpha = np.deg2rad(alpha)
            # initial_engine_state = True
            engine = True
        else:
            # initial_engine_state = False
            engine = False

        # if t > 20:
        #     if t > 20 and y < 45000 and theta < np.deg2rad(5):
        #         target_throttle = 0.3
        #         target_alpha = np.deg2rad(6.2)
        #         engine = True


        if t > 20 and y < 50000 :
            if theta < np.deg2rad(0) and lw_ratio != 0:
                target_throttle = 0.4
                target_alpha = np.deg2rad(6.2)
                engine = True
            elif theta > np.deg2rad(0) and lw_ratio != 0:
                target_alpha = np.deg2rad(0)
                engine = False
            elif lw_ratio == 0 and d != tyaga:
                target_throttle = 0.15
                target_alpha = np.deg2rad(4)
                engine = True
            elif lw_ratio == 0 and d == tyaga:
                target_throttle = 0.1
                target_alpha = np.deg2rad(2)
                engine = True
            else:
                target_alpha = np.deg2rad(0)
                engine = False

        # if t > 20 and nya <= costheta or theta <= np.deg2rad(0) and planning == False:
        #     target_throttle = 0.9
        #     target_alpha = np.deg2rad(9)
        #     engine = True
        # elif theta >= np.deg2rad(0):
        #     planning = True
        #     print("yuea")
        #
        # if planning:
        #     target_throttle = 0.2
        #     target_alpha = np.deg2rad(3)
        #     engine = True


        # # # Переход к горизонтальному полету
        # if t > 40:
        #     # if theta < 0:  # Пикирование
        #     #     # print(f"Время: {t}, Высота: {y}, Угол траектории: {np.rad2deg(theta)}, Тяга: {tyaga}")
        #     #     target_alpha = np.deg2rad(7)  # Угол атаки для выхода из пикирования
        #     #     target_throttle = 0.4
        #     #     engine = True
        #     # elif 0 <= theta <= np.deg2rad(5):  # Горизонтальный полет
        #     #     target_alpha = np.deg2rad(3)
        #     #     target_throttle = 0.2
        #     #     engine = True
        #     # else:
        #     #     target_alpha = np.deg2rad(5)  # Подъем для горизонтального полета
        #     #     target_throttle = 0.3
        #     #     engine = True
        #     if d > 1000:  # точка перегиба
        #         target_alpha = np.deg2rad(12)  # Угол атаки для выхода из пикирования
        #         target_throttle = 0.5
        #         engine = True

            # elif 1 <= lw_ratio <= 5:  # Горизонтальный полет
            #     target_alpha = np.deg2rad(6)
            #     target_throttle = 0.4
            #     engine = True
            # elif d > tyaga:
            #     target_alpha = np.deg2rad(2)  # Подъем для горизонтального полета
            #     target_throttle = 0.1
            #     engine = True

        # if y > 45000:
        #     engine = False


        engine, engine_duration, target_throttle = engine_control(engine,
                                                                  engine_duration,
                                                                  engine_duration_limit,
                                                                  eng_dt,
                                                                  target_throttle)
        # engine, engine_duration, target_throttle, engine_cooldown = engine_control(
        #     engine if t >= 2000 else initial_engine_state,
        #     engine_duration,
        #     engine_duration_limit,
        #     eng_dt,
        #     target_throttle,
        #     engine_cooldown,
        #     engine_cooldown_limit
        # )
        throttle = control_throttle(target_throttle)
        alpha = control_alpha(target_alpha)

        x, y, v, theta, m = runge_kutta_step(t, dt, x, y, v, theta, m, alpha, throttle)

        #region Добавляем данные каждой итерации в массив data
        data["x"].append(x/1000)
        data["y"].append(y/1000)
        data["v"].append(v)
        data["theta"].append(np.rad2deg(theta))
        data["alpha"].append(np.rad2deg(alpha))
        data["m"].append(m)
        data["weight"].append(w/1000)
        data["Thrust"].append(thrust(y, v, m, throttle)/1000)
        data["Drag"].append(d/1000)
        data["Lift"].append(l/1000)
        data["wd_ratio"].append(wd_ratio)
        data["lw_ratio_normal"].append(lw_ratio)
        data["tw_ratio"].append(tw_ratio)
        data["acceleration"].append(acceleration)
        data["specificImpulse"].append(specific_impulse(y, v, m))
        data["air_mass_flow_rate"].append(air_mass_flow_rate(y, v))
        data["fuel_mass_flow_rate"].append(fuel_mass_flow_rate(y, v, throttle))
        data["throttle"].append(throttle*10)
        data["costheta"].append(costheta)
        data["nxa"].append(nxa)
        data["nya"].append(nya)

        data["t"].append(t)
        #endregion

        t += dt  # Обновляем время

    return data


def plot_data(data):
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
        ("t", "nxa", "nxa"),
        ("t", "nya", "nya"),
        ("t", "costheta", "costheta"),
        ("t", "alpha", "Угол атаки, градус от времени"),
        ("t", "Thrust", "Сила тяги, кН от времени"),
        ("t", "Drag", "Сила сопротивления, кН от времени"),
        ("t", "Lift", "Подъемная сила, кН от времени"),
        ("t", "air_mass_flow_rate", "Массовый расход воздуха ПВРД, кг/с от времени"),
        ("t", "fuel_mass_flow_rate", "Массовый расход топлива ПВРД, кг/с от времени"),
        ("t", "specificImpulse", "Удельный импульс, от времени"),
        ("t", "throttle", "throttle"),
        ("t", "tw_ratio", "tw_ratio"),
        ("t", "wd_ratio", "wd_ratio"),
        ("t", "lw_ratio_normal", "lw_ratio_normal"),
        ("x", "y", "Траектория полета ЛА")
    ]

    # Добавление данных на график с использованием цикла
    for x_key, y_key, name in traces:
        fig.add_trace(go.Scatter(x=data[x_key], y=data[y_key], mode='lines',
                                 name=name, line=dict(width=3),
                                 hovertemplate="Время: %{x}<br>Значение: %{y}"))

    # Настройка макета графика
    fig.update_layout(
        title="<b>График данных полета</b>",
        title_x=0.35,  # Выравнивание заголовка по центру
        # Добавление отступа между значением тика и самим тиком
        xaxis=dict(
            title="<b>Время, с</b>",
            tickfont=dict(size=14),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=12  # Отступ между значением тика и самим тиком
        ),
        yaxis=dict(
            title="<b>Значение</b>",
            tickfont=dict(size=14),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=12  # Отступ между значением тика и самим тиком
        ),
        width=1400,
        height=700,
        showlegend=True,
        legend=dict(font=dict(size=14))
    )

    fig.show()

def plot_all_trajectories(all_sim_data):
    fig = go.Figure()

    # Находим средний X по всем симуляциям
    mean_x = sum(sim_info["SimulationData"]["x"][-1] for sim_info in all_sim_data) / len(all_sim_data)

    # Сортируем симуляции по дальности
    sorted_sim_data = sorted(all_sim_data, key=lambda x: x["SimulationData"]["x"][-1], reverse=True)
    # Определяем минимальный и максимальный X для нормализации
    min_x = min(sim_info["SimulationData"]["x"][-1] for sim_info in sorted_sim_data)
    max_x = max(sim_info["SimulationData"]["x"][-1] for sim_info in sorted_sim_data)

    for sim_info in sorted_sim_data:
        simulation_data = sim_info["SimulationData"]
        theta = sim_info["Theta"]
        alpha = sim_info["Alpha"]
        alt = sim_info["Altitude"]
        vel = sim_info["vel"]
        label = f"θ: {theta}, α: {alpha}, H: {alt}, V: {vel/a}"

        # Определение цвета в зависимости от значения X
        color = 'black' if simulation_data["x"][-1] >= mean_x else 'gray'

        # Нормализация дальности и вычисление толщины линии
        normalized_x = (simulation_data["x"][-1] - min_x) / (max_x - min_x)
        line_width = 0.5 + normalized_x * (3 - 0.5)

        fig.add_trace(go.Scatter(x=simulation_data["x"], y=simulation_data["y"], mode='lines',
                                 name=label, line=dict(width=line_width, color=color),
                                 hovertemplate="X: %{x}<br>Y: %{y}"))

    fig.update_layout(
        title="<b>Траектории полета всех симуляций</b>",
        title_x=0.5,  # Выравнивание заголовка по центру
        xaxis=dict(
            title="<b>Положение ЛА по оси X, км</b>",
            tickfont=dict(size=14),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=12,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            linecolor='gray',  # Цвет осей
            linewidth=1,  # Толщина линии оси
            dtick=50     # Шаг тиков
        ),
        yaxis=dict(
            title="<b>Положение ЛА по оси Y, км</b>",
            tickfont=dict(size=14),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=12,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            zerolinecolor='gray',  # Цвет линии на уровне 0
            linecolor='gray',  # Цвет осей
            linewidth=1,  # Толщина линии оси
            dtick=10     # Шаг тиков
        ),
        width=1400,
        height=700,
        showlegend=True,
        legend=dict(font=dict(size=14)),
        paper_bgcolor='white',  # Установка белого фона бумаги
        plot_bgcolor='white'    # Установка белого фона графика
    )

    fig.show()

def data_to_excel(data, filename=None):
    # Если имя файла не указано, используйте текущее время
    if filename is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"data_{current_time}.xlsx"

    # Создание DataFrame из данных
    df = pd.DataFrame(data)

    # Сохранение DataFrame в файл Excel
    df.to_excel(filename, index=False)

    print(f"Данные успешно сохранены в файл {filename}")

def simulate_one(alt, mach, theta, alpha, throttle):
    vel = mach * a
    simulation_data = simulation(
        0.0, 2000, 0.01,
        0, alt, vel, np.deg2rad(theta), np.deg2rad(alpha), False,
        2000, 1000, throttle,
        0.3, 0.5)

    last_x = simulation_data["x"][-1]
    return {
        "Theta": theta,
        "Alpha": alpha,
        "Altitude": alt,
        "vel": vel,
        "X": last_x,
        "SimulationData": simulation_data
    }

def run_simulations_parallel(thetas, alphas, altitudes, machs, throttle):
    start_time = time.time()

    num_cores = -1  # используем все доступные ядра процессора
    # num_cores = 4  # Ограничиваем количество ядер
    all_simulations_data = Parallel(n_jobs=num_cores)(
        delayed(simulate_one)(alt, mach, theta, alpha, throttle)
        for alt, mach, theta, alpha in product(altitudes, machs, thetas, alphas)
    )

    end_time = time.time()
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60

    # Найдем лучшую симуляцию
    max_x = float('-inf')
    best_sim_data = None
    for sim_data in all_simulations_data:
        if sim_data["X"] > max_x:
            max_x = sim_data["X"]
            best_sim_data = sim_data

    return all_simulations_data, best_sim_data, total_time_minutes

def plot_data_for_best(best_sim_data):
    fig = go.Figure()

    traces = [
        ("t", "x", "Положение ЛА по оси X, км от времени"),
        ("t", "y", "Положение ЛА по оси Y, км от времени"),
        ("t", "v", "Скорость ЛА, м/с от времени"),
        ("t", "m", "Масса ЛА, кг от времени"),
        ("t", "weight", "Вес ЛА, кН от времени"),
        ("t", "acceleration", "Ускорение, м/с² от времени"),
        ("t", "theta", "Угол наклона траектории, градус от времени"),
        ("t", "nxa", "nxa"),
        ("t", "nya", "nya"),
        ("t", "costheta", "costheta"),
        ("t", "alpha", "Угол атаки, градус от времени"),
        ("t", "Thrust", "Сила тяги, кН от времени"),
        ("t", "Drag", "Сила сопротивления, кН от времени"),
        ("t", "Lift", "Подъемная сила, кН от времени"),
        ("t", "air_mass_flow_rate", "Массовый расход воздуха ПВРД, кг/с от времени"),
        ("t", "fuel_mass_flow_rate", "Массовый расход топлива ПВРД, кг/с от времени"),
        ("t", "specificImpulse", "Удельный импульс, от времени"),
        ("t", "throttle", "throttle"),
        ("t", "tw_ratio", "tw_ratio"),
        ("t", "wd_ratio", "wd_ratio"),
        ("t", "lw_ratio_normal", "lw_ratio_normal")
    ]

    for x_key, y_key, name in traces:
        fig.add_trace(go.Scatter(x=best_sim_data["SimulationData"][x_key], y=best_sim_data["SimulationData"][y_key], mode='lines', name=name))

    fig.update_layout(
        title="<b>График лучшей траектории полета</b>",
        title_x=0.5,  # Выравнивание заголовка по центру
        xaxis_title="<b>Время, с</b>",
        yaxis_title="<b>Значение</b>",
        width=1400,
        height=700,
        showlegend=True,
        legend=dict(font=dict(size=14))
    )

    fig.show()

def run_simulations(thetas, alphas, altitudes, machs, throttle):
    all_simulations_data = []
    max_x = float('-inf')
    best_simulation = None
    start_time = time.time()

    for alt, mach, theta, alpha in product(altitudes, machs, thetas, alphas):
        vel = mach * a
        simulation_data = simulation(
            0.0, 2000, 0.01,
            0, alt, vel, np.deg2rad(theta), np.deg2rad(alpha), False,
            2000, 1000, throttle,
            0.3, 0.5)

        last_x = simulation_data["x"][-1]
        simulation_info = {
            "Theta": theta,
            "Alpha": alpha,
            "Altitude": alt,
            "vel": vel,
            "X": last_x,
            "SimulationData": simulation_data  # Сохраняем данные симуляции целиком
        }
        all_simulations_data.append(simulation_info)

        if last_x > max_x:
            max_x = last_x
            best_simulation = simulation_data

    end_time = time.time()
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60
    print(f"Количество симуляций: {len(all_simulations_data)}, общее время: {total_time_minutes:.2f} минут")

    return all_simulations_data, best_simulation

if __name__ == '__main__':
    sim = True
    if sim:

        # Начальные данные
        thetas = np.arange(15, 50, 5)                        # Угол наклона траектории
        alphas = np.arange(0, 1, 1)                         # Угол атаки
        altitudes = np.arange(5000, 15000, 2000)           # Высота
        machs = np.arange(4, 5, 1)                          # Скорость изм с 3 до 4

        # Рассчет количества симуляций
        simulation_count = len(list(product(thetas, alphas, altitudes, machs)))
        print(f"Количество симуляций: {simulation_count}")

        # all_sim_data, best_sim_data = run_simulations(thetas, alphas, altitudes, machs, throttle)
        all_sim_data_parallel, best_sim_data, total_time_minutes = run_simulations_parallel(thetas, alphas, altitudes, machs, throttle)

        print(f"Количество симуляций: {len(all_sim_data_parallel)}, общее время: {total_time_minutes:.2f} минут")
        # data_to_excel(all_sim_data)
        # data_to_excel(all_sim_data_parallel)
        # plot_all_trajectories(all_sim_data)
        plot_all_trajectories(all_sim_data_parallel)

        if best_sim_data is not None:
            plot_data_for_best(best_sim_data)
        else:
            print("Ни одна из симуляций не завершилась успешно.")
    else:
        simul2 = simulation(0.0, 2000, 0.01,
                           0, 5000, 4 * a, np.deg2rad(16), np.deg2rad(0), False,
                           2000, 1000, throttle,
                           0.3, 0.5)
        plot_data(simul2)
# if __name__ == '__main__':
#     sim = True
#     if sim:
#         start_time = time.time()  # Замер времени начала симуляции
#
#         # Начальные данные
#         altitudes = np.arange(8000, 14000, 2000)  # Высота
#         velocities = np.arange(3, 4, 0.5)  # Скорость
#         thetas = np.deg2rad(np.arange(20, 45, 5))  # Угол наклона траектории
#         # alphas = np.deg2rad(np.arange(0, 1, 1))  # Угол атаки
#         # theta_for_eng_trues = np.arange(-10, 10, 5)
#         # theta_for_eng_offs = np.arange(-10, 10, 5)
#         # engine_times = np.arange(10, 20, 5)
#
#         # Рассчет количества симуляций
#         simulation_count = len(list(product(altitudes, velocities, thetas)))
#         print(f"Количество симуляций: {simulation_count}")
#
#         max_x = float('-inf')
#         best_simulation = None
#         all_simulations_data = []
#         completed_simulations = 0
#
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             futures = []
#             for alt, vel, theta in product(altitudes, velocities, thetas):
#                 futures.append(executor.submit(simulation,
#                                                0.0, 1000, 0.01,
#                                                0, alt, a * vel, theta, np.deg2rad(0),
#                                                False, 2000, 1000,
#                                                throttle, 0.3, 0.5))
#         # Получение результатов
#         for future in concurrent.futures.as_completed(futures):
#             simulation_data = future.result()
#
#             last_x = simulation_data["x"][-1]  # Получение последнего значения x из данных симуляции
#             alt, vel, theta = future.args[3:6]  # Extract inputs from future
#             # Получение входных данных симуляции из future
#             simulation_info = {"Theta": np.rad2deg(theta),
#                                "Alpha": np.rad2deg(alpha),
#                                "Altitude": alt,
#                                "Vel": vel,
#                                "X": last_x}
#             all_simulations_data.append(simulation_info)
#
#             if last_x > max_x:
#                 max_x = last_x
#                 best_simulation = simulation_data  # содержит информацию о симуляции с самой дальней траекторией
#
#             completed_simulations += 1
#
#         end_time = time.time()  # Замер времени завершения симуляции
#         total_time_seconds = end_time - start_time
#         total_time_minutes = total_time_seconds / 60
#         print(f"Количество завершенных симуляций: {completed_simulations}, общее время: {total_time_minutes:.2f} минут")
#
#         data_to_excel(all_simulations_data)
#
#         if best_simulation is not None:
#             plot_data(best_simulation)
#         else:
#             print("Ни одна из симуляций не завершилась успешно.")
#     else:
#         simul2 = simulation(0.0, 1900, 0.05,
#                            0, 5000, 4 * a, np.deg2rad(30), np.deg2rad(0), False,
#                            2000, 1000,
#                            throttle, 0.3, 0.5)
#         plot_data(simul2)

