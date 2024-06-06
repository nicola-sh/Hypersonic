import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors
import pandas as pd
from datetime import datetime
import time
from multiprocessing import Pool
import concurrent.futures
from itertools import product
from joblib import Parallel, delayed
from isa import density, temperature

def simulation(t, t_end, dt, x, y, v, theta, alpha,
               engine, m, mf, throttle, A0, Afin):
    # region data
    mha = m - mf
    data = {
        "x": [], "y": [], "v": [], "theta": [], "alpha": [], "m": [],
        "weight": [], "acceleration": [],
        "Thrust": [], "Drag": [], "Lift": [],
        "air_mass_flow_rate": [], "fuel_mass_flow_rate": [],
        "specificImpulse": [], "throttle": [],
        "tw_ratio": [], "dw_ratio": [], "lw_ratio_normal": [],
        "nxa": [], "nya": [], "costheta": [], "t": []
    }
    # endregion data
    # region func
    def control_alpha(target_alpha):
        global alpha
        max_alpha_increase_rate = np.deg2rad(0.5)

        if target_alpha > alpha:
            alpha += min(max_alpha_increase_rate, target_alpha - alpha)
        elif target_alpha < alpha:
            alpha -= min(max_alpha_increase_rate, alpha - target_alpha)

        return alpha

    def control_throttle(throttle_pedal):
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
        max_thrust = 30000
        return throttle * max_thrust

    def air_mass_flow_rate(altitude, v):
        return density(altitude) * A0 * v

    def fuel_mass_flow_rate(altitude, v, throttle):
        fuel_ratio = 14.7         # Стехеометрический коэффициент для керосин/воздух
        # fuel_ratio = 34         # Стехеометрический коэффициент для водород/воздух
        # fuel_ratio = 8         # Стехеометрический коэффициент для водород/кислород
        return air_mass_flow_rate(altitude, v) / fuel_ratio * throttle

    def thrust(altitude, v, m, throttle):
        mach_ratio = (v / a) / 12
        thrust_ref = engine_power(throttle)
        gffuel = fuel_mass_flow_rate(altitude, v, throttle)
        thrust_xx = (gffuel * thrust_ref) * mach_ratio
        if m > mha and v > 2 * a:
            if thrust_xx > 50000:
                thrust = 50000
            else:
                thrust = thrust_xx
        else:
            thrust = 0

        return thrust

    def drag(altitude, v):
        if v > 6 * a:
            drag_coefficient = 0.2
        else:
            drag_coefficient = 0.4
        return .5 * A0 * drag_coefficient * density(altitude) * v ** 2

    def lift(altitude, v, alpha):
        lift_coefficient = 0.65
        # if alpha == 0:
        #     lift_coefficient = 0
        # else:
        #     lift_coefficient = 0.6
        return (.5 * Afin * lift_coefficient * density(altitude) * v ** 2)

    def specific_impulse(y, v, m):
        T = thrust(y, v, m, throttle)
        g_fuel = fuel_mass_flow_rate(y, v, throttle) * g
        return T / g_fuel if T > 0 and g_fuel > 0 else 0

    def calculate_derivatives(t, x, y, v, theta, m, alpha, throttle):
        P = thrust(y, v, m, throttle)
        Drag = drag(y, v)
        Lift = lift(y, v, alpha)
        g_fuel = fuel_mass_flow_rate(y, v, throttle)

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
    # endregion func

    def engine_control(engine_duration_limit, alpha_engine, target_throttle, engine):
        global engine_duration

        target_alpha = np.deg2rad(alpha_engine)
        dt = 0.01
        if engine:
            if engine_duration >= engine_duration_limit:
                engine = False
                engine_duration = 0
            else:
                engine_duration += dt
        else:
            if dt < 0.1:
                engine = True
                engine_duration = 0
            target_throttle = 0

        return target_alpha, target_throttle, engine, engine_duration

    while y > 0 and t < t_end:
        # region equations
        thrust_value = thrust(y, v, m, throttle)
        d = drag(y, v)
        l = lift(y, v, alpha)
        weight = m * g
        acceleration = thrust_value / weight

        tw_ratio = thrust_value / weight
        dw_ratio = d / weight
        lw_ratio = l / weight # нормальная переггрузка когда = 1 => горизонт полет

        nxa = (thrust_value * np.cos(alpha) - d) / weight
        nya = (thrust_value * np.sin(alpha) + l) / weight
        costheta = np.cos(theta)
        # endregion equations

        # region Баллистическая траектория
        # if t != 0 and theta > 0:
        #     target_throttle = 1
        #     target_alpha = 0
        #     engine_duration_limit = 250
        #     engine = True
        # else:
        #     engine = False
        # endregion

        # region Начальный запуск
        if t < 20:
            target_throttle = 1
            target_alpha = 0
            engine_duration_limit = 20
            engine = True
        else:
            engine = False
        # endregion

        # Точка перегиба?? и переход к Рикошету или Горизонт

        # region Рикошетирующая траектория
        if t > 20 and y < 50000:
            if l>1:
                target_throttle = 0.6
                target_alpha = 6
                engine_duration_limit = 20
                engine = True
            elif lw_ratio == 1:
                target_throttle = 0.3
                target_alpha = 0
                engine_duration_limit = 20
                engine = True
            else:
                target_alpha = 0
                engine = False
        # endregion

        # region Горизонтальная траектория
        # if t > 20:
        #     if y < 50000 and theta < np.deg2rad(20):
        #         target_throttle = 0.4
        #         target_alpha = 6
        #         engine = True
        #     else:
        #         engine = False
        # endregion Горизонтальная траектория

        target_alpha, target_throttle, engine, engine_duration = engine_control(
            engine_duration_limit,
            target_alpha,
            target_throttle,
            engine)

        # if engine_duration >= 20:
        #     engine = False

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
        data["weight"].append(weight/1000)
        data["Thrust"].append(thrust(y, v, m, throttle)/1000)
        data["Drag"].append(d/1000)
        data["Lift"].append(l/1000)
        data["dw_ratio"].append(dw_ratio)
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

# region plots
def plot_data(data):
    fig = go.Figure()

    # Определение параметров для добавления на график
    traces = [
        ("t", "x", "Положение по оси X, км"),
        ("t", "y", "Положение по оси Y, км"),
        ("t", "v", "Скорость, м/с"),
        ("t", "m", "Масса, кг"),
        ("t", "weight", "Вес, кН"),
        ("t", "acceleration", "Ускорение, м/с²"),
        ("t", "theta", "θ, градусы"),
        ("t", "nxa", "nxa"),
        ("t", "nya", "nya"),
        ("t", "costheta", "costheta"),
        ("t", "alpha", "Угол атаки, градус"),
        ("t", "Thrust", "Сила тяги, кН"),
        ("t", "Drag", "Сила сопротивления, кН"),
        ("t", "Lift", "Подъемная сила, кН"),
        ("t", "air_mass_flow_rate", "Расход воздуха, кг/с"),
        ("t", "fuel_mass_flow_rate", "Расход топлива, кг/с"),
        ("t", "specificImpulse", "Удельный импульс, с"),
        ("t", "throttle", "Throttle"),
        ("t", "tw_ratio", "TW"),
        ("t", "dw_ratio", "DW"),
        ("t", "lw_ratio_normal", "LW"),
        ("x", "y", "Траектория полета")
    ]

    # Список цветов для линий, превышающих средний X
    color_palette = plotly.colors.qualitative.Vivid

    # Добавление данных на график с использованием цикла
    for i, (x_key, y_key, name) in enumerate(traces):
        color = color_palette[i % len(color_palette)]  # Подсчет цвета по индексу
        fig.add_trace(go.Scatter(x=data[x_key],
                                 y=data[y_key],
                                 mode='lines',
                                 name=name,
                                 line=dict(width=3, color=color),
                                 hovertemplate="Время: %{x}<br>Значение: %{y}"))

    # region Настройка макета графика
    fig.update_layout(
        title=dict(
            text="<b>Данные полета ЛА</b>",
            x=0.5,  # Выравнивание заголовка по центру
            y=0.9,  # Смещение заголовка по вертикали (чуть ниже)
            font=dict(
                color='black'  # Установка цвета заголовка
            )
        ),
        xaxis=dict(
            title=dict(
                text="<b>Время, с</b>",
                font=dict(
                    size=14,
                    color='black')),
            tickfont=dict(
                size=14,
                family='Arial Black',
                color='black'),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=5,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            linecolor='gray',  # Цвет осей
            linewidth=1  # Толщина линии оси
        ),
        yaxis=dict(
            title=dict(
                text="<b>Значение</b>",
                font=dict(
                    size=14,
                    color='black')),
            tickfont=dict(
                size=14,
                family='Arial Black',
                color='black'),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=5,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            zerolinecolor='gray',  # Цвет линии на уровне 0
            linecolor='gray',  # Цвет осей
            linewidth=1  # Толщина линии оси
        ),
        width=1000,
        height=500,
        showlegend=True,
        legend=dict(
            font=dict(size=15, family='Arial', color='black'),
            x=1.01,  # Adjust as needed
            y=0.5,  # Adjust as needed
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.7)',  # Adjust the background color and opacity
            bordercolor='black',  # Adjust the border color
            borderwidth=1,  # Adjust the border width
            # traceorder='normal',  # Keep the legend items in the same order as they appear in the plot
            # itemsizing='constant',  # Keep legend items constant in size
            # itemwidth=10  # Adjust the width of legend items
        ),
        paper_bgcolor='white',  # Установка белого фона бумаги
        plot_bgcolor='white'    # Установка белого фона графика
    )
    # endregion Настройка макета графика

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

    # Список цветов для линий, превышающих средний X
    color_palette = plotly.colors.qualitative.Vivid
    color_index = 0

    for sim_info in sorted_sim_data:
        simulation_data = sim_info["SimulationData"]
        theta = sim_info["Theta"]
        alpha = sim_info["Alpha"]
        alt = sim_info["Altitude"]
        vel = sim_info["vel"]
        label = f"θ: {theta}, a:{alpha} H: {alt/1000}, V: {vel/a}"

        # Определение цвета в зависимости от значения X
        # color = 'black' if simulation_data["x"][-1] >= mean_x else 'gray'

        # Определение цвета в зависимости от значения X
        if simulation_data["x"][-1] >= mean_x:
            color = color_palette[color_index % len(color_palette)]
            color_index += 1
        else:
            color = 'black'

        # Нормализация дальности и вычисление толщины линии
        normalized_x = (simulation_data["x"][-1] - min_x) / (max_x - min_x)
        line_width = 1 + normalized_x * (3 - 1)

        fig.add_trace(go.Scatter(x=simulation_data["x"], y=simulation_data["y"], mode='lines',
                                 name=label, line=dict(width=line_width, color=color),
                                 hovertemplate="X: %{x}<br>Y: %{y}"))
    # region Настройка макета графика
    fig.update_layout(
        title=dict(
            text="<b>Дальность полета: сравнительный анализ моделирования</b>",
            x=0.5,  # Выравнивание заголовка по центру
            y=0.9,  # Смещение заголовка по вертикали (чуть ниже)
            font=dict(
                color='black'  # Установка цвета заголовка
            )
        ),
        xaxis=dict(
            title=dict(
                text="<b>Дальность полета, км</b>",
                font=dict(
                    size=14,
                    color='black')),
            tickfont=dict(
                size=14,
                family='Arial Black',
                color='black'),
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
            title=dict(
                text="<b>Высота полета, км</b>",
                font=dict(
                    size=14,
                    color='black')),
            tickfont=dict(
                size=14,
                family='Arial Black',
                color='black'),
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
        width=1000,
        height=500,
        showlegend=True,
        legend=dict(
            font=dict(size=15, family='Arial', color='black'),
            x=1.01,  # Adjust as needed
            y=0.5,  # Adjust as needed
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.7)',  # Adjust the background color and opacity
            bordercolor='black',  # Adjust the border color
            borderwidth=1,  # Adjust the border width
            # traceorder='normal',  # Keep the legend items in the same order as they appear in the plot
            # itemsizing='constant',  # Keep legend items constant in size
            # itemwidth=10  # Adjust the width of legend items
        ),
        paper_bgcolor='white',  # Установка белого фона бумаги
        plot_bgcolor='white'    # Установка белого фона графика
    )
    # endregion Настройка макета графика

    fig.show()

def plot_data_for_best(best_sim_data):
    fig = go.Figure()

    traces = [
        ("t", "x", "Положение по оси X, км"),
        ("t", "y", "Положение по оси Y, км"),
        ("t", "v", "Скорость, м/с"),
        ("t", "m", "Масса, кг"),
        ("t", "weight", "Вес, кН"),
        ("t", "acceleration", "Ускорение, м/с²"),
        ("t", "theta", "θ, градусы"),
        ("t", "nxa", "nxa"),
        ("t", "nya", "nya"),
        ("t", "costheta", "costheta"),
        ("t", "alpha", "Угол атаки, градус"),
        ("t", "Thrust", "Сила тяги, кН"),
        ("t", "Drag", "Сила сопротивления, кН"),
        ("t", "Lift", "Подъемная сила, кН"),
        ("t", "air_mass_flow_rate", "Расход воздуха, кг/с"),
        ("t", "fuel_mass_flow_rate", "Расход топлива, кг/с"),
        ("t", "specificImpulse", "Удельный импульс, с"),
        ("t", "throttle", "Throttle"),
        ("t", "tw_ratio", "TW"),
        ("t", "dw_ratio", "DW"),
        ("t", "lw_ratio_normal", "LW"),
    ]

    # Список цветов для линий, превышающих средний X
    color_palette = plotly.colors.qualitative.Vivid

    # Добавление данных на график с использованием цикла
    for i, (x_key, y_key, name) in enumerate(traces):
        color = color_palette[i % len(color_palette)]  # Подсчет цвета по индексу
        fig.add_trace(go.Scatter(x=best_sim_data["SimulationData"][x_key],
                                 y=best_sim_data["SimulationData"][y_key],
                                 mode='lines',
                                 name=name,
                                 line=dict(width=3, color=color),
                                 hovertemplate="Время: %{x}<br>Значение: %{y}"))

    # region Настройка макета графика
    fig.update_layout(
        title=dict(
            text="<b>Данные полета ЛА</b>",
            x=0.5,  # Выравнивание заголовка по центру
            y=0.9,  # Смещение заголовка по вертикали (чуть ниже)
            font=dict(
                color='black'  # Установка цвета заголовка
            )
        ),
        xaxis=dict(
            title=dict(
                text="<b>Время, с</b>",
                font=dict(
                    size=14,
                    color='black')),
            tickfont=dict(
                size=14,
                family='Arial Black',
                color='black'),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=5,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            linecolor='gray',  # Цвет осей
            linewidth=1  # Толщина линии оси
        ),
        yaxis=dict(
            title=dict(
                text="<b>Значение</b>",
                font=dict(
                    size=14,
                    color='black')),
            tickfont=dict(
                size=14,
                family='Arial Black',
                color='black'),
            title_font=dict(size=14),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=5,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            zerolinecolor='gray',  # Цвет линии на уровне 0
            linecolor='gray',  # Цвет осей
            linewidth=1  # Толщина линии оси
        ),
        width=1000,
        height=500,
        showlegend=True,
        legend=dict(
            font=dict(size=15, family='Arial', color='black'),
            x=1.01,  # Adjust as needed
            y=0.5,  # Adjust as needed
            xanchor='left',
            yanchor='middle',
            bgcolor='rgba(255, 255, 255, 0.7)',  # Adjust the background color and opacity
            bordercolor='black',  # Adjust the border color
            borderwidth=1,  # Adjust the border width
            # traceorder='normal',  # Keep the legend items in the same order as they appear in the plot
            # itemsizing='constant',  # Keep legend items constant in size
            # itemwidth=10  # Adjust the width of legend items
        ),
        paper_bgcolor='white',  # Установка белого фона бумаги
        plot_bgcolor='white'    # Установка белого фона графика
    )
    # endregion Настройка макета графика

    fig.show()
# endregion plots

def simulate_one(alt, mach, theta, alpha, throttle):
    vel = mach * a
    simulation_data = simulation(
        0.0, 1500, 0.01,
        0, alt, vel, np.deg2rad(theta), np.deg2rad(alpha), False,
        1800, 1000, throttle,
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

if __name__ == '__main__':
    # region Константы
    g = 9.8  # Ускорение свободного падения, м/с²
    Radius = 6371000  # Радиус Земли, м
    a = 331  # Скорость звука, м/с
    pi = 3.141592653589793  # Число Пи
    throttle = 0
    alpha = 0
    engine_duration = 0
    # endregion
    # sim = False
    sim = True
    if sim:
        # # Начальные диапазоны данных для Баллистической
        # thetas = np.arange(15, 50, 5)
        # alphas = np.arange(0, 1, 1)
        # altitudes = np.arange(5000, 20000, 5000)
        # machs = np.arange(4, 5, 1)
        # # Начальные диапазоны данных для Рикошетирующей
        thetas = np.arange(15, 50, 4)
        alphas = np.arange(0, 1, 1)
        altitudes = np.arange(5000, 20000, 8000)
        machs = np.arange(4, 5, 1)
        # # Начальные диапазоны данных для Горизонтальной
        # thetas = np.arange(15, 50, 5)
        # alphas = np.arange(0, 1, 1)
        # altitudes = np.arange(5000, 20000, 5000)
        # machs = np.arange(4, 5, 1)

        print(f"Количество симуляций: {len(list(product(thetas, alphas, altitudes, machs)))}")

        all_sim_data_parallel, best_sim_data, total_time_minutes = run_simulations_parallel(
            thetas, alphas, altitudes, machs, throttle)
        print(f"Количество симуляций: {len(all_sim_data_parallel)}, общее время: {total_time_minutes:.2f} минут")
        plot_all_trajectories(all_sim_data_parallel)

        if best_sim_data is not None:
            plot_data_for_best(best_sim_data)
        else:
            print("Ни одна из симуляций не завершилась успешно.")
    else:
        modeling = simulation(0.0, 2000, 0.01,
                           0, 5000, 4 * a, np.deg2rad(35), np.deg2rad(0), False,
                           1800, 1000, throttle,
                           0.3, 0.5)
        plot_data(modeling)