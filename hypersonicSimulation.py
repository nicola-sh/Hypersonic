import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from isa import density, temperature, pressure
from itertools import product
from joblib import Parallel, delayed
from datetime import datetime

# region Константы
g = 9.8                 # Ускорение свободного падения, м/с²
Radius = 6371000        # Радиус Земли, м
a = 331                 # Скорость звука, м/с
pi = 3.141592653589793  # Число Пи
# endregion

def simulation(t, t_end, dt, x, y, v, theta, alpha,
               m, mf, F0, Ffin, throttle):
    # region data
    mha = m - mf
    data = {
        "x": [], "y": [], "v": [], "theta": [], "alpha": [], "m": [],
        "weight": [], "acceleration": [],
        "Thrust": [], "Drag": [], "Lift": [],
        "air_mass_flow_rate": [], "fuel_mass_flow_rate": [],
        "specificImpulse": [], "throttle": [],
        "TW": [], "DW": [], "Thrust_req": [],
        "nx": [], "ny": [], "costheta": [], "K": [], "t": []
    }
    # endregion data

    # region func
    # region def control_throttle(throttle_pedal):
    #     global throttle
    #     max_throttle_change_rate = 0.001
    #     if throttle_pedal == 0:
    #         throttle = 0
    #     elif throttle_pedal > throttle:
    #         throttle += min(max_throttle_change_rate, throttle_pedal - throttle)
    #     else:
    #         throttle -= min(max_throttle_change_rate, throttle - throttle_pedal)
    #     return throttle
    # endregion

    def control_alpha(target_alpha):
        global alpha
        max_alpha_increase_rate = np.deg2rad(0.5)
        if target_alpha > alpha:
            alpha += min(max_alpha_increase_rate, target_alpha - alpha)
        elif target_alpha < alpha:
            alpha -= min(max_alpha_increase_rate, alpha - target_alpha)
        return alpha

    def air_mass_flow_rate(y, v):
        return density(y) * F0 * v

    def fuel_mass_flow_rate(y, v, throttle):
        # fuel_kerosene_ratio = 14.7         # Стехеометрический коэффициент для 1 керосин/ 14.7 воздух
        fuel_ratio_h2 = 8         # Стехеометрический коэффициент для 1 водород надо 8 кислорода
        co2 = 0.21
        # mfr = air_mass_flow_rate(y, v) / fuel_kerosene_ratio * throttle
        mfr = co2 * air_mass_flow_rate(y, v) / fuel_ratio_h2 * throttle
        return mfr

    def thrust(y, v, m, throttle):
        spec_g = 4000
        mach_ratio = 5 / (v / a)
        if m > mha and v > 2 * a:
            return fuel_mass_flow_rate(y, v, throttle) * spec_g * mach_ratio
        else:
            return 0

    def drag(y, v):
        mach = v / a
        if mach >= 2:
            Cd = 0.32
        elif mach >= 3:
            Cd = 0.28
        elif mach >= 4:
            Cd = 0.25
        elif mach >= 5:
            Cd = 0.24
        elif mach >= 6:
            Cd = 0.22
        elif mach >= 7:
            Cd = 0.2
        elif mach >= 8:
            Cd = 0.2
        else:
            Cd = 0.42
        return .5 * F0 * Cd * density(y) * v ** 2

    def lift(y, v, alpha):
        if alpha == 0:
            lift_coefficient = 0
        else:
            lift_coefficient = 0.5
        # if alpha == 0:
        #     lift_coefficient = 0
        # else:
        #     lift_coefficient = 0.6
        return (.5 * Ffin * lift_coefficient * density(y) * v ** 2)

    def specific_impulse(y, v, m):
        P = thrust(y, v, m, throttle)
        g_fuel = fuel_mass_flow_rate(y, v, throttle)
        return P / g_fuel if P > 0 and g_fuel > 0 else 0

    def calculate_derivatives(t, x, y, v, theta, m, alpha, throttle):
        P = thrust(y, v, m, throttle)
        Drag = drag(y, v)
        Lift = lift(y, v, alpha)
        g_fuel = fuel_mass_flow_rate(y, v, throttle)
        v_sq = (m * np.float64(v) ** 2 * np.cos(theta)) / (Radius + y)

        dxdt = (v * np.cos(theta) * Radius) / (Radius + y)
        dydt = v * np.sin(theta)
        dvdt = (P * np.cos(alpha) - Drag - (m * g * np.sin(theta))) / m
        dthetadt = (P * np.sin(alpha) + Lift - m * g * np.cos(theta) + v_sq) / (m * v)
        dmdt = -g_fuel
        return dxdt, dydt, dvdt, dthetadt, dmdt

    def runge_kutta_step(t, dt, x, y, v, theta, m, alpha, throttle):
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

    while y > 0 and t < t_end:
        # region equations
        P = thrust(y, v, m, throttle)
        W = m * g
        D = drag(y, v)
        L = lift(y, v, alpha)
        A = P / W
        Nx = (P - D) / W
        Ny = L / W
        K = 0
        PK = 0
        # endregion equations

        # region Баллистическая траектория
        # target_alpha = 0
        # throttle = 1
        # endregion

        # region Рикошетирующая траектория КОГДА ДЕЛАЛ ГОРИЗОНТ ПОМЕНЯЛ КОЭФФИЦИЕНТ ОПДЬЕМНОЙ С НУЛЯ ДО 6 В САМОЙ ФУНКЦИИ
        if D != 0:
            K = L / D
        if K != 0:
            PK = W / K

        if t < 20:
            throttle = min(1, throttle + 0.2)
            target_alpha = 0
        else:
            if y < 50000:
                if theta < np.deg2rad(5) and Ny != 0:
                    target_alpha = 6.5
                    throttle = min(1, throttle + 0.2)
                elif theta > np.deg2rad(5) and Ny != 0:
                    target_alpha = 0
                    throttle = max(0, throttle - 0.0001)
                elif Ny == 0:
                    target_alpha = 4
                    throttle = min(1, throttle + 0.001)
                else:
                    target_alpha = 0
                    throttle = max(0, throttle - 0.0001)
        # endregion Рикошетирующая траектория

        # region Горизонтальная траектория
        # if D != 0:
        #     K = L / D
        # if K != 0:
        #     PK = W / K
        #
        # if theta <= np.deg2rad(0.02):
        #     target_alpha = 6.5
        # else:
        #     target_alpha = 0
        #
        # if t < 50:
        #     throttle = min(1, throttle + 0.2)
        #     # target_alpha = 0
        # else:
        #     if y < 50000 and theta < np.deg2rad(50):
        #         # Проверяем условие равенства подъемной силы к весу ЛА
        #         if Ny == 1:
        #             throttle = min(1, throttle + 0.001)
        #             # Проверяем условие текущей силы тяги к силе сопротивления
        #             if P < D:
        #                 throttle = min(1, throttle + 0.0001)
        #             elif P > D:
        #                 throttle = max(0, throttle - 0.0001)
        #         else:
        #             if Ny < 1:
        #                 throttle = min(1, throttle + 0.0001)
        #             elif Ny > 1:
        #                 throttle = max(0, throttle - 0.0001)
        #     else:
        #         throttle = 0
        # endregion

        alpha = control_alpha(target_alpha)
        # throttle = control_throttle(target_throttle)
        x, y, v, theta, m = runge_kutta_step(t, dt, x, y, v, theta, m, alpha, throttle)
        #region Добавляем данные каждой итерации в массив data
        data["x"].append(x/1000)
        data["y"].append(y/1000)
        data["v"].append(v/a)
        data["m"].append(m)
        data["theta"].append(np.rad2deg(theta))
        data["alpha"].append(np.rad2deg(alpha))
        data["weight"].append(W/1000)
        data["Thrust"].append(P/1000)
        data["Thrust_req"].append(PK/100)
        data["specificImpulse"].append(specific_impulse(y, v, m))
        data["acceleration"].append(A)
        data["Drag"].append(D/1000)
        data["Lift"].append(L/1000)
        data["air_mass_flow_rate"].append(air_mass_flow_rate(y, v))
        data["fuel_mass_flow_rate"].append(fuel_mass_flow_rate(y, v, throttle))
        data["nx"].append(Nx)
        data["ny"].append(Ny)
        data["throttle"].append(throttle*100)
        data["K"].append(K)
        data["t"].append(t)
        #endregion
        t += dt
    return data

# region plots
def plot_data(data):
    fig = go.Figure()

    # Определение параметров для добавления на график
    traces = [
        ("t", "x", "Дальность, км"),
        ("t", "y", "Высота, км"),
        ("t", "v", "Скорость, мах"),
        ("t", "m", "Масса, кг"),
        ("t", "theta", "Угол Накл. Тр., гр."),
        ("t", "alpha", "Угол атаки, гр."),
        ("t", "weight", "Вес, кН"),
        ("t", "Thrust", "Сила тяги, кН"),
        ("t", "specificImpulse", "Уд. импульс, с"),
        ("t", "acceleration", "Ускорение, м/с²"),
        ("t", "Drag", "Сила сопр., кН"),
        ("t", "Lift", "Подъемная сила, кН"),
        ("t", "air_mass_flow_rate", "Расход возд., кг/с"),
        ("t", "fuel_mass_flow_rate", "Расход топл., кг/с"),
        ("t", "Thrust_req", "Потребная СТ, кН"),
        ("t", "nx", "Прод-ая перегрузка"),
        ("t", "ny", "Норм-ая перегрзука"),
        # ("t", "DW", "DW"),
        ("t", "throttle", "Throttle"),
        # ("t", "costheta", "costheta"),
        ("x", "y", "Траектория")
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
            y=0.98,  # Смещение заголовка по вертикали (чуть ниже)
            font=dict(
                family='Arial Black',
                size=16,
                color='black'  # Установка цвета заголовка
            )
        ),
        margin=dict(
            t=45,  # Верхний отступ (поднимает график)
            b=25,  # Нижний отступ
            l=30,
            r=30,
            pad=0
        ),
        xaxis=dict(
            title=dict(
                text="<b>Время, с</b>",
                font=dict(
                    size=16,
                    family='Arial Black',
                    color='black'
                )
            ),
            tickfont=dict(
                size=16,
                family='Arial Black',
                color='black'),
            title_font=dict(size=16),
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
                    size=16,
                    family='Arial Black',
                    color='black')),
            tickfont=dict(
                size=16,
                family='Arial Black',
                color='black'),
            title_font=dict(size=16),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=5,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            zerolinecolor='gray',  # Цвет линии на уровне 0
            linecolor='gray',  # Цвет осей
            linewidth=1  # Толщина линии оси
        ),
        width=1000,
        height=590,
        showlegend=True,
        legend=dict(
            font=dict(size=14, family='Arial Black', color='black'),
            x=1.02,  # Adjust as needed
            y=0.535,  # Adjust as needed
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
        label = f"θ={theta:.2f}°, α={alpha:.0f}°, H={alt/1000:.0f}, V={vel/a:.0f}"
        # label = f"θ={theta:.2f}"

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
            y=0.98,  # Смещение заголовка по вертикали (чуть ниже)
            font=dict(
                family='Arial Black',
                size=16,
                color='black'  # Установка цвета заголовка
            )
        ),
        margin=dict(
            t=45,  # Верхний отступ (поднимает график)
            b=25,  # Нижний отступ
            l=30,
            r=30,
            pad=0
        ),
        xaxis=dict(
            title=dict(
                text="<b>Дальность полета, км</b>",
                font=dict(
                    size=16,
                    family='Arial Black',
                    color='black'
                )
            ),
            tickfont=dict(
                size=16,
                family='Arial Black',
                color='black'),
            title_font=dict(size=16),
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
                    size=16,
                    family='Arial Black',
                    color='black')),
            tickfont=dict(
                size=16,
                family='Arial Black',
                color='black'),
            title_font=dict(size=16),
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
        legend=dict(
            font=dict(size=14, family='Arial Black', color='black'),
            x=1.02,  # Adjust as needed
            y=0.535,  # Adjust as needed
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
        ("t", "x", "Дальность, км"),
        ("t", "y", "Высота, км"),
        ("t", "v", "Скорость, Мах"),
        ("t", "m", "Масса, кг"),
        ("t", "theta", "Угол накл. тр., град."),
        ("t", "alpha", "Угол атаки, град."),
        ("t", "weight", "Вес, кН"),
        ("t", "Thrust", "Сила тяги, кН"),
        ("t", "specificImpulse", "Уд. импульс, сек"),
        ("t", "acceleration", "Ускорение, м/с²"),
        ("t", "Drag", "Сила сопр., кН"),
        ("t", "Lift", "Подъемная сила, кН"),
        ("t", "air_mass_flow_rate", "МР воздуха, кг/с"),
        ("t", "fuel_mass_flow_rate", "МР топлива, кг/с"),
        ("t", "K", "---K---"),
        ("t", "Thrust_req", "Потребная СТ, кН"),
        ("t", "nx", "Прод-ая перегрузка"),
        ("t", "ny", "Норм-ая перегрзука"),
        # ("t", "DW", "DW"),
        ("t", "throttle", "Throttle"),
        # ("t", "costheta", "costheta")
    ]
    # region сглаживание
    # # Преобразование данных в DataFrame для удобного сглаживания
    # df = pd.DataFrame(best_sim_data["SimulationData"])
    #
    # # Применение сглаживания для thrust и throttle
    # window_size = 2000  # Размер окна для скользящего среднего
    # df["Thrust_smooth"] = smooth_data(df["Thrust"], window_size)
    # df["throttle_smooth"] = smooth_data(df["throttle"], window_size)
    #
    # # Список цветов для линий, превышающих средний X
    # color_palette = plotly.colors.qualitative.Vivid
    #
    # # Добавление данных на график с использованием цикла
    # for i, (x_key, y_key, name) in enumerate(traces):
    #     color = color_palette[i % len(color_palette)]  # Подсчет цвета по индексу
    #     if y_key == "Thrust":
    #         y_data = df["Thrust_smooth"]
    #     elif y_key == "throttle":
    #         y_data = df["throttle_smooth"]
    #     else:
    #         y_data = df[y_key]
    #
    #     fig.add_trace(go.Scatter(x=df["t"],
    #                              y=y_data,
    #                              mode='lines',
    #                              name=name,
    #                              line=dict(width=3, color=color),
    #                              hovertemplate="Время: %{x}<br>Значение: %{y}"))
    # Список цветов для линий, превышающих средний X
    # endregion сглаживание
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
            y=0.98,  # Смещение заголовка по вертикали (чуть ниже)
            font=dict(
                family='Arial Black',
                size=16,
                color='black'  # Установка цвета заголовка
            )
        ),
        margin=dict(
            t=45,  # Верхний отступ (поднимает график)
            b=25,  # Нижний отступ
            l=30,
            r=30,
            pad=0
        ),
        xaxis=dict(
            title=dict(
                text="<b>Время, с</b>",
                font=dict(
                    size=16,
                    family='Arial Black',
                    color='black'
                )
            ),
            tickfont=dict(
                size=16,
                family='Arial Black',
                color='black'),
            title_font=dict(size=16),
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
                    size=16,
                    family='Arial Black',
                    color='black')),
            tickfont=dict(
                size=16,
                family='Arial Black',
                color='black'),
            title_font=dict(size=16),
            ticks="outside",
            tickformat="f",  # Настройка формата чисел на оси
            ticklen=5,  # Отступ между значением тика и самим тиком
            gridcolor='gray',  # Цвет сетки
            zerolinecolor='gray',  # Цвет линии на уровне 0
            linecolor='gray',  # Цвет осей
            linewidth=1  # Толщина линии оси
        ),
        width=1000,
        height=590,
        showlegend=True,
        legend=dict(
            font=dict(size=14, family='Arial Black', color='black'),
            x=1.02,  # Adjust as needed
            y=0.535,  # Adjust as needed
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

def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()
# endregion plots

def run_simulations_parallel(thetas, alphas, altitudes, machs, throttle):
    start_time = time.time()
    num_cores = -1  # используем все доступные ядра процессора
    # num_cores = 4  # Ограничиваем количество ядер
    all_simulations_data = Parallel(n_jobs=num_cores)(
        delayed(simulate_one)(alt, mach, theta, alpha, throttle)
        for alt, mach, theta, alpha in product(altitudes, machs, thetas, alphas)
    )
    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    # Найдем лучшую симуляцию
    max_x = float('-inf')
    best_sim_data = None
    for sim_data in all_simulations_data:
        if sim_data["X"] > max_x:
            max_x = sim_data["X"]
            best_sim_data = sim_data
    return all_simulations_data, best_sim_data, total_time_minutes

def simulate_one(alt, mach, theta, alpha, throttle):
    vel = mach * a
    simulation_data = simulation(
        0.0, 2000, 0.01,
        0, alt, vel, np.deg2rad(theta), np.deg2rad(alpha),
        1500, 500,
        0.2, 0.5, throttle)
    last_x = simulation_data["x"][-1]
    return {
        "Theta": theta,
        "Alpha": alpha,
        "Altitude": alt,
        "vel": vel,
        "X": last_x,
        "SimulationData": simulation_data
    }

if __name__ == '__main__':
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    throttle = 0
    alpha = 0
    sim = True
    if sim:
        # Начальные диапазоны данных для Баллистической
        # region Тест 1 Влияние ВЫСОТЫ и УНТ
        # alphas = np.arange(0, 1, 1)
        # machs = np.arange(3, 4, 1)
        # altitudes = np.arange(2000, 22000, 5000)
        # thetas = np.arange(20, 50, 2)
        # F0 = 0.1
        # Fs = 0.5
        # m = 1500/500

        # region Тест 2 Влияние УНТ
        # alphas = np.arange(0, 1, 1)
        # machs = np.arange(3, 4, 1)
        # altitudes = np.arange(2000, 3000, 2000)
        # thetas = np.arange(25, 35, 0.5)
        # Тест 4 Влияние МАХОВ
        # Тест 5 Влияние F0

        # Начальные диапазоны данных для Рикошетирующей
        # region Тест 1 Влияние ВЫСОТЫ
        alphas = np.arange(0, 1, 1)
        machs = np.arange(3, 4, 1)
        altitudes = np.arange(2000, 16000, 6000)
        thetas = np.arange(0, 50, 7)
        # Тест 2 Влияние УНТ
        # Тест 3 Влияние УА
        # Тест 4 Влияние МАХОВ
        # Тест 5 Влияние F0
        # Тест 6 Влияние K
        # F0 = 0.1
        # K = ?

        # Начальные диапазоны данных для Горизонтальной
        # Тест 1 Влияние ВЫСОТЫ
        # Тест 2 Влияние УНТ
        # Тест 3 Влияние УА
        # Тест 4 Влияние МАХОВ
        # Тест 5 Влияние F0
        # Тест 6 Влияние K

        print(f"Количество симуляций: {len(list(product(thetas, alphas, altitudes, machs)))}")

        all_sim_data_parallel, best_sim_data, total_time_minutes = run_simulations_parallel(
            thetas, alphas, altitudes, machs, throttle)
        print(f"Количество симуляций: {len(all_sim_data_parallel)}, общее время: {total_time_minutes:.2f} минут")
        plot_all_trajectories(all_sim_data_parallel)
        plot_data_for_best(best_sim_data)
    else:
        modeling = simulation(0.0, 2000, 0.01,
                              0, 5000, 4 * a,
                              np.deg2rad(35), np.deg2rad(0),
                              1800, 1000,
                              0.3, 0.5, throttle)
        plot_data(modeling)