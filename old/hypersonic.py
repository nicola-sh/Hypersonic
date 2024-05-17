#region Imports
import math
import plotly.graph_objects as go
from isa import temperature, pressure, density
#endregion

#region Constants
pi = 3.141592653589793
g = 9.81  # acceleration due to gravity
Radius = 6371000  # radius of the Earth
a = 331  # speed of sound
R = 8.31446261815324  # gas law constant
#endregion

#region Aircraft parameters
length = 12.5  # length HA,m
diameter = 0.5  # diameter HA,m
radius = diameter / 2  # radius HA,m

Ae = pi * radius ** 2   # Выходное сечение
Am = Ae                 # Сечение миделя ракеты
A0 = 0.54 * Ae          # Сечение воздухозаборник
Akr = 0.47 * Ae
Afin = 0.2              # Площадь крыльев

# print("Inlet area: " + str(A0) + " m^2")
# print("Midsection area: " + str(Am) + " m^2")
# print("Exit area: " + str(Ae) + " m^2")

# Mass parameters
mf = 450  # initial fuel mass, kg
mha = 990  # mass of aircraft, kg
m = 1440  # initial total mass, kg

engine = False

#endregion


data = {
    "x": [],
    "y": [],
    "V": [],
    "theta": [],
    "m": [],

    "Thrust": [],
    "Drag": [],
    "Lift": [],

    "massFlowRate": [],
    "velocityExhaustGas": [],

    "t": []
}

#region DATA
# CxData = {
#     0: 0.32,
#     0.5: 0.28,
#     1: 0.24,
#     1.5: 0.42,
#     1.8: 0.3,
#     2: 0.3,
#     3: 0.26,
#     4: 0.25,
#     5: 0.23,
#     6: 0.22,
#     7: 0.21,
#     8: 0.2
# }


# def get_Cd(mach):
#     mach_floor = max([m for m in CxData.keys() if m <= mach])
#     mach_ceil = min([m for m in CxData.keys() if m >= mach])
#
#     if mach_floor == mach_ceil:
#         return CxData.get(mach_floor)
#
#     cd_floor = CxData[mach_floor]
#     cd_ceil = CxData[mach_ceil]
#
#     cd = cd_floor + (mach - mach_floor) * (cd_ceil - cd_floor) / (mach_ceil - mach_floor)
#     return cd

# (H, M): ImpulseSpecific
# SpecificImpulseData = {
#     (2000, 3): 11300,
#     (2000, 3.5): 11492,
#     (2000, 4): 11338,
#     (2000, 4.5): 10926,
#     (2000, 5): 10322,
#     (2000, 5.5): 9583.2,
#     (2000, 6): 9000,
#
#     (5000, 3): 11493,
#     (5000, 3.5): 11760,
#     (5000, 4): 11610,
#     (5000, 4.5): 11302,
#     (5000, 5): 10704,
#     (5000, 5.5): 9984.9,
#     (5000, 6): 9200.7,
#
#     (11000, 3): 11848,
#     (11000, 3.5): 12322,
#     (11000, 4): 12253,
#     (11000, 4.5): 12018,
#     (11000, 5): 11511,
#     (11000, 5.5): 10858,
#     (11000, 6): 10105,
#
#     (25000, 3): 11729,
#     (25000, 3.5): 12230,
#     (25000, 4): 12176,
#     (25000, 4.5): 11925,
#     (25000, 5): 11423,
#     (25000, 5.5): 10744,
#     (25000, 6): 9989.6,
#
#     (30000, 3): 11554,
#     (30000, 3.5): 11969,
#     (30000, 4): 11870,
#     (30000, 4.5): 11592,
#     (30000, 5): 11053,
#     (30000, 5.5): 10348,
#     (30000, 6): 9568.6,
#
#     (35000, 3): 11363,
#     (35000, 3.5): 11700,
#     (35000, 4): 11563,
#     (35000, 4.5): 11251,
#     (35000, 5): 10687,
#     (35000, 5.5): 9957.4,
#     (35000, 6): 9141.1,
#
#     (40000, 3): 11000,
#     (40000, 3.5): 11430,
#     (40000, 4): 11258,
#     (40000, 4.5): 10914,
#     (40000, 5): 10321,
#     (40000, 5.5): 9568.3,
#     (40000, 6): 8729.9
# }


# def get_specific_impulse(height, mach):
#     # Check if exact values are present in the data
#     if (height, mach) in SpecificImpulseData:
#         return SpecificImpulseData[(height, mach)]
#     else:
#         # Find the nearest heights
#         lower_height = max([h for h, _ in SpecificImpulseData if h < height])
#         upper_height = min([h for h, _ in SpecificImpulseData if h > height])
#
#         # Find the nearest mach numbers
#         lower_mach = max([m for _, m in SpecificImpulseData if m < mach])
#         upper_mach = min([m for _, m in SpecificImpulseData if m > mach])
#
#         # Interpolate between the nearest values
#         specific_impulse_lower = SpecificImpulseData.get((lower_height, lower_mach))
#         specific_impulse_upper = SpecificImpulseData.get((upper_height, upper_mach))
#
#         # If one of the heights or machs is not available
#         if specific_impulse_lower is None or specific_impulse_upper is None:
#             return None
#
#         specific_impulse = (specific_impulse_lower + specific_impulse_upper) / 2
#         return specific_impulse
#endregion


specific_impulse_const = 11000 #m/s
fuel_mass_flow_rate = 3  # Массовый расход топлива через ПВРД
k = 1.4     # gamma for air and 2h2o
Cd = 0.27
Cl = 0.35
ratio = 15
Gc = 3.5  # fuel consumption rate, kg/s


# def specificThrust(altitude, velocityAircraft, fuel_mass_flow_rate):
#     mach = velocityAircraft / a
#
#     specific_thrust = thrust(altitude, mach, fuel_mass_flow_rate) / airMFR(altitude, velocityAircraft)
#
#     return specific_thrust

def fuelMFR(altitude, velocityAircraft):
    return density(altitude) * Akr * velocityAircraft


def airMFR(altitude, velocityAircraft):
    return density(altitude)*A0*velocityAircraft


def thrust(altitude, velocityAircraft, m):
    global engine
    
    mach = velocityAircraft / a

    if m > mha and engine == True:
        # thrust = airMFR(altitude, velocityAircraft) * (velocityExhaustGas(altitude) - velocityAircraft)
        thrust = specific_impulse_const * fuelMFR(altitude, velocityAircraft)
    else:
        # thrust = massFlowRate(velocityAircraft) * (a - velocityAircraft)
        thrust = 0
        engine = False

    return thrust


def drag(altitude, velocityAircraft):
    return .5 * Cd * density(altitude) * velocityAircraft ** 2 * A0


def lift(altitude, velocityAircraft):
    return .5 * Cl * density(altitude) * velocityAircraft ** 2 * Afin


# Function to calculate the derivatives using the Runge-Kutta method
def calculate_derivatives(t, x, y, V, theta, m):
    P = thrust(y, V, m)
    Drag = drag(y, V)
    Lift = lift(y, V)

    dxdt = V * math.cos(theta) * Radius / (Radius + y)
    dydt = V * math.sin(theta)
    dVdt = (P * math.cos(alpha) - Drag - (m * g * math.sin(theta))) / m
    dthetadt = (P * math.sin(alpha) + Lift - m * g * math.cos(theta) +
                (m * V ** 2 * math.cos(theta)) / (Radius + y)) / (m * V)
    dmdt = -Gc

    return dxdt, dydt, dVdt, dthetadt, dmdt


# Function to perform one step of the Runge-Kutta method
def runge_kutta_step(t, dt, x, y, V, theta, m):
    k1_x, k1_y, k1_V, k1_theta, k1_m = calculate_derivatives(t,
                                                             x,
                                                             y,
                                                             V,
                                                             theta,
                                                             m)
    k2_x, k2_y, k2_V, k2_theta, k2_m = calculate_derivatives(t + dt / 2,
                                                             x + k1_x * dt / 2,
                                                             y + k1_y * dt / 2,
                                                             V + k1_V * dt / 2,
                                                             theta + k1_theta * dt / 2,
                                                             m + k1_m * dt / 2)
    k3_x, k3_y, k3_V, k3_theta, k3_m = calculate_derivatives(t + dt / 2,
                                                             x + k2_x * dt / 2,
                                                             y + k2_y * dt / 2,
                                                             V + k2_V * dt / 2,
                                                             theta + k2_theta * dt / 2,
                                                             m + k2_m * dt / 2)
    k4_x, k4_y, k4_V, k4_theta, k4_m = calculate_derivatives(t + dt,
                                                             x + k3_x * dt,
                                                             y + k3_y * dt,
                                                             V + k3_V * dt,
                                                             theta + k3_theta * dt,
                                                             m + k3_m * dt)

    x = x + dt * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    y = y + dt * (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
    V = V + dt * (k1_V + 2 * k2_V + 2 * k3_V + k4_V) / 6
    theta = theta + dt * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6

    if m > mha:
        m = m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
    else:
        m = mha

    return x, y, V, theta, m


#region Initial V,x,y,theta,alpha
V = 3 * a  # initial velocity
x = 0  # initial x position
y = 2000  # initial y position

theta = math.radians(45)  # initial angle
alpha = math.radians(0)  # angle of attack
#endregion

#region Integration parameters
t = 0.0  # initial time, sec
t_end = 1200  # end time, sec
dt = 0.01  # time step, sec
#endregion

while y >= 0 and t < t_end:

    if t > 0.5:
        engine = True

    x, y, V, theta, m = runge_kutta_step(t, dt, x, y, V, theta, m)

    # Append the current x and y values to the trajectory arrays
    data["x"].append(x)
    data["y"].append(y)
    data["V"].append(V)
    data["theta"].append(theta)
    data["m"].append(m)

    data["Thrust"].append(thrust(y, V, m))
    data["Drag"].append(drag(y, V))
    data["Lift"].append(lift(y, V))

    data["massFlowRate"].append(airMFR(y, V))
    # data["velocityExhaustGas"].append(velocityExhaustGas(y))

    data["t"].append(t)

    t += dt  # Update the time

#region Вывод данных
fig = go.Figure()

# # Добавление данных на график
# for key, name in [("x", "Положение ЛА по оси X"),
#                   ("y", "Положение ЛА по оси Y"),
#                   ("V", "Скорость ЛА"),
#                   ("theta", "Угол наклона траектории"),
#                   ("m", "Масса ЛА"),
#                   ("Thrust", "Тяга"),
#                   ("Drag", "Сила сопротивление"),
#                   ("Lift", "Подъемная сила"),
#                   # ("temperature", "Температура на данной высоте"),
#                   # ("pressure", "Давление на данной высоте"),
#                   # ("density", "Плотность на данной высоте"),
#                   ("massFlowRate", "Массовый расход воздуха ПВРД"),
#                   ("velocityExhaustGas", "Скорость выхлопных газов")
#                   ]:
#     fig.add_trace(go.Scatter(x=data["t"], y=data[key], mode='lines', name=name, line=dict(width=2)))


# Добавление графика траектории (X vs Y)
fig.add_trace(go.Scatter(x=data["t"], y=data["x"], mode='lines', name="Положение ЛА по оси X", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["y"], mode='lines', name="Положение ЛА по оси Y", line=dict(width=2)))

fig.add_trace(go.Scatter(x=data["t"], y=data["V"], mode='lines', name="Скорость ЛА", line=dict(width=2)))
fig.add_trace(
    go.Scatter(x=data["t"], y=data["theta"], mode='lines', name="Угол наклона траектории", line=dict(width=2)))

fig.add_trace(go.Scatter(x=data["t"], y=data["Thrust"], mode='lines', name="Сила тяги", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["Drag"], mode='lines', name="Сила сопротивление", line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["Lift"], mode='lines', name="Подъемная сила", line=dict(width=2)))

fig.add_trace(go.Scatter(x=data["t"], y=data["massFlowRate"], mode='lines', name="Массовый расход воздуха ПВРД",
                         line=dict(width=2)))
fig.add_trace(go.Scatter(x=data["t"], y=data["velocityExhaustGas"], mode='lines', name="Скорость выхлопных газов",
                         line=dict(width=2)))

# Преобразование данных в числовой формат
specific_impulse = []
for thrust, velocity_ex_gas in zip(data["Thrust"], data["velocityExhaustGas"]):
    if velocity_ex_gas != 0:
        specific_impulse.append(thrust / velocity_ex_gas)
    else:
        specific_impulse.append(None)

# График удельной тяги
fig.add_trace(go.Scatter(x=data["t"], y=specific_impulse, mode='lines',
                         name="Удельная тяга", line=dict(width=2)))


fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode='lines', name="Траектория полета ЛА", line=dict(width=2)))

# Настройка макета графика
fig.update_layout(
    title="График",
    xaxis=dict(
        title="Время",
        tickfont=dict(size=18, family='Arial'),
        title_font=dict(size=18, family='Arial')
    ),
    yaxis=dict(
        title="Значение",
        tickfont=dict(size=18, family='Arial'),
        title_font=dict(size=18, family='Arial')
    ),
    height=1000,
    width=1700,
    showlegend=True,
    legend=dict(font=dict(size=18, family='Arial')),
)

# print(fig)
fig.show()

# # Plot x vs y
# plt.figure(figsize=(8, 6))
# plt.plot(data["x"], data["y"])
# plt.xlabel('x (km)')
# plt.ylabel('y (km)')
# plt.title('Rocket Trajectory (x vs y)')
# plt.grid(True)
# plt.show()
#
# # Plot mass against time
# plt.figure(figsize=(10, 6))
# plt.plot(data["t"], data["m"])
# plt.xlabel('Time (s)')
# plt.ylabel('Mass (kg)')
# plt.title('Rocket Mass vs Time')
# plt.grid(True)
# plt.show()
#
# # Plot velocity (V), lift, and drag against time (t)
# plt.figure(figsize=(10, 6))
# plt.plot(data["t"], data["Thrust"], color='b', linewidth=2, label='Velocity')
# plt.plot(data["t"], data["Lift"], color='g', linestyle='--', linewidth=2, label='Lift')
# plt.plot(data["t"], data["Drag"], color='r', linestyle='-.', linewidth=2, label='Drag')
# plt.xlabel('Time (s)')
# plt.ylabel('Force (kN) / Velocity (m/s)')
# plt.title('Rocket Velocity vs Time and Forces')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Создание нового окна с графиками
# plt.figure(figsize=(10, 6))
#
# # График температуры
# plt.plot(data["t"], data["temperature"], label='Temperature')
#
# # График давления
# plt.plot(data["t"], data["pressure"], label='Pressure')
#
# # График плотности
# plt.plot(data["t"], data["density"], label='Density')
#
# # Добавление заголовка и меток осей
# plt.title('Environmental Conditions vs Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Value')
#
# # Добавление легенды
# plt.legend()
#
# # Отображение графика
# plt.grid(True)
# plt.show()
#endregion