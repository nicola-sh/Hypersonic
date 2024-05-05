import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from isa import temperature, pressure, density

# Constants
g = 9.81  # acceleration due to gravity
Radius = 6371000  # radius of the Earth
a = 331  # speed of sound
R = 8.31446261815324  # gas law constant

# Aircraft parameters
length = 12.5  # length HA,m
diameter = 0.5  # diameter HA,m
radius = diameter / 2  # radius HA,m

# Mass parameters
mf = 450  # initial fuel mass, kg
mha = 990  # mass of aircraft, kg
m = 1440  # initial total mass, kg

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


# РАЗОБРАТЬСЯ
def massFlowRate(velocityAircraft):
    Pt = Pc * math.pow((1 + (k - 1) / 2), (-k / (k - 1)))  # N/m^2
    Tt = Tc * (1 / (1 + (k - 1) / 2))  # Kelvin

    return (Ae * Pt / math.sqrt(Tt)) * math.sqrt(
        (k * Mmw) / R) * (velocityAircraft / a) * math.pow(
        (1 + ((k - 1) / 2) * math.pow(
            (velocityAircraft / a), 2)), (-(k + 1) / (2 * (k - 1))))


# РАЗОБРАТЬСЯ
def velocityExhaustGas(altitude):
    return math.sqrt(((Tc * R) / Mmw) * ((2 * k) / (k - 1)) * (1 - math.pow((pressure(altitude) / Pc), ((k - 1) / k))))


# РАЗОБРАТЬСЯ
def thrust(altitude, velocityAircraft, m):
    if m > mha:
        thrust = massFlowRate(velocityAircraft) * (velocityExhaustGas(altitude) - velocityAircraft)
    else:
        # thrust = massFlowRate(velocityAircraft) * (a - velocityAircraft)
        thrust = 0

    return thrust


# РАЗОБРАТЬСЯ
def drag(altitude, velocityAircraft):
    return .5 * Cd * density(altitude) * velocityAircraft ** 2 * A0


# РАЗОБРАТЬСЯ
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
    dthetadt = (P * math.sin(alpha) + Lift - m * g * math.cos(theta) + (m * V ** 2 * math.cos(theta)) / (Radius + y)) / (m * V)
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
    k2_x, k2_y, k2_V, k2_theta, k2_m = calculate_derivatives(t + dt/2,
                                                         x + k1_x*dt/2,
                                                         y + k1_y*dt/2,
                                                         V + k1_V*dt/2,
                                                         theta + k1_theta*dt/2,
                                                         m + k1_m*dt/2)
    k3_x, k3_y, k3_V, k3_theta, k3_m = calculate_derivatives(t + dt/2,
                                                         x + k2_x*dt/2,
                                                         y + k2_y*dt/2,
                                                         V + k2_V*dt/2,
                                                         theta + k2_theta*dt/2,
                                                         m + k2_m*dt/2)
    k4_x, k4_y, k4_V, k4_theta, k4_m = calculate_derivatives(t + dt,
                                                         x + k3_x*dt,
                                                         y + k3_y*dt,
                                                         V + k3_V*dt,
                                                         theta + k3_theta*dt,
                                                         m + k3_m*dt)

    x = x + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    y = y + dt * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    V = V + dt * (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
    theta = theta + dt * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta) / 6

    if m > mha:
        m = m + dt * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6
    else:
        m = mha

    return x, y, V, theta, m


# Integration parameters
t = 0.0         # initial time, sec
t_end = 1200     # end time, sec
dt = 0.01       # time step, sec

# Initial parameters
Ae = math.pi * math.pow(radius, 2)
A0 = 0.54 * Ae
Afin = 0.3

V = 3 * a  # initial velocity
x = 0  # initial x position
y = 2000  # initial y position

Cd = 0.3
Cl = 0.35

Gc = 3  # fuel consumption rate, kg/s

k = 1.4  # gamma for air and 2h2o
Mmw = 0.02003  # 2H20, kg/mol \\ molecular weight
Tc = 1600  # Chamber Temp, Kelvin
Pc = 200000  # Chamber Pressure, Pa

theta = math.radians(47)  # initial angle
alpha = math.radians(0)  # angle of attack


while y >= 0 and t < t_end:
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

    data["massFlowRate"].append(massFlowRate(V))
    data["velocityExhaustGas"].append(velocityExhaustGas(y))

    data["t"].append(t)

    t += dt  # Update the time


# Создание графиков
fig1 = go.Figure()
fig2 = go.Figure()

# Добавление данных на графики
for key, name in [("x", "Положение по оси X"),
                  ("y", "Положение по оси Y"),
                  ("V", "Скорость"),
                  ("theta", "Угол"),
                  ("m", "Масса"),
                  ("Thrust", "Тяга"),
                  ("Drag", "Сопротивление"),
                  ("Lift", "Подъемная сила"),
                  # ("temperature", "Температура"),
                  # ("pressure", "Давление"),
                  # ("density", "Плотность"),
                  ("massFlowRate", "Расход массы"),
                  ("velocityExhaustGas", "Скорость выхлопных газов"),
                  ("t", "Время")]:
    fig1.add_trace(go.Scatter(x=data["t"], y=data[key], mode='lines', name=name, line=dict(width=2)))

fig2.add_trace(go.Scatter(x=data["x"], y=data["y"], mode='lines', name='Траектория', line=dict(width=2)))

# Настройка подписей к осям для первого графика
fig1.update_xaxes(title_text="Время")
fig1.update_yaxes(title_text="Значение")

# Настройка подписей к осям для второго графика
fig2.update_xaxes(title_text="x")
fig2.update_yaxes(title_text="y")

# Соединение графиков в одну HTML-страницу
fig = make_subplots(rows=2, cols=1, subplot_titles=("График", "График траектории полета"))

# Добавление данных на общий график
fig.add_traces(fig1.data, rows=[1]*len(fig1.data), cols=[1]*len(fig1.data))
fig.add_traces(fig2.data, rows=[2]*len(fig2.data), cols=[1]*len(fig2.data))

fig.update_layout(height=1800, width=1800, showlegend=True)

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