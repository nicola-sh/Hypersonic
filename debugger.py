def control_throttle(throttle_pedal):
    """
    Изменяет тягу постепенно в соответствии с педалью управления тягой.
    :param throttle_pedal: Уровень "нажатия педали" управления тягой.
    :return: Уровень тяги после изменения.
    """
    global throttle
    max_throttle_change_rate = 0.05

    if throttle_pedal == 0:
        throttle = 0
    elif throttle_pedal > throttle:
        throttle += min(max_throttle_change_rate, throttle_pedal - throttle)
    else:
        throttle -= min(max_throttle_change_rate, throttle - throttle_pedal)

    return throttle


throttle = 0.5  # Начальное значение уровня тяги

# Первый вызов функции control_throttle с throttle_pedal = 0.5
throttle_pedal = 0.5
throttle = control_throttle(throttle_pedal)
print("Уровень тяги после первого вызова:", throttle)

# Второй вызов функции control_throttle с throttle_pedal = 0.3
throttle_pedal = 0.3
throttle = control_throttle(throttle_pedal)
print("Уровень тяги после второго вызова:", throttle)