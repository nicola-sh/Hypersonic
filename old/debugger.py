# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from scipy.interpolate import RegularGridInterpolator
#
# # Исходные данные
# mach_numbers = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
# heights = np.array([2000, 5000, 11000, 25000, 30000, 35000, 40000])  # высоты в метрах
# specific_impulse = np.array([
#     [11300.0, 11493.0, 11848.0, 11729.0, 11554.0, 11363.0, 11000.0],
#     [11492.0, 11760.0, 12322.0, 12230.0, 11969.0, 11700.0, 11430.0],
#     [11338.0, 11610.0, 12253.0, 12176.0, 11969.0, 11563.0, 111258.0],
#     [10926.0, 11302.0, 12018.0, 11925.0, 11592.0, 11563.0, 111258.0],
#     [10322.0, 10704.0, 11511.0, 11423.0, 111053.0, 10687.0, 10321.0],
#     [9583.2, 9984.9, 10858.0, 10744.0, 10348.0, 9957.4, 9568.3],
#     [9000.0, 9200.7, 10105.0, 9989.6, 9568.6, 9141.1, 8729.9]
# ])
#
# # Создание функции интерполяции
# interp_func = RegularGridInterpolator((heights, mach_numbers), specific_impulse.T, bounds_error=False, fill_value=None)
#
# # Определение диапазонов для интерполяции
# height_range = np.linspace(2000, 40000, 1000)  # от 2000 до 40000 метров
# mach_range = np.linspace(3.0, 6.0, 1000)       # от 3.0 до 6.0
#
# # Создание сетки для интерполяции
# mesh_height, mesh_mach = np.meshgrid(height_range, mach_range, indexing='ij')
# points = np.array([mesh_height.flatten(), mesh_mach.flatten()]).T
#
# # Интерполяция
# interp_values = interp_func(points)
# interp_values = interp_values.reshape((1000, 1000))
#
# # Построение графика
# plt.figure(figsize=(10, 8))
# plt.contourf(mesh_mach, mesh_height / 10000, interp_values, levels=50, cmap='viridis')
# plt.colorbar(label='Specific Impulse (s)')
# plt.xlabel('Mach Number')
# plt.ylabel('Height (km)')
# plt.title('Interpolated Specific Impulse')
# plt.show()

# import numpy as np
# from scipy.interpolate import interp2d
# from scipy.interpolate import RegularGridInterpolator
#
# mach_numbers = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
# heights = np.array([2, 5, 11, 25, 30, 35, 40])
# specific_impulse = np.array([
#     [11300.0, 11493.0, 11848.0, 11729.0, 11554.0, 11363.0, 11000.0],
#     [11492.0, 11760.0, 12322.0, 12230.0, 11969.0, 11700.0, 11430.0],
#     [11338.0, 11610.0, 12253.0, 12176.0, 11870.0, 11563.0, 11258.0],
#     [10926.0, 11302.0, 12018.0, 11925.0, 11592.0, 11251.0, 10914.0],
#     [10322.0, 10704.0, 11511.0, 11423.0, 11053.0, 10687.0, 10321.0],
#     [9583.2, 9984.9, 10858.0, 10744.0, 10348.0, 9957.4, 9568.3],
#     [9000.0, 9200.7, 10105.0, 9989.6, 9568.6, 9141.1, 8729.9]
# ])
#
#
# # Функция для получения удельного импульса
# def get_specific_impulse(height, mach, heights, mach_numbers, specific_impulse):
#     # Создаем объект интерполяции
#     interp_function = RegularGridInterpolator((heights, mach_numbers), specific_impulse)
#
#     # Получаем значение удельного импульса
#     impulse = interp_function((height, mach))
#
#     return impulse
#
#
# # Пример использования
# height = 41  # высота
# mach = 6  # число Маха
#
# impulse = get_specific_impulse(height, mach, heights, mach_numbers, specific_impulse)
# print(f'Удельный импульс на высоте {height} км и числе Маха {mach}: {impulse}')

# def get_specific_impulse(height, mach):
#     mach_numbers = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
#     heights = np.array([2, 5, 11, 25, 30, 35, 40])
#     specific_impulse = np.array([
#         [11300.0, 11493.0, 11848.0, 11729.0, 11554.0, 11363.0, 11000.0],
#         [11492.0, 11760.0, 12322.0, 12230.0, 11969.0, 11700.0, 11430.0],
#         [11338.0, 11610.0, 12253.0, 12176.0, 11870.0, 11563.0, 11258.0],
#         [10926.0, 11302.0, 12018.0, 11925.0, 11592.0, 11251.0, 10914.0],
#         [10322.0, 10704.0, 11511.0, 11423.0, 11053.0, 10687.0, 10321.0],
#         [9583.2, 9984.9, 10858.0, 10744.0, 10348.0, 9957.4, 9568.3],
#         [9000.0, 9200.7, 10105.0, 9989.6, 9568.6, 9141.1, 8729.9]])
#
#     # Проверяем, если выходит за пределы данных
#     if height > heights[-1]:
#         height = heights[-1]
#
#     if mach > mach_numbers[-1]:
#         mach = mach_numbers[-1]
#
#     # Создаем объект интерполяции
#     interp_function = RegularGridInterpolator((heights, mach_numbers), specific_impulse)
#
#     # Получаем значение удельного импульса
#     impulse = interp_function((height, mach))
#     return impulse