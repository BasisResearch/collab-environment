import numpy as np


def find_angle(first, second):
    print('first ', first)
    print('second ', second)
    print('dot ', np.dot(first, second))
    print('norm first ', np.linalg.norm(first))
    print('norm second ', np.linalg.norm(second))

    if np.linalg.norm(first) != 0.0 and np.linalg.norm(second) != 0.0:
        return np.arccos(np.dot(first, second) /
                         (np.linalg.norm(first) * np.linalg.norm(second)))
    else:
        return 0.0


def calc_angles(first, second):
    theta_x = 0
    theta_y = 0
    first_zero_x = np.array([0.0, first[1], first[2]])
    second_zero_x = np.array([0.0, second[1], second[2]])
    # print("angle = ", angle)
    theta_x = find_angle(first_zero_x, second_zero_x)

    first_zero_y = np.array([first[0], 0.0, first[2]])
    second_zero_y = np.array([second[0], 0.0, second[2]])
    theta_y = find_angle(first_zero_y, second_zero_y)

    first_zero_z = np.array([first[0], first[1], 0.0])
    second_zero_z = np.array([second[0], second[1], 0.0])
    theta_z = find_angle(first_zero_z, second_zero_z)
    print(theta_z)
    return theta_x, theta_y, theta_z
