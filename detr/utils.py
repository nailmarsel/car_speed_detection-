def inPolygon(x, y, xp, yp, w, h):
    c = 0
    for i in range(len(xp)):
        if (((yp[i] * h <= y and y < yp[i - 1] * h) or (yp[i - 1] * h <= y and y < yp[i] * h)) and
                (x > (xp[i - 1] * w - xp[i] * w) * (y - yp[i] * h) / (yp[i - 1] * h - yp[i] * h) + xp[
                    i] * w)): c = 1 - c
    return c


def crop_image(frame, left_bottom_coords: tuple, right_top_coords: tuple):
    # Вырезаем часть изображения по заданным координатам
    cropped_image = frame[left_bottom_coords[1]:right_top_coords[1], left_bottom_coords[0]:right_top_coords[0]]

    return cropped_image


def calculate_rectangle_area(x1: float, y1: float, x2: float, y2: float) -> float:
    # Расчет длины и ширины прямоугольника
    length = abs(x2 - x1)
    width = abs(y2 - y1)

    # Расчет площади прямоугольника
    area = length * width

    return area