import numpy as np


def lineGenerator(pointOne, pointTwo):
    """
    :param pointOne: point A
    :param pointTwo: point B
    :return: coefficients of line between A and B in the form of
    ax + b + c = 0
    """
    A = (pointOne[1] - pointTwo[1])
    B = (pointTwo[0] - pointOne[0])
    C = (pointOne[0] * pointTwo[1] - pointTwo[0] * pointOne[1])
    return A / C, B / C, 1


def intersectionGenerator(lineOne, lineTwo):
    """
    :param lineOne: equation of line one
    :param lineTwo: equation of line two
    :return: intersection point of two lines
    """
    x, y, z = np.cross(lineOne, lineTwo)
    return int(x / z), int(y / z)


def verifyPointLocation(point, line):
    """
    :param point: point coordinates
    :param line: equation of line
    :return: if point lies on line or not
    """
    point = np.asarray(point)[:, np.newaxis]
    point = np.vstack((point, 1))
    line = np.asarray(line)[:, np.newaxis]
    valued = np.dot(point.T, line)

    if valued < (1 * e - 3):
        print('Point lies on line')
    else:
        print('Point does not lie on the line')


def distanceCalculator(pointOne, pointTwo):
    """
    :param pointOne: coordinates of first point
    :param pointTwo: coordinates of second point
    :return: distance between both points
    """
    return np.linalg.norm(pointOne - pointTwo)


def acceptInput(string):
    print(string)
    return np.array([int(n) for n in input().split()])


# values from PDF for reference:
# b1 222 43
# b2 389 168
# t1 232 311
# ~t1 344 340
# t2 331 391
# v 247 713
# u' 590 451
# u 767 451


def main():
    b1 = acceptInput('Enter b1')
    print(b1)
    b2 = acceptInput('Enter b2')
    print(b2)

    v = acceptInput('Enter v')
    print(v)

    t1 = acceptInput('Enter t1')
    print(t1)
    t2 = acceptInput('Enter t2')
    print(t2)

    # b1 = np.array([[222, 43]])
    # b2 = np.array([[389, 168]])
    # v = np.array([[247, 713]])
    # t2 = np.array([[331, 391]])
    # t1 = np.array([232, 311])

    b1b2 = lineGenerator(np.squeeze(b1), np.squeeze(b2))

    np.random.seed(2)
    l = lineGenerator([np.random.randint(1, 10), 451],
                      [np.random.randint(21, 30), 451])  # line assumed as the equation is missing

    # calculating vanishing point coordinate
    u = intersectionGenerator(b1b2, l)
    print('vanishing point coordinates are: ', u)

    # generating lines needed for calculating the ratio
    b1v = lineGenerator(np.squeeze(b1), np.squeeze(v))
    print('coefficients of b1v line: ', b1v)
    b2v = lineGenerator(np.squeeze(b2), np.squeeze(v))
    print('coefficients of b2v line: ', b2v)
    t1u = lineGenerator(np.squeeze(t1), np.squeeze(u))
    print('coefficients of t1u line: ', t1u)

    tilde_t1 = intersectionGenerator(t1u, b2v)

    print('~t1: ', tilde_t1)

    # generating distances along the given lines
    vn = distanceCalculator(v, b2)
    tilde_t1n = distanceCalculator(tilde_t1, b2)
    t2n = distanceCalculator(t2, b2)

    print('vn: \n', vn)
    print('tilde_t1n: \n', tilde_t1n)
    print('t2n: \n', t2n)

    # formula adopted from the book page 222
    ratio = (tilde_t1n * (vn - t2n)) / (t2n * (vn - tilde_t1n))
    print('Ratio of the segments ', ratio)


if __name__ == '__main__':
    main()
