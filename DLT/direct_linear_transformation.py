import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def readData(filename):
    """
    :param filename: filename having data
    :return: pointSet1 and pointSet2
    """

    # pointSet1 = np.empkty((25, 2))
    # pointSet2 = np.empty((25, 2))
    temporary = np.empty((25, 2))

    line_counter = 0

    file = open(filename)

    for line in file.readlines():
        # using rstrip to remove \n youtube
        fname = line.rstrip().split(';')

        for i in range(0, len(fname) + 1):
            if i == 0:
                temporary[i] = np.asarray((fname[i][8:].split(',')))

            elif 0 < i < len(fname) - 1:
                temporary[i] = np.asarray((fname[i][1:].split(',')))

            elif i == len(fname) - 1:
                temporary[i] = np.asarray((fname[i][1:-1].split(',')))

        if line_counter == 0:
            pointSet1 = temporary
            line_counter += 1
            temporary = np.empty((25, 2))

        elif line_counter == 1:
            pointSet2 = temporary

    return pointSet1, pointSet2


def normalization(x):
    """
    :param x: input to be normalized
    :return: normalized values
    """
    # centroid calculation
    mean = np.mean(x, 0)

    # scale calculation
    summer = []

    for i in range(0, x.shape[0]):
        summer.append(np.sqrt((x[i, 0] - mean[0]) ** 2 + (x[i, 1] - mean[1]) ** 2))
        summer = np.asarray(summer)
        scale = (np.sqrt(2) / np.mean(summer))
        x_prime = np.hstack((x, np.ones((25, 1))))

        norm = np.array([[scale, 0, -scale * mean[0]],
                         [0, scale, -scale * mean[1]],
                         [0, 0, 1]])

        x_norm = np.dot(norm, x_prime.T).T
        return x_norm, norm


def DLT(pointSet1, pointSet2, norm_input, norm_output):
    """
    :param norm_output: norm factor for pointSet2
    :param norm_input: norm factor for pointSet1
    :param pointSet1: original coordinates normalised
    :param pointSet2: transformed coordinates normalised
    :return: normalised and denormalised homographies
    """
    A = []

    for i in range(pointSet1.shape[0]):
        x, y = pointSet1[i, 0], pointSet1[i, 1]
        x1, y1 = pointSet2[i, 0], pointSet2[i, 1]
        A.append([-x, -y, -1, 0, 0, 0, x * x1, y * x1, x1])  # definition adopted from the class notes
        A.append([0, 0, 0, -x, -y, -1, x * y1, y * y1, y1])

    A = np.asarray(A)
    U, sig, V = np.linalg.svd(A)

    H = V[-1, :] / V[-1, -1]  # calculating the values for the homography matrix

    norm_H = H.reshape(3, 3)

    # denormalizing the homography matrix
    denorm_H = np.dot(np.linalg.inv(norm_output), (np.dot(norm_H, norm_input)))  # definition adopted from book

    return norm_H, denorm_H


def DLTCV(pointSet1, pointSet2, pointSet1norm, pointSet2norm):
    """
    :param pointSet1norm: original coordinates normalised
    :param pointSet2norm: transformed coordinates normalised
    :param pointSet1: original coordinates
    :param pointSet2: transformed coordinates

    :return: homographies computed using opencv functions
    """
    norm_H, norm_status_H = cv2.findHomography(pointSet1norm,
                                               pointSet2norm)
    denorm_H, denorm_status_H = cv2.findHomography(pointSet1,
                                                   pointSet2)
    return norm_H, denorm_H


def errorComputation(manualnormH, manualdenormH, cvnormH, cvdenormH):
    """
    :param manualnormH: manually computed normalised H
    :param manualdenormH: manually computed denormalised H
    :param cvnormH: opencv computation of normalised H
    :param cvdenormH: opencv computation of denormalised H
    :return: relative error % of normalised and denormalised H matrix
    """
    normError = ((cvnormH - manualnormH)/cvnormH)*100
    denormError = ((cvdenormH - manualdenormH)/cvdenormH)*100

    return normError, denormError


def main():
    original_coordinates, transformed_coordinates = readData('homography.txt')
    # print(original_coordinates)
    # print(transformed_coordinates)

    original_coordinates_norm, norm_original = normalization(original_coordinates)
    # print('norm original \n', norm_original)
    transformed_coordinates_norm, norm_transform = normalization(transformed_coordinates)
    # print('norm transform \n', norm_transform)
    manualnormH, manualdenormH = DLT(original_coordinates_norm,
                                     transformed_coordinates_norm,
                                     norm_original,
                                     norm_transform)
    print('manualnormH: \n', manualnormH)
    print('manualdenormH: \n', manualdenormH)

    cvnormH, cvdenormH = DLTCV(original_coordinates,
                               transformed_coordinates,
                               original_coordinates_norm,
                               transformed_coordinates_norm)
    print('cvnormH: \n', cvnormH)
    print('cvdenormH: \n', cvdenormH)

    normalisedError, denormalisedError = errorComputation(manualnormH,
                                                          manualdenormH,
                                                          cvnormH,
                                                          cvdenormH)
    print('normalised error % is: \n', normalisedError)
    print('denormalised error % is: \n', denormalisedError)


if __name__ == '__main__':
    main()
