import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations_with_replacement
from numpy.linalg import inv

DEBUG = True

R, G, B = 0, 1, 2


def box_filter(I, r):
    M, N = I.shape
    dest = np.zeros((M, N))

    # cumulative sum over Y axis
    sumY = np.cumsum(I, axis=0)
    # difference over Y axis
    dest[:r + 1] = sumY[r: 2 * r + 1]
    dest[r + 1:M - r] = sumY[2 * r + 1:] - sumY[:M - 2 * r - 1]
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1:M - r - 1]

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    # difference over Y axis
    dest[:, :r + 1] = sumX[:, r:2 * r + 1]
    dest[:, r + 1:N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - \
        sumX[:, N - 2 * r - 1:N - r - 1]

    return dest


def guided_filter(I, p, r=40, eps=1e-3):
    M, N = p.shape
    base = box_filter(np.ones((M, N)), r)

    # each channel of I filtered with the mean filter
    means = [box_filter(I[:, :, i], r) / base for i in range(3)]
    # p filtered with the mean filter
    mean_p = box_filter(p, r) / base
    # filter I with p then filter it with the mean filter
    means_IP = [box_filter(I[:, :, i] * p, r) / base for i in range(3)]
    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]

    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = box_filter(
            I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14

    # ECCV10 eq.15
    b = mean_p - a[:, :, R] * means[R] - \
        a[:, :, G] * means[G] - a[:, :, B] * means[B]

    # ECCV10 eq.16
    q = (box_filter(a[:, :, R], r) * I[:, :, R] + box_filter(a[:, :, G], r) *
         I[:, :, G] + box_filter(a[:, :, B], r) * I[:, :, B] + box_filter(b, r)) / base

    return q


def get_dark_channel(I, w):
    M, N, _ = I.shape
    padded = np.pad(I, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    dark_channel = np.zeros((M, N))
    for i, j in np.ndindex(dark_channel.shape):
        dark_channel[i, j] = np.min(
            padded[i:i + w, j:j + w, :])
    return dark_channel


def get_atomspheric_light(I, dark_channel, p):
    M, N = dark_channel.shape
    flatI = I.reshape(M * N, 3)
    flatdark = dark_channel.ravel()
    # find top M * N * p indexes
    searchidx = (-flatdark).argsort()[:int(M * N * p)]

    if DEBUG:
        print('atmospheric light region:', [
              (i // N, i % N) for i in searchidx])

    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


def get_transmission(I, A, omega, w):
    return 1 - omega * get_dark_channel(I / A, w)


def get_radiance(I, A, t):
    img = np.zeros_like(I)  # tiled to M * N * 3
    img[:, :, R] = img[:, :, G] = img[:, :, B] = t
    # print(img[:, :, B])
    # print(t)
    return (I - A) / img + A  # CVPR09, eq.16


def dehaze(I, tmin=0.2, Amax=220, w=15, p=0.0001,
           omega=0.95, guided=True, r=40, eps=1e-3):
    dark_channel = get_dark_channel(I, w)

    if DEBUG:
        plt.figure('dark channel')
        plt.imshow(dark_channel, cmap='gray')

    A = get_atomspheric_light(I, dark_channel, p)
    A = np.minimum(A, Amax)

    if DEBUG:
        print('atmosphere', A)

    raw_transmission = get_transmission(I, A, omega, w)

    raw_transmission = refined_transmission = np.maximum(
        raw_transmission, tmin)

    if guided:
        normI = (I - I.min()) / (I.max() - I.min())  # normalize I
        refined_transmission = guided_filter(
            normI, refined_transmission, r, eps)

    if DEBUG:
        print('refined transmission rate')
        print('between [%.4f, %.4f]' %
              (refined_transmission.min(), refined_transmission.max()))

    # white = np.full_like(dark_channel, 255)
    # return get_radiance(I, A, refined_transmission)
    radiance = get_radiance(I, A, refined_transmission)

    def to_img(radiance):

        if DEBUG:
            print('Range for each channel:')
            for ch in range(3):
                print('[%.2f, %.2f]' %
                      (radiance[:, :, ch].max(), radiance[:, :, ch].min()))

        return np.maximum(np.minimum(radiance, 255), 0).astype(np.uint8)

    return [to_img(radiance) for radiance in (
        get_radiance(I, A, raw_transmission),
        get_radiance(I, A, refined_transmission)
    )]


def main():
    img = io.imread('img/ny1.png')

    I = np.asarray(img, dtype=np.float64)

    result_raw, result_refined = dehaze(I)
    plt.figure('origin')
    plt.imshow(img)

    plt.figure('result')
    plt.imshow(result_raw)

    plt.figure('refined')
    plt.imshow(result_refined)

    plt.show()


if __name__ == '__main__':
    main()
