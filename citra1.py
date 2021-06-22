import cv2
import numpy as np
import math

image = cv2.imread('plate1.jpg')

#menampilkan gambar Asli
cv2.imshow("1. Gambar Asli", image)

#menampilkan hasil grayscale
r, g, b = cv2.split(image)
image = cv2.merge((r, g, b))
gray = image[:, :, 1]
cv2.imshow("2. Grayscale", gray)
cv2.imwrite("hasil grayscale.jpg", gray)

#menampilkan hasil bilateral filter
grayscale_image = cv2.imread("hasil grayscale.jpg")
def bilateral_filter(image_matrix, window_length=7,sigma_color=25,sigma_space=9,mask_image = None):
    mask_image = np.zeros((image_matrix.shape[0], image_matrix.shape[1]))\
        if mask_image is None else mask_image
    image_matrix = image_matrix.astype(np.int32)

    def limit(x):
        x = 0 if x < 0 else x
        x = 255 if x > 255 else x
        return x
    limit_ufun = np.vectorize(limit, otypes=[np.uint8])
    def look_for_gaussion_table(delta):
        return delta_gaussion_dict[delta]
    def generate_bf_distance(window_length,sigma):
        distance_matrix = np.zeros((window_length,window_length,3))
        left_bias = int(math.floor(-(window_length - 1) / 2))
        right_bias = int(math.floor((window_length - 1) / 2))
        for i in range(left_bias,right_bias+1):
            for j in range(left_bias,right_bias+1):
                distance_matrix[i-left_bias][j-left_bias] = math.exp(-(i**2+j**2)/(2*(sigma**2)))
        return distance_matrix
    delta_gaussion_dict = {i: math.exp(-i ** 2 / (2 *(sigma_color**2))) for i in range(256)}
    look_for_gaussion_table_ufun = np.vectorize(look_for_gaussion_table, otypes=[np.float64])
    bf_distance = generate_bf_distance(window_length,sigma_space)

    margin = int(window_length / 2)
    left_bias = math.floor(-(window_length - 1) / 2)
    right_bias = math.floor((window_length - 1) / 2)
    filter_image = image_matrix.astype(np.float64)

    for i in range(0 + margin, image_matrix.shape[0] - margin):
        for j in range(0 + margin, image_matrix.shape[1] - margin):
            if mask_image[i][j]==0:
                filter_input = image_matrix[i + left_bias:i + right_bias + 1, j + left_bias:j + right_bias + 1]
                bf_value = look_for_gaussion_table_ufun(np.abs(filter_input-image_matrix[i][j]))
                bf_matrix = np.multiply(bf_value, bf_distance)
                bf_matrix = bf_matrix/np.sum(bf_matrix,keepdims=False,axis=(0,1))
                filter_output = np.sum(np.multiply(bf_matrix,filter_input),axis=(0,1))
                filter_image[i][j] = filter_output
    filter_image = limit_ufun(filter_image)
    return filter_image

BF = bilateral_filter(grayscale_image)
cv2.imshow("3. Bilateral Filter", BF)

#menampilkan hasil deteksi tepi
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4. Deteksi Tepi", edged)

#menampilkan hasil Final
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("5. Plate Detected", image)

cv2.waitKey(0)
