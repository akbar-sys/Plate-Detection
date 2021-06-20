import cv2

image = cv2.imread('plate1.jpg')

#menampilkan gambar Asli
cv2.imshow("1. Gambar Asli", image)

#menampilkan hasil grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("2. Grayscale", gray)

#menampilkan hasil bilateral filter
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("3. Bilateral Filter", gray)

#menampilkan hasil deteksi tepi
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4. Deteksi Tepi", edged)

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
