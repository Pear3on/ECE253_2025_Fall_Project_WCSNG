import cv2

img = cv2.imread("/home/pearson/Projects/ECE253/Image_process/example_jpg/input/IMG_3583.jpeg", cv2.IMREAD_COLOR)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
cl = clahe.apply(l)

merged = cv2.merge((cl, a, b))
output = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

cv2.imwrite("/home/pearson/Projects/ECE253/Image_process/example_jpg/output/output_clahe_IMG_3583.jpeg", output)