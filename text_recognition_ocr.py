import cv2
import numpy as np
from pyzbar.pyzbar import decode

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def decode_qr_code(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Binary Mask", binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    if len(approx) == 4:
        pts1 = approx.reshape(4, 2).astype('float32')
    else:
        print("未能找到二维码的四个角点。")
        return image, image, binary
    
    original_image = cv2.imread('qrcode.jpg')
    warped = four_point_transform(original_image, pts1)
    
    cv2.imwrite('warped_qrcode.jpg', warped)
    
    decoded_objs = decode(warped)
    for obj in decoded_objs:
        print("二维码内容:", obj.data.decode('utf-8'))
    
    for obj in decoded_objs:
        pts = obj.polygon
        if len(pts) == 4:
            pts = np.array(pts, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(warped, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    
    return image, warped, binary

image = cv2.imread('qrcode.jpg')
original_with_bbox, result_image, binary_image = decode_qr_code(image)

cv2.imshow("Original with Bounding Box", original_with_bbox)
cv2.imshow("Warped Result", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
