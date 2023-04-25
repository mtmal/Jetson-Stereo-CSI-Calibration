#!/usr/bin/python3

import cv2
from cv2 import aruco


aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(5, 11, 1.5, 1.2, aruco_dict)
# rubbish, the size in cm is rubbish. The board resulted in 1.35 and 1.1
imboard = board.draw((1080, 2400))
cv2.imwrite("./chessboard4.png", imboard)
