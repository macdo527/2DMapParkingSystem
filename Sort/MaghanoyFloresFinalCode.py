from main import stream as PSinfo
import cv2
import threading
import numpy as np
import time
videoframe = None
bl = [75, 1305]
tl = [232, 372]
tr = [1573, 301]
br = [2208, 702]
pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0, 0], [0, 1080], [1920, 0], [1920, 1080]])
B1 = [[400, 255], [237, 417], [332, 508], [500, 340]]
B2 = [[615, 255], [447, 418], [540, 508], [705, 342]]
B3 = [[830, 255], [665, 420], [753, 508], [918, 345]]
B4 = [[1048, 255], [873, 425], [975, 508], [1143, 340]]
B5 = [[1265, 255], [1090, 422], [1195, 508], [1365, 337]]  # test
B6 = [[1480, 253], [1317, 418], [1403, 508], [1568, 345]]
B7 = [[1690, 255], [1513, 427], [1615, 508], [1783, 345]]
A1 = [[330, 545], [152, 717], [253, 803], [427, 632]]
A2 = [[540, 542], [367, 717], [467, 803], [637, 633]]
A3 = [[755, 543], [583, 715], [683, 802], [852, 635]]
A4 = [[968, 545], [798, 717], [898, 803], [1065, 635]]
A5 = [[1182, 548], [1018, 712], [1118, 798], [1278, 640]]
A6 = [[1393, 550], [1220, 723], [1320, 812], [1490, 638]]
A7 = [[1602, 557], [1435, 718], [1537, 805], [1698, 647]]
rectangles = [B1, B2, B3, B4, B5, B6, B7, A1, A2, A3, A4, A5, A6, A7]
matrix = cv2.getPerspectiveTransform(pts1, pts2)
video_path = 'videos/forPVcut.mp4'
frame = cv2.VideoCapture(video_path)
def main():
    global videoframe
    while True:
            map = cv2.imread('Sort/map2.png')
            detectedcars={}
            car_images = {}
            global deletestat
            deletestat=False
            vacant = 14
            begin = time.time()
            if videoframe is None:
                continue
            cardata = PSinfo(videoframe)[1]
            for key, value in cardata.items(): 
                print(detectedcars)
                if value[2] == "Occupied":
                        vacant-=1
                        polypoints = globals()[key]
                        pcx, pcy = getCenter(polypoints[1][0], polypoints[3][0], polypoints[1][1], polypoints[3][1])
                        x,y = getCenter(value[6][0],value[6][2],value[6][1],value[6][3])  
                        dot_coord = np.float32([x,y])
                        new_dot_coord = cv2.perspectiveTransform(np.array([[[dot_coord[0], dot_coord[1]]]], dtype=np.float32), matrix)[0][0]
                        new_dot_coord = np.round(new_dot_coord).astype(np.int32)
                        y_coordinate = 140 if new_dot_coord[1] <= 550 else 940
                        tol = -100 if new_dot_coord[1] <= 550 else 100
                        startcoords = 0 if key in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'] else 1
                        endcoords = 3 if key in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'] else 2
                        ocx, ocy = getCenter(polypoints[startcoords][0], polypoints[endcoords][0], polypoints[startcoords][1], polypoints[endcoords][1])                             
                        index = rectangles.index(polypoints)
                        detectedcars[key] = [key,(pcx,pcy),value[2],value[4],value[5]]     
                        cv2.circle(map,(pcx,pcy),10,(0,0,255),-1)
                        cv2.circle(map,(new_dot_coord[0],new_dot_coord[1]),10,(0,255,0),-1)
                        cv2.polylines(map,[np.array(polypoints,np.int32)],True,(0,0,255),6)
                elif value[2] == "Unoccupied":
                        polypoints = globals()[key]
                        pcx, pcy = getCenter(polypoints[1][0], polypoints[3][0], polypoints[1][1], polypoints[3][1])
                        detectedcars[key] = [key,(pcx,pcy),value[2],value[4],value[5]]
                        cv2.polylines(map,[np.array(polypoints,np.int32)],True,(0,255,0),6)
                elif value[2] == "Unparked":                   
                        pcx,pcy = getCenter(value[6][0],value[6][2],value[6][1],value[6][3])
                        dot_coord = np.float32([pcx,pcy])
                        new_dot_coord = cv2.perspectiveTransform(np.array([[[dot_coord[0], dot_coord[1]]]], dtype=np.float32), matrix)[0][0]
                        new_dot_coord = np.round(new_dot_coord).astype(np.int32)
                        y_coordinate = 140 if new_dot_coord[1] <= 550 else 940
                        tol = -100 if new_dot_coord[1] <= 550 else 100
                        detectedcars[key] = [key,(new_dot_coord[0]-tol,y_coordinate),value[2],value[4],value[5]]
                        cv2.circle(map,(new_dot_coord[0]-tol,y_coordinate),10,(0,255,0),-1)
                        cv2.circle(map,(new_dot_coord[0],new_dot_coord[1]),10,(0,255,0),-1)
            cv2.namedWindow('2D Preview', 0)
            cv2.resizeWindow('2D Preview', 960, 540)
            cv2.imshow('2D Preview', map)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break                           
def preview_video():
    global videoframe
    video = cv2.VideoCapture(video_path)
    while True:
        
        ret, framezz = video.read()
        transformed_frame = cv2.warpPerspective(framezz, matrix, (1920, 1080))
        if not ret:
            break
        videoframe = framezz.copy()
        cv2.namedWindow('Video Preview', 0)
        cv2.resizeWindow('Video Preview', 960, 540)
        cv2.imshow('Video Preview', transformed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def getCenter(x1, x2, y1, y2):
    pcx = int((x1 + x2) / 2)
    pcy = int((y1 + y2) / 2) 
    return pcx, pcy

if __name__ == "__main__":
    APIkey = threading.Thread(target=main, args=())
    video = threading.Thread(target=preview_video, args=())
    APIkey.start()
    video.start()

