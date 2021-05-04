from matplotlib import pyplot as plt
import numpy as np 
import cv2
import glob
import json

def draw_matches(src1, src2, kp1, kp2, gm, i):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    _1_255 = np.expand_dims( np.array( range( 0, 256 ), dtype='uint8' ), 1 )
    _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

    for i, m in enumerate(gm):
        left = kp1[m.queryIdx].pt
        right = tuple(sum(x) for x in zip(kp2[m.trainIdx].pt, (src1.shape[1], 0)))
        colormap_idx = int( (left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5) ) # manhattan gradient

        color = tuple( map(int, _colormap[ colormap_idx,0,: ]) )
        cv2.circle(output, tuple(map(int, left)), 1, color, 2)
        cv2.circle(output, tuple(map(int, right)), 1, color, 2)


    cv2.imshow('show', output)
    cv2.imwrite("out_sift" + str(i) + ".jpg", output)
    cv2.waitKey()

def MatchFeatures(Im1, Im2, i):
    # https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(Im1,None)
    kp2, des2 = sift.detectAndCompute(Im2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good_matches = []
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            good_matches.append(m)

    Pts1 = []
    Pts2 = []
    for i, m in enumerate(good_matches):
        # print(int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1]), \
        #     '->',int(kp2[m.trainIdx].pt[0]) ,int(kp2[m.trainIdx].pt[1]))
        x1 = int(kp1[m.queryIdx].pt[0])#input
        y1 = int(kp1[m.queryIdx].pt[1])
        x2 = int(kp2[m.trainIdx].pt[0])#template
        y2 = int(kp2[m.trainIdx].pt[1])
        Pts1.append([x1, y1])
        Pts2.append([x2, y2])
        # if i < 0:
        #     fig = plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(Im1)
        #     plt.plot(x1, y1, 'r+')

        #     plt.subplot(1, 2, 2)
        #     plt.imshow(Im2)
        #     plt.plot(x2, y2, 'r+')
        #     plt.show()

    # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(Im1,kp1,Im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # # cv2.imshow("show", img3)
    # cv2.imwrite("out_sift.jpg", img3)
    # cv2.waitKey(0)
    draw_matches(Im1, Im2, kp1, kp2, good_matches, i)

    return Pts1, Pts2


if __name__ == "__main__":
    
    Im1 = cv2.imread("../data/test1.jpeg")
    Im2 = cv2.imread("../data/test2.jpeg")
    
    Pts2, Pts1 = MatchFeatures(Im2, Im1, 0)
    print("Num Matches ", len(Pts2))
    






   