import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import walk


def extract_orb(orb, img):
    

    # # find the keypoints with ORB
    # kp = orb.detect(img,None)


    # # print(kp)

    # # compute the descriptors with ORB
    # kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    # img2 = cv2.drawKeypoints(img,kp,outImage = None,color=(0,255,0), flags=0)
    # plt.imshow(img2),plt.show()



    # fast = cv2.FastFeatureDetector_create()

    # kp = fast.detect(img,None)

    kp, des = orb.detectAndCompute(img, None)



    pts = cv2.KeyPoint_convert(kp)

    print(pts.shape)


    img2 = cv2.merge([img, img, img])
    cv2.drawKeypoints(img, kp, outImage = img2, color = (255, 0, 0), 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    plt.imshow(img2),plt.show()

def main(image_folder_path):
    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures = 10000,
                        scaleFactor = 1.2,
                        nlevels = 8,
                        edgeThreshold = 31,
                        firstLevel = 0,
                        WTA_K = 4,
                        patchSize = 2,
                        fastThreshold = 20 )

    _, _, filenames = next(walk(image_folder_path))
    for image_file in filenames:
        # img = cv2.imread('IMG_2372.JPG',0)
        img = cv2.imread(image_file, 0)




if __name__ == "__main__":
    main()