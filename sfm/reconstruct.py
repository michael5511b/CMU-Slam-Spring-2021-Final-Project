'''
    Appended reconstruction pipeline to modified gms based feature matching
'''
import math
from enum import Enum
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy import optimize
from sift_matcher import *
cv2.ocl.setUseOpenCL(False)

input_data_path_base = "../SfM_quality_evaluation/Benchmarking_Camera_Calibration_2008/Herz-Jesus-P8/"
input_data_path = input_data_path_base + "images/"

input_data_path = "custom/"
# K = np.genfromtxt(fname=input_data_path+"K.txt")
K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])

THRESHOLD_FACTOR = 6

ROTATION_PATTERNS = [
    [1, 2, 3,
     4, 5, 6,
     7, 8, 9],

    [4, 1, 2,
     7, 5, 3,
     8, 9, 6],

    [7, 4, 1,
     8, 5, 2,
     9, 6, 3],

    [8, 7, 4,
     9, 5, 1,
     6, 3, 2],

    [9, 8, 7,
     6, 5, 4,
     3, 2, 1],

    [6, 9, 8,
     3, 5, 7,
     2, 1, 4],

    [3, 6, 9,
     2, 5, 8,
     1, 4, 7],

    [2, 3, 6,
     1, 5, 9,
     4, 7, 8]]

class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


class GmsMatcher:
    def __init__(self, descriptor, matcher):
        self.scale_ratios = [1.0, 1.0 / 2, 1.0 / math.sqrt(2.0), math.sqrt(2.0), 2.0]
        # Normalized vectors of 2D points
        self.normalized_points1 = []
        self.normalized_points2 = []
        # Matches - list of pairs representing numbers
        self.matches = []
        self.matches_number = 0
        # Grid Size
        self.grid_size_right = Size(0, 0)
        self.grid_number_right = 0
        # x      : left grid idx
        # y      :  right grid idx
        # value  : how many matches from idx_left to idx_right
        self.motion_statistics = []

        self.number_of_points_per_cell_left = []
        # Inldex  : grid_idx_left
        # Value   : grid_idx_right
        self.cell_pairs = []

        # Every Matches has a cell-pair
        # first  : grid_idx_left
        # second : grid_idx_right
        self.match_pairs = []

        # Inlier Mask for output
        self.inlier_mask = []
        self.grid_neighbor_right = []

        # Grid initialize
        self.grid_size_left = Size(20, 20)
        self.grid_number_left = self.grid_size_left.width * self.grid_size_left.height

        # Initialize the neihbor of left grid
        self.grid_neighbor_left = np.zeros((self.grid_number_left, 9))

        self.descriptor = descriptor
        self.matcher = matcher
        self.gms_matches = []
        self.keypoints_image1 = []
        self.keypoints_image2 = []

    def empty_matches(self):
        self.normalized_points1 = []
        self.normalized_points2 = []
        self.matches = []
        self.gms_matches = []

    def compute_matches(self, img1, img2):
        self.keypoints_image1, descriptors_image1 = self.descriptor.detectAndCompute(img1, np.array([]))
        self.keypoints_image2, descriptors_image2 = self.descriptor.detectAndCompute(img2, np.array([]))
        size1 = Size(img1.shape[1], img1.shape[0])
        size2 = Size(img2.shape[1], img2.shape[0])

        if self.gms_matches:
            self.empty_matches()

        all_matches = self.matcher.match(descriptors_image1, descriptors_image2)
        self.normalize_points(self.keypoints_image1, size1, self.normalized_points1)
        self.normalize_points(self.keypoints_image2, size2, self.normalized_points2)
        self.matches_number = len(all_matches)
        self.convert_matches(all_matches, self.matches)
        self.initialize_neighbours(self.grid_neighbor_left, self.grid_size_left)

        mask, num_inliers = self.get_inlier_mask(False, False)
        print('Found', num_inliers, 'matches')

        for i in range(len(mask)):
            if mask[i]:
                self.gms_matches.append(all_matches[i])
        return self.gms_matches

    # Normalize Key points to range (0-1)
    def normalize_points(self, kp, size, npts):
        for keypoint in kp:
            npts.append((keypoint.pt[0] / size.width, keypoint.pt[1] / size.height))

    # Convert OpenCV match to list of tuples
    def convert_matches(self, vd_matches, v_matches):
        for match in vd_matches:
            v_matches.append((match.queryIdx, match.trainIdx))

    def initialize_neighbours(self, neighbor, grid_size):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_size)

    def get_nb9(self, idx, grid_size):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_size.width
        idx_y = idx // grid_size.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_size.width or idx_yy < 0 or idx_yy >= grid_size.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_size.width

        return nb9

    def get_inlier_mask(self, with_scale, with_rotation):
        max_inlier = 0
        self.set_scale(0)

        if not with_scale and not with_rotation:
            max_inlier = self.run(1)
            return self.inlier_mask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                for rotation_type in range(1, 9):
                    num_inlier = self.run(rotation_type)
                    if num_inlier > max_inlier:
                        vb_inliers = self.inlier_mask
                        max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        elif with_rotation and not with_scale:
            vb_inliers = []
            for rotation_type in range(1, 9):
                num_inlier = self.run(rotation_type)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        else:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                num_inlier = self.run(1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier

    def set_scale(self, scale):
        self.grid_size_right.width = self.grid_size_left.width * self.scale_ratios[scale]
        self.grid_size_right.height = self.grid_size_left.height * self.scale_ratios[scale]
        self.grid_number_right = self.grid_size_right.width * self.grid_size_right.height

        # Initialize the neighbour of right grid
        self.grid_neighbor_right = np.zeros((int(self.grid_number_right), 9))
        self.initialize_neighbours(self.grid_neighbor_right, self.grid_size_right)

    def run(self, rotation_type):
        self.inlier_mask = [False for _ in range(self.matches_number)]

        # Initialize motion statistics
        self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
        self.match_pairs = [[0, 0] for _ in range(self.matches_number)]

        for GridType in range(1, 5):
            self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
            self.cell_pairs = [-1 for _ in range(self.grid_number_left)]
            self.number_of_points_per_cell_left = [0 for _ in range(self.grid_number_left)]

            self.assign_match_pairs(GridType)
            self.verify_cell_pairs(rotation_type)

            # Mark inliers
            for i in range(self.matches_number):
                if self.cell_pairs[int(self.match_pairs[i][0])] == self.match_pairs[i][1]:
                    self.inlier_mask[i] = True

        return sum(self.inlier_mask)

    def assign_match_pairs(self, grid_type):
        for i in range(self.matches_number):
            lp = self.normalized_points1[self.matches[i][0]]
            rp = self.normalized_points2[self.matches[i][1]]
            lgidx = self.match_pairs[i][0] = self.get_grid_index_left(lp, grid_type)

            if grid_type == 1:
                rgidx = self.match_pairs[i][1] = self.get_grid_index_right(rp)
            else:
                rgidx = self.match_pairs[i][1]

            if lgidx < 0 or rgidx < 0:
                continue
            self.motion_statistics[int(lgidx)][int(rgidx)] += 1
            self.number_of_points_per_cell_left[int(lgidx)] += 1

    def get_grid_index_left(self, pt, type_of_grid):
        x = pt[0] * self.grid_size_left.width
        y = pt[1] * self.grid_size_left.height

        if type_of_grid == 2:
            x += 0.5
        elif type_of_grid == 3:
            y += 0.5
        elif type_of_grid == 4:
            x += 0.5
            y += 0.5

        x = math.floor(x)
        y = math.floor(y)

        if x >= self.grid_size_left.width or y >= self.grid_size_left.height:
            return -1
        return x + y * self.grid_size_left.width

    def get_grid_index_right(self, pt):
        x = int(math.floor(pt[0] * self.grid_size_right.width))
        y = int(math.floor(pt[1] * self.grid_size_right.height))
        return x + y * self.grid_size_right.width

    def verify_cell_pairs(self, rotation_type):
        current_rotation_pattern = ROTATION_PATTERNS[rotation_type - 1]

        for i in range(self.grid_number_left):
            if sum(self.motion_statistics[i]) == 0:
                self.cell_pairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.grid_number_right)):
                value = self.motion_statistics[i]
                if value[j] > max_number:
                    self.cell_pairs[i] = j
                    max_number = value[j]

            idx_grid_rt = self.cell_pairs[i]
            nb9_lt = self.grid_neighbor_left[i]
            nb9_rt = self.grid_neighbor_right[idx_grid_rt]
            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = nb9_lt[j]
                rr = nb9_rt[current_rotation_pattern[j] - 1]
                if ll == -1 or rr == -1:
                    continue

                score += self.motion_statistics[int(ll), int(rr)]
                thresh += self.number_of_points_per_cell_left[int(ll)]
                numpair += 1

            thresh = THRESHOLD_FACTOR * math.sqrt(thresh/numpair)
            if score < thresh:
                self.cell_pairs[i] = -2

    def draw_matches(self, src1, src2, drawing_type, i):
        height = max(src1.shape[0], src2.shape[0])
        width = src1.shape[1] + src2.shape[1]
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:src1.shape[0], 0:src1.shape[1]] = src1
        output[0:src2.shape[0], src1.shape[1]:] = src2[:]

        if drawing_type == DrawingType.ONLY_LINES:
            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

        elif drawing_type == DrawingType.LINES_AND_POINTS:
            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
                cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

        elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY :
            _1_255 = np.expand_dims( np.array( range( 0, 256 ), dtype='uint8' ), 1 )
            _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))

                if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                    colormap_idx = int(left[0] * 256. / src1.shape[1] ) # x-gradient
                if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                    colormap_idx = int(left[1] * 256. / src1.shape[0] ) # y-gradient
                if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                    colormap_idx = int( (left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5) ) # manhattan gradient

                color = tuple( map(int, _colormap[ colormap_idx,0,: ]) )
                cv2.circle(output, tuple(map(int, left)), 1, color, 2)
                cv2.circle(output, tuple(map(int, right)), 1, color, 2)


        cv2.imshow('show', output)
        cv2.imwrite("out_gms" + str(i) + ".jpg", output)
        cv2.waitKey()

def form_projection_matrix(R, t):
    # to return the 4x4 projection matrix for easy mul
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat

def pts2ply(pts,colors,filename='out.ply'): 
    """Saves an ndarray of 3D coordinates (in meshlab format)"""

    with open(filename,'w') as f: 
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))
        
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        
        f.write('end_header\n')
        
        #pdb.set_trace()
        colors = (255*colors).astype(int)
        for pt, cl in zip(pts,colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                cl[0],cl[1],cl[2]))

def rodrigues(r):
    theta = np.sqrt(np.sum(np.square(r)))
    if(theta == 0):
    	R = np.eye(3)
    else:
    	u = r / theta
    	ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    	R = np.eye(3) * np.cos(theta) + ((1 - np.cos(theta)) * np.matmul(u, u.T)) + np.sin(theta) * ux

    return R

def invRodrigues(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if(theta != 0):
    	omega = (0.5 / np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    	omega = omega.reshape(3, 1)
    	r = omega * theta
    else:
    	r = np.zeros((3, 1))

    return r

def rodriguesResidual(K1, M1, p1, K2, p2, x):
    w = x[0 : -6]
    w = w.reshape(p1.shape[0], 3)
    W = np.vstack((w.T, np.ones((1, p1.shape[0]))))

    R2 = rodrigues(x[-6 : -3])
    M2 = np.hstack((R2, x[-3 :].reshape(3, 1)))

    p1_hat = np.matmul(np.matmul(K1, M1), W)#3 * N
    p1_hat = p1_hat[0 : 2, :] / p1_hat[2, :]

    p2_hat = np.matmul(np.matmul(K2, M2), W)#3 * N
    p2_hat = p2_hat[0 : 2, :] / p2_hat[2, :]

    residuals = np.concatenate([(p1 - p1_hat.T).reshape([-1]), (p2 - p2_hat.T).reshape([-1])])

    return residuals.reshape(-1, 1)

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    N = p1.shape[0]
    r_init = invRodrigues(M2_init[0 : 3, 0 : 3])
    r_init = r_init.reshape(3, 1)
    t_init = M2_init[0 : 3, 3].reshape(3, 1)

    initialx = np.append(P_init, np.array([r_init, t_init]))
    # print(P_init.shape)

    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()

    res = optimize.minimize(func, initialx, method='L-BFGS-B')

    result = res.x
    w = result[0 : -6].reshape(N, 3)
    R2 = rodrigues(result[-6 : -3])
    M2 = np.hstack((R2, result[-3 :].reshape(3, 1)))

    return M2, w

def calc_reprojection_error(M2, points3d, points2):
    p2 = M2 @ points3d.T
    p2 /= p2[2]
    reproj = np.mean(np.sum((p2[:2].T - points2)**2, axis = 1))
    print(reproj)
    return reproj


if __name__ == '__main__':

    images_list = glob.glob(input_data_path + '*.jpeg')
    images_list.sort()
    # images_list = images_list[::20]

    global_points = np.empty((0,4))
    all_colors = np.empty((0,3))


    curr_pose = np.eye(4)
    
    for i in range(0, len(images_list) - 1):
        img1 = cv2.imread(images_list[i])
        img2 = cv2.imread(images_list[i+1])

        ########## find feat corres
        # GMS
        '''
        orb = cv2.ORB_create(10000)
        orb.setFastThreshold(0)
        if cv2.__version__.startswith('3'):
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        gms = GmsMatcher(orb, matcher)

        matches = gms.compute_matches(img1, img2)
        
        points1 = [gms.keypoints_image1[mat.queryIdx].pt for mat in matches] 
        points2 = [gms.keypoints_image2[mat.trainIdx].pt for mat in matches]
        # corres = np.hstack((np.asarray(points1), np.asarray(points2)))
        # all_corres.append(corres)

        # gms.draw_matches(img1, img2, DrawingType.ONLY_LINES)
        gms.draw_matches(img1, img2, DrawingType.COLOR_CODED_POINTS_XpY, i)
        '''
        points1, points2 = MatchFeatures(img1, img2, i)

        ########## Find essential matrix
        points1 = np.int32(points1)
        points2 = np.int32(points2)
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)
        points1 = points1[mask.ravel()==1]
        points2 = points2[mask.ravel()==1]
        E = K.T @ F @ K
        # decompose to find R t
        _, R, t, _ = cv2.recoverPose(E, points1, points2)
        # p1 = np.array([points1[0,0], points1[0,1], 1])
        # p2 = np.array([points2[0,0], points2[0,1], 1])
        # print(p1 @ F @ p2.T)

        # Camera matrices K @ [R | t] # local frame (frame - frame reconstruction)
        M2 = K @ np.hstack((R, t))
        M1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        # curr_pose = form_projection_matrix(R, t) @ form_projection_matrix(M1[:3,:3], M1[:,3])
        # print(M2, M1)

        # triangulate 
        # @todo: convert to global frame
        
        # points1 = np.float32(points1)
        # points2 = np.float32(points2)
        # # p1 = np.linalg.inv(K) @ np.vstack((points1.T, np.ones((1, points1.shape[0]))))
        # # p2 = np.linalg.inv(K) @ np.vstack((points2.T, np.ones((1, points1.shape[0]))))
        # points3d = cv2.triangulatePoints(M1, M2, points1.T, points2.T)
        # points3d /= points3d[3]
        # points3d = points3d.T
        
        
        NumPoints = points1.shape[0]
        points3d = []
        Color = []
        valid_indices = []
        for Point in range(NumPoints):
            A = np.zeros((4, 4))
            A[0] = points1[Point, 1] * M1[2] - M1[1]
            A[1] = M1[0] - points1[Point, 0] * M1[2]
            A[2] = points2[Point, 1] * M2[2] - M2[1]
            A[3] = M2[0] - points2[Point, 0] * M2[2]
            U, D, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X /= X[3]
            if(abs(X[0]) > 10 or abs(X[1]) > 10 or abs(X[2]) > 10):
                continue
            Color.append(img1[points1[Point, 1], points1[Point, 0]] / 255.0)
            points3d.append(X)
            valid_indices.append(Point)

            # print(X)

        points3d = np.array(points3d)
        feat2d1 = points1[valid_indices]
        feat2d2 = points2[valid_indices]
        
        err = calc_reprojection_error(M2, points3d, feat2d2)
        # print(form_projection_matrix(R, t), curr_pose)
        extrinsics2 = np.hstack((R, t))
        extrinsics1 = np.hstack((np.eye(3), np.zeros((3,1))))
        # print(points3d)
        # extrinsics2, points3d = bundleAdjustment(K, extrinsics1, feat2d1, K, extrinsics2, feat2d2, points3d[:, :3])
        # points3d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
        # # # print(points3d)
        # calc_reprojection_error(K @ extrinsics2, points3d, feat2d2)

        points3d = curr_pose @ points3d.T
        points3d = points3d.T
        curr_pose = np.linalg.inv(form_projection_matrix(extrinsics2[:3,:3], extrinsics2[:3,3]) @ curr_pose)

        if err < np.inf:
            global_points = np.vstack((global_points, points3d))
            all_colors = np.vstack((all_colors, Color))

    # coz im lazy to figure out how to use np transpose :p
    rgb = np.zeros_like(all_colors)
    rgb[:, 0] = all_colors[:, 2] 
    rgb[:, 1] = all_colors[:, 1] 
    rgb[:, 2] = all_colors[:, 0] 
    # print(rgb)
    pts2ply(global_points, rgb, "out.ply")

    dsf = 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(global_points[::dsf, 0], global_points[::dsf, 1], global_points[::dsf, 2], c = rgb[::dsf])
    plt.show()