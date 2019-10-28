#-------------------------------------------------------------------------------
# Name:     pose_estimation
# Purpose:  ES-EKF pose estimation of vehicle with IMU Lidar and GPS data
#
# Author:      Bo Ji
#
# Created:     25/10/2019

#-------------------------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

## global varibale
#variance
var_imu_f = 0.001
var_imu_w = 0.001
var_bias_f = 0.005
var_bias_w = 0.005
var_gnss  = 0.01
var_lidar = 1
##var_imu_f = 0.004
##var_imu_w = 0.004
##var_bias_f = 0.001
##var_bias_w = 0.001
##var_gnss  = 0.04
##var_lidar = 1
#initial varibale


def getData(path):

    with open(path, 'rb') as file:
        data = pickle.load(file)

    imu_a = data['imu_f']  # acceleration
    imu_w = data['imu_w']  # gyro rate
    lidar = data['lidar']
    gnss  = data['gnss']
    gt    = data['gt']  # groud truth

    return gt,imu_a,imu_w,lidar,gnss

def calibrationRotMatrix():
    #calibration rotation matrix to transform the lidar data to inertial frame
    C_li = np.array([
        [ 0.99376, -0.09722,  0.05466],
        [ 0.09971,  0.99401, -0.04475],
        [-0.04998,  0.04992,  0.9975 ]])

    t_i_li = np.array([0.5, 0.1, 0.5])

    return C_li,t_i_li

def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check, a_bias_check, w_bias_check, g_check):
    # measurement model jacobian
    h_jac = np.zeros([3, 18])
    h_jac[:, :3] = np.eye(3)
    R = sensor_var * np.identity(3)

    #step1 Compute Kalman Gain
    K_k = p_cov_check.dot(h_jac.T).dot(np.linalg.inv(h_jac.dot(p_cov_check).dot(h_jac.T)+R))

    #step2 Compute error state
    error = K_k.dot(y_k-p_check)

    #step3 Correct predicted state
    p_hat = p_check + error[0:3]
    v_hat = v_check + error[3:6]
    phi_del = error[6:9]
    q_hat = Quaternion(euler=phi_del).quat_mult_right(q_check)
    a_bias_hat = a_bias_check + error[9:12]
    w_bias_hat = w_bias_check + error[12:15]
    g_hat = g_check + error[15:18]
    #step4 Compute corrected covariance

    p_cov_hat = (np.identity(18) - K_k.dot(h_jac)).dot(p_cov_check)

    p_conv_hat = 0.5*(p_cov_hat.T+p_cov_hat)

    return p_hat, v_hat, q_hat, a_bias_hat, w_bias_hat, g_hat, p_cov_hat

def plotResult(gt_p,gt_v,gt_r,p_est,v_est,euler_est,a_bias_est,w_bias_est,g_est,time_est,time_gt):
    #pos
    est_traj_fig = plt.figure()
    ax = est_traj_fig.add_subplot(111, projection='3d')
    ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
    ax.plot(gt_p[:,0], gt_p[:,1], gt_p[:,2], label='Ground Truth')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('Ground Truth and Estimated Trajectory')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_zlim(-2, 2)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_zticks([-2, -1, 0, 1, 2])
    ax.legend(loc=(0.62,0.77))
    ax.view_init(elev=45, azim=-50)
    plt.show()

    # velocity
    vel_fig = plt.figure()
    vel_fig.subplots_adjust(hspace=0.6, wspace=0.6)
    ax = vel_fig.add_subplot(311)
    ax.plot(time_est[0:len(v_est)],v_est[:,0], label='Estimated')
    ax.plot(time_gt[0:len(gt_v)],gt_v[:,0],label='Ground Truth')
    ax.set_xlabel('time[ms]')
    ax.set_ylabel('velocity [m/s]')
    ax.set_title('Ground Truth and Estimated velocity X')
    ax.legend(loc=(0.9,.6))
    ay = vel_fig.add_subplot(312)
    ay.plot(time_est[0:len(v_est)],v_est[:,1],label='Estimated')
    ay.plot(time_gt[0:len(gt_v)],gt_v[:,1],label='Ground Truth')
    ay.set_xlabel('time[ms]')
    ay.set_ylabel('velocity [m/s]')
    ay.set_title('Ground Truth and Estimated velocity Y')
    ay.legend(loc=(0.9,.6))
    az = vel_fig.add_subplot(313)
    az.plot(time_est[0:len(v_est)],v_est[:,2],label='Estimated')
    az.plot(time_gt[0:len(gt_v)],gt_v[:,2],label='Ground Truth')
    az.set_xlabel('time[ms]')
    az.set_ylabel('velocity [m/s]')
    az.set_title('Ground Truth and Estimated velocity Z')
    az.legend(loc=(0.9,.6))
    #ax.view_init(elev=45, azim=-50)
    plt.show()

    angle_fig = plt.figure()
    angle_fig.subplots_adjust(hspace=0.6, wspace=0.6)
    ax = angle_fig.add_subplot(311)
    ax.plot(time_est[0:len(euler_est)],euler_est[:,0], label='Estimated')
    ax.plot(time_gt[0:len(gt_r)],180/np.pi*gt_r[:,0],label='Ground Truth')
    ax.set_xlabel('time[ms]')
    ax.set_ylabel('angle [degree]')
    ax.set_title('Ground Truth and Estimated velocity roll')
    ax.legend(loc=(0.9,.6))
    ay = angle_fig.add_subplot(312)
    ay.plot(time_est[0:len(euler_est)],euler_est[:,1], label='Estimated')
    ay.plot(time_gt[0:len(gt_r)],180/np.pi*gt_r[:,1],label='Ground Truth')
    ay.set_xlabel('time[ms]')
    ay.set_ylabel('angle [degree]')
    ay.set_title('Ground Truth and Estimated velocity pitch')
    ay.legend(loc=(0.9,.6))
    az = angle_fig.add_subplot(313)
    az.plot(time_est[0:len(euler_est)],euler_est[:,2], label='Estimated')
    az.plot(time_gt[0:len(gt_r)],180/np.pi*gt_r[:,2],label='Ground Truth')
    az.set_xlabel('time[ms]')
    az.set_ylabel('angle [degree]')
    az.set_title('Ground Truth and Estimated velocity yaw')
    az.legend(loc=(0.9,.6))
    #ax.view_init(elev=45, azim=-50)
    plt.show()


def rmse(gt,est):
    square_error_x = 0
    square_error_y = 0
    square_error_z = 0
    for i in range(gt.shape[0]):
        dx = gt[i,0]- est[i,0]
        dy = gt[i,1]- est[i,1]
        dz = gt[i,2]- est[i,2]
        square_error_x += (dx**2)
        square_error_y += (dy**2)
        square_error_z += (dz**2)
    rmse_x = np.sqrt(square_error_x/gt.shape[0])
    rmse_y = np.sqrt(square_error_y/gt.shape[0])
    rmse_z = np.sqrt(square_error_z/gt.shape[0])
    return rmse_x, rmse_y, rmse_z


def main():
    path = 'data/pt1_data.pkl'

    gt,imu_a,imu_w,lidar,gnss = getData(path)

    C_li,t_i_li = calibrationRotMatrix()
    #transform the lidar data to inertial frame
    lidar.data = (C_li @ lidar.data.T).T + t_i_li

    #lidar index
    lidar_idx = 0
    #gnss index
    gnss_idx = 0

    ##pre_computed matrix
    I = np.identity(3)
    #pertubation matrix
    L_jac = np.zeros((18,12))
    L_jac[3:15,:] = np.identity(12)


    # save the estimated result
    p_est = np.zeros([imu_a.data.shape[0], 3])  # position estimates
    v_est = np.zeros([imu_a.data.shape[0], 3])  # velocity estimates
    q_est = np.zeros([imu_a.data.shape[0], 4])  # orientation estimates as quaternions
    a_bias_est = np.zeros([imu_a.data.shape[0],3]) #acc bias estimation
    w_bias_est = np.zeros([imu_a.data.shape[0],3]) #w bias estimation
    g_est = np.zeros([imu_a.data.shape[0],3])
    p_cov = np.zeros([imu_a.data.shape[0], 18, 18])  # covariance matrices at each timestep

    # intial state
    p_est[0] = gt.p[0]
    v_est[0] = gt.v[0]
    q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
    #bias are initially set to 0
    g = np.array([0, 0, -9.81])  # gravity
    g_est[0] = g
    p_cov[0] = np.zeros(18)  # covariance of estimate

    #start the motion prediction
    for k in range(1,imu_a.data.shape[0]):
        # compute dt
        delta_t = imu_a.t[k]-imu_a.t[k-1]
        # computed transform matrix to the common/navigayion frame
        rotation_matrix = Quaternion(*q_est[k-1]).to_mat()

        # step 1 motion propogation using nonlinear function for normial state
        p_est[k] = p_est[k-1] + v_est[k-1]*delta_t + (0.5*delta_t**2)*(rotation_matrix.dot(imu_a.data[k-1]-a_bias_est[k-1])+g_est[k-1])
        v_est[k] = v_est[k-1] + delta_t*(rotation_matrix.dot(imu_a.data[k-1]-a_bias_est[k-1])+g_est[k-1])
        q_est[k] = Quaternion(axis_angle=(imu_w.data[k-1]-w_bias_est[k-1])*delta_t).quat_mult_right(q_est[k-1])
        a_bias_est[k] = a_bias_est[k-1]
        w_bias_est[k] = w_bias_est[k-1]
        g_est[k] = g_est[k-1]

        #step 2 covariance propogation
        F_m = np.identity(18)
        F_m[0:3,3:6] = delta_t * I
        F_m[3:6,6:9] = -(delta_t*rotation_matrix.dot(skew_symmetric(imu_a.data[k-1]-a_bias_est[k-1])))
        F_m[3:6,9:12] = -(delta_t * rotation_matrix)
        F_m[3:6,15:18] = delta_t * I
        F_m[6:9,12:15] = -(delta_t*rotation_matrix)
        Q = np.zeros((12,12))
        Q[0:3,0:3] = (var_imu_f*delta_t**2) * I
        Q[3:6,3:6] = (var_imu_w*delta_t**2) * I
        Q[6:9,6:9] = (var_bias_f*delta_t**2) * I
        Q[9:12,9:12] = (var_bias_w*delta_t**2) * I

        p_cov[k] = F_m.dot(p_cov[k-1]).dot(F_m.T) + L_jac.dot(Q).dot(L_jac.T)

        #step3 measurement correction

        if lidar_idx < lidar.data.shape[0] and imu_a.t[k-1]==lidar.t[lidar_idx]:
            p_est[k],v_est[k],q_est[k],a_bias_est[k],w_bias_est[k],g_est[k],p_cov[k] = measurement_update(var_lidar,p_cov[k],lidar.data[lidar_idx],p_est[k],v_est[k],q_est[k],a_bias_est[k],w_bias_est[k],g_est[k])
            lidar_idx += 1
        if gnss_idx < gnss.data.shape[0] and imu_a.t[k-1]==gnss.t[gnss_idx]:
            p_est[k],v_est[k],q_est[k],a_bias_est[k],w_bias_est[k],g_est[k],p_cov[k] = measurement_update(var_gnss,p_cov[k],gnss.data[gnss_idx],p_est[k],v_est[k],q_est[k],a_bias_est[k],w_bias_est[k],g_est[k])
            gnss_idx += 1

    #convert from quaternion to euler angle
    euler_est = np.zeros((q_est.shape[0],3))
    for i in range(len(q_est)):
        euler_est[i] = Quaternion(*q_est[i]).to_euler()
    #convert to degree
    euler_est = 180 /np.pi * euler_est

    plotResult(gt.p,gt.v,gt.r,p_est,v_est,euler_est,a_bias_est,w_bias_est,g_est,imu_a.t,gt._t)
    # evaluate rmse of pos
    rmse_px,rmse_py,rmse_pz = rmse(gt.p,p_est)
    print(rmse_px,rmse_py,rmse_pz)
    # evalue rmse of velocity
    rmse_vx,rmse_vy,rmse_vz = rmse(gt.v,v_est)
    print(rmse_vx,rmse_vy,rmse_vz)


if __name__ == '__main__':
    main()
