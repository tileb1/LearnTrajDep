"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
from utils import forward_kinematics as fk
from PIL import Image


class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", label=['GT', 'Pred']):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, c='black', label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, c='black'))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='red', label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='red'))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.legend(loc='lower left')
        self.ax.view_init(120, -90)

    def update(self, gt_channels, pred_channels):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 96, "channels should have 96 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (32, -1))
        lcolor = "black"
        rcolor = "black"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 96, "channels should have 96 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (32, -1))
        lcolor = "red"
        rcolor = "red"
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)

        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        # self.ax.set_aspect('equal')


def plot_predictions(expmap_gt, expmap_pred, fig, ax, f_title):
    # Load all the data
    parent, offset, rotInd, expmapInd = fk._some_variables()

    nframes_pred = expmap_pred.shape[0]

    # Compute 3d points for each frame
    xyz_gt = np.zeros((nframes_pred, 96))
    for i in range(nframes_pred):
        xyz_gt[i, :] = fk.fkl(expmap_gt[i, :], parent, offset, rotInd, expmapInd).reshape([96])
    xyz_pred = np.zeros((nframes_pred, 96))
    for i in range(nframes_pred):
        xyz_pred[i, :] = fk.fkl(expmap_pred[i, :], parent, offset, rotInd, expmapInd).reshape([96])

    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    for i in range(nframes_pred):

        ob.update(xyz_gt[i, :], xyz_pred[i, :])
        # ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.show(block=False)

        fig.canvas.draw()
        # plt.pause(0.05)


frame_index_lst = [0]
def plot_predictions2(expmap_gt, expmap_pred, fig, ax, f_title, mao_pred=None, action='', gt=True):
    nframes_pred = expmap_pred.shape[0]

    xyz_gt = expmap_gt
    xyz_pred = expmap_pred

    # === Plot and animate ===
    ob = Ax3DPose2(ax)
    # Plot the prediction
    frame_index_lst[0] = 0
    for i in range(nframes_pred):
        ob = Ax3DPose2(ax)
        ob.update2(xyz_gt[i, :], xyz_pred[i, :], mao_pred[i, :])
        # ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        plt.show(block=False)

        fig.canvas.draw()
        plt.savefig('./fig/img-' + str(frame_index_lst[0]).zfill(3) + '.png')

        pic = Image.open('./fig/img-' + str(frame_index_lst[0]).zfill(3) + '.png')
        A_img = np.asarray(pic)
        A_img = A_img[120:-120, 210:-190, :]
        im = Image.fromarray(A_img)
        im.save('fig/{}{}_gt_{}.png'.format(action, frame_index_lst[0], str(gt)))
        frame_index_lst[0] += 1
        # plt.pause(0.05)
        ax.cla()


class Ax3DPose2(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", label=['GT', 'Ours', 'Mao et al.']):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """
        self.frame_index = 0

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, c='black'))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, c='black'))

        # self.plots_pred_mao = []
        # for i in np.arange(len(self.I)):
        #     x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
        #     y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
        #     z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
        #     if i == 0:
        #         self.plots_pred_mao.append(self.ax.plot(x, y, z, lw=2, c='blue', label=label[2]))
        #     else:
        #         self.plots_pred_mao.append(self.ax.plot(x, y, z, lw=2, c='blue'))

        # self.plots_pred = []
        # for i in np.arange(len(self.I)):
        #     x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
        #     y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
        #     z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
        #     if i == 0:
        #         self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='red', label=label[1]))
        #     else:
        #         self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='red'))


        # self.ax.set_xlabel("x")
        # self.ax.set_ylabel("y")
        # self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.legend(loc='lower left')
        self.ax.view_init(120, -90)

        # Get rid of the ticks and tick labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        self.ax.get_xaxis().set_ticklabels([])
        self.ax.get_yaxis().set_ticklabels([])
        self.ax.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        self.ax.w_xaxis.set_pane_color(white)
        self.ax.w_yaxis.set_pane_color(white)
        self.ax.w_zaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        self.ax.w_xaxis.line.set_color(white)
        self.ax.w_yaxis.line.set_color(white)
        self.ax.w_zaxis.line.set_color(white)

    def update2(self, gt_channels, pred_channels, pred_channels_mao):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """
        self.frame_index += 1
        assert gt_channels.size == 96, "channels should have 96 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (32, -1))
        lcolor = "red"
        rcolor = "red"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)
        for index in [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]:
            joint = gt_vals[index]
            self.ax.scatter(joint[0], joint[1], joint[2], c='black', zorder=2, s=5)

        # assert pred_channels_mao.size == 96, "channels should have 96 entries, it has %d instead" % pred_channels_mao.size
        # pred_vals = np.reshape(pred_channels_mao, (32, -1))
        # lcolor = "blue"
        # rcolor = "blue"
        # for i in np.arange(len(self.I)):
        #     x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
        #     y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
        #     z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
        #     self.plots_pred_mao[i][0].set_xdata(x)
        #     self.plots_pred_mao[i][0].set_ydata(y)
        #     self.plots_pred_mao[i][0].set_3d_properties(z)
        #     self.plots_pred_mao[i][0].set_color(lcolor if self.LR[i] else rcolor)
        #     self.plots_pred_mao[i][0].set_alpha(0.7)
        #
        # assert pred_channels.size == 96, "channels should have 96 entries, it has %d instead" % pred_channels.size
        # pred_vals = np.reshape(pred_channels, (32, -1))
        # lcolor = "red"
        # rcolor = "red"
        # for i in np.arange(len(self.I)):
        #     x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
        #     y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
        #     z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
        #     self.plots_pred[i][0].set_xdata(x)
        #     self.plots_pred[i][0].set_ydata(y)
        #     self.plots_pred[i][0].set_3d_properties(z)
        #     self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
        #     self.plots_pred[i][0].set_alpha(0.7)


        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        # self.ax.set_aspect('equal')