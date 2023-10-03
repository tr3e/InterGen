import torch
import torch.nn as nn

from utils.utils import *

kinematic_chain = [[0, 2, 5, 8, 11],
                 [0, 1, 4, 7, 10],
                 [0, 3, 6, 9, 12, 15],
                 [9, 14, 17, 19, 21],
                 [9, 13, 16, 18, 20]]

class InterLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(InterLoss, self).__init__()
        self.nb_joints = nb_joints
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.normalizer = MotionNormalizerTorch()

        self.weights = {}
        self.weights["RO"] = 0.01
        self.weights["JA"] = 3
        self.weights["DM"] = 3

        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        else:
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2, -3)) / (mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, motion_pred, motion_gt, mask, timestep_mask):
        B, T = motion_pred.shape[:2]
        self.losses["simple"] = self.seq_masked_mse(motion_pred, motion_gt, mask)
        target = self.normalizer.backward(motion_gt, global_rt=True)
        prediction = self.normalizer.backward(motion_pred, global_rt=True)

        self.pred_g_joints = prediction[..., :self.nb_joints * 3].reshape(B, T, -1, self.nb_joints, 3)
        self.tgt_g_joints = target[..., :self.nb_joints * 3].reshape(B, T, -1, self.nb_joints, 3)

        self.mask = mask
        self.timestep_mask = timestep_mask

        self.forward_distance_map(thresh=1)
        self.forward_joint_affinity(thresh=0.1)
        self.forward_relatvie_rot()
        self.accum_loss()


    def forward_relatvie_rot(self):
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = self.pred_g_joints[..., r_hip, :] - self.pred_g_joints[..., l_hip, :]
        across = across / across.norm(dim=-1, keepdim=True)
        across_gt = self.tgt_g_joints[..., r_hip, :] - self.tgt_g_joints[..., l_hip, :]
        across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

        y_axis = torch.zeros_like(across)
        y_axis[..., 1] = 1

        forward = torch.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward_gt = torch.cross(y_axis, across_gt, axis=-1)
        forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)

        pred_relative_rot = qbetween(forward[..., 0, :], forward[..., 1, :])
        tgt_relative_rot = qbetween(forward_gt[..., 0, :], forward_gt[..., 1, :])

        self.losses["RO"] = self.mix_masked_mse(pred_relative_rot[..., [0, 2]],
                                                            tgt_relative_rot[..., [0, 2]],
                                                            self.mask[..., 0, :], self.timestep_mask) * self.weights["RO"]


    def forward_distance_map(self, thresh):
        pred_g_joints = self.pred_g_joints.reshape(self.mask.shape[:-1] + (-1,))
        tgt_g_joints = self.tgt_g_joints.reshape(self.mask.shape[:-1] + (-1,))

        pred_g_joints1 = pred_g_joints[..., 0:1, :].reshape(-1, self.nb_joints, 3)
        pred_g_joints2 = pred_g_joints[..., 1:2, :].reshape(-1, self.nb_joints, 3)
        tgt_g_joints1 = tgt_g_joints[..., 0:1, :].reshape(-1, self.nb_joints, 3)
        tgt_g_joints2 = tgt_g_joints[..., 1:2, :].reshape(-1, self.nb_joints, 3)

        pred_distance_matrix = torch.cdist(pred_g_joints1.contiguous(), pred_g_joints2).reshape(
            self.mask.shape[:-2] + (1, -1,))
        tgt_distance_matrix = torch.cdist(tgt_g_joints1.contiguous(), tgt_g_joints2).reshape(
            self.mask.shape[:-2] + (1, -1,))

        distance_matrix_mask = (pred_distance_matrix < thresh).float()

        self.losses["DM"] = self.mix_masked_mse(pred_distance_matrix, tgt_distance_matrix,
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask, dm_mask=distance_matrix_mask) * self.weights["DM"]

    def forward_joint_affinity(self, thresh):
        pred_g_joints = self.pred_g_joints.reshape(self.mask.shape[:-1] + (-1,))
        tgt_g_joints = self.tgt_g_joints.reshape(self.mask.shape[:-1] + (-1,))

        pred_g_joints1 = pred_g_joints[..., 0:1, :].reshape(-1, self.nb_joints, 3)
        pred_g_joints2 = pred_g_joints[..., 1:2, :].reshape(-1, self.nb_joints, 3)
        tgt_g_joints1 = tgt_g_joints[..., 0:1, :].reshape(-1, self.nb_joints, 3)
        tgt_g_joints2 = tgt_g_joints[..., 1:2, :].reshape(-1, self.nb_joints, 3)

        pred_distance_matrix = torch.cdist(pred_g_joints1.contiguous(), pred_g_joints2).reshape(
            self.mask.shape[:-2] + (1, -1,))
        tgt_distance_matrix = torch.cdist(tgt_g_joints1.contiguous(), tgt_g_joints2).reshape(
            self.mask.shape[:-2] + (1, -1,))

        distance_matrix_mask = (tgt_distance_matrix < thresh).float()

        self.losses["JA"] = self.mix_masked_mse(pred_distance_matrix, torch.zeros_like(tgt_distance_matrix),
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask, dm_mask=distance_matrix_mask) * self.weights["JA"]

    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses["total"] = loss
        return self.losses



class GeometricLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, name):
        super(GeometricLoss, self).__init__()
        self.name = name
        self.nb_joints = nb_joints
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.normalizer = MotionNormalizerTorch()
        self.fids = [7, 10, 8, 11]

        self.weights = {}
        self.weights["VEL"] = 30
        self.weights["BL"] = 10
        self.weights["FC"] = 30
        self.weights["POSE"] = 1
        self.weights["TR"] = 100

        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)  # [b,t,p,4,1]
        else:
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, motion_pred, motion_gt, mask, timestep_mask):
        B, T = motion_pred.shape[:2]
        # self.losses["simple"] = self.seq_masked_mse(motion_pred, motion_gt, mask)  # * 0.01
        target = self.normalizer.backward(motion_gt, global_rt=True)
        prediction = self.normalizer.backward(motion_pred, global_rt=True)

        self.first_motion_pred =motion_pred[:,0:1]
        self.first_motion_gt =motion_gt[:,0:1]

        self.pred_g_joints = prediction[..., :self.nb_joints * 3].reshape(B, T, self.nb_joints, 3)
        self.tgt_g_joints = target[..., :self.nb_joints * 3].reshape(B, T, self.nb_joints, 3)
        self.mask = mask
        self.timestep_mask = timestep_mask

        self.forward_vel()
        self.forward_bone_length()
        self.forward_contact()
        self.accum_loss()
        # return self.losses["simple"]

    def get_local_positions(self, positions, r_rot):
        '''Local pose'''
        positions[..., 0] -= positions[..., 0:1, 0]
        positions[..., 2] -= positions[..., 0:1, 2]
        '''All pose face Z+'''
        positions = qrot(r_rot[..., None, :].repeat(1, 1, positions.shape[-2], 1), positions)
        return positions

    def forward_local_pose(self):
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

        pred_g_joints = self.pred_g_joints.clone()
        tgt_g_joints = self.tgt_g_joints.clone()

        across = pred_g_joints[..., r_hip, :] - pred_g_joints[..., l_hip, :]
        across = across / across.norm(dim=-1, keepdim=True)
        across_gt = tgt_g_joints[..., r_hip, :] - tgt_g_joints[..., l_hip, :]
        across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

        y_axis = torch.zeros_like(across)
        y_axis[..., 1] = 1

        forward = torch.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward_gt = torch.cross(y_axis, across_gt, axis=-1)
        forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)

        z_axis = torch.zeros_like(forward)
        z_axis[..., 2] = 1
        noise = torch.randn_like(z_axis) *0.0001
        z_axis = z_axis+noise
        z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)


        pred_rot = qbetween(forward, z_axis)
        tgt_rot = qbetween(forward_gt, z_axis)

        B, T, J, D = self.pred_g_joints.shape
        pred_joints = self.get_local_positions(pred_g_joints, pred_rot).reshape(B, T, -1)
        tgt_joints = self.get_local_positions(tgt_g_joints, tgt_rot).reshape(B, T, -1)

        self.losses["POSE_"+self.name] = self.mix_masked_mse(pred_joints, tgt_joints, self.mask, self.timestep_mask) * self.weights["POSE"]

    def forward_vel(self):
        B, T = self.pred_g_joints.shape[:2]

        pred_vel = self.pred_g_joints[:, 1:] - self.pred_g_joints[:, :-1]
        tgt_vel = self.tgt_g_joints[:, 1:] - self.tgt_g_joints[:, :-1]

        pred_vel = pred_vel.reshape(pred_vel.shape[:-2] + (-1,))
        tgt_vel = tgt_vel.reshape(tgt_vel.shape[:-2] + (-1,))

        self.losses["VEL_"+self.name] = self.mix_masked_mse(pred_vel, tgt_vel, self.mask[:, :-1], self.timestep_mask) * self.weights["VEL"]


    def forward_contact(self):

        feet_vel = self.pred_g_joints[:, 1:, self.fids, :] - self.pred_g_joints[:, :-1, self.fids,:]
        feet_h = self.pred_g_joints[:, :-1, self.fids, 1]
        # contact = target[:,:-1,:,-8:-4] # [b,t,p,4]

        contact = self.foot_detect(feet_vel, feet_h, 0.001)

        self.losses["FC_"+self.name] = self.mix_masked_mse(feet_vel, torch.zeros_like(feet_vel), self.mask[:, :-1],
                                                          self.timestep_mask,
                                                          contact) * self.weights["FC"]



    def forward_bone_length(self):
        pred_g_joints = self.pred_g_joints
        tgt_g_joints = self.tgt_g_joints
        pred_bones = []
        tgt_bones = []
        for chain in kinematic_chain:
            for i, joint in enumerate(chain[:-1]):
                pred_bone = (pred_g_joints[..., chain[i], :] - pred_g_joints[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                         keepdim=True)  # [B,T,P,1]
                tgt_bone = (tgt_g_joints[..., chain[i], :] - tgt_g_joints[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                      keepdim=True)
                pred_bones.append(pred_bone)
                tgt_bones.append(tgt_bone)

        pred_bones = torch.cat(pred_bones, dim=-1)
        tgt_bones = torch.cat(tgt_bones, dim=-1)

        self.losses["BL_"+self.name] = self.mix_masked_mse(pred_bones, tgt_bones, self.mask, self.timestep_mask) * self.weights[
            "BL"]


    def forward_traj(self):
        B, T = self.pred_g_joints.shape[:2]

        pred_traj = self.pred_g_joints[..., 0, [0, 2]]
        tgt_g_traj = self.tgt_g_joints[..., 0, [0, 2]]

        self.losses["TR_"+self.name] = self.mix_masked_mse(pred_traj, tgt_g_traj, self.mask, self.timestep_mask) * self.weights["TR"]


    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses[self.name] = loss

    def foot_detect(self, feet_vel, feet_h, thres):
        velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor(
            [0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)

        feet_x = (feet_vel[..., 0]) ** 2
        feet_y = (feet_vel[..., 1]) ** 2
        feet_z = (feet_vel[..., 2]) ** 2

        contact = (((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)).float()
        return contact