# FLAME face model, modified from https://github.com/radekd91/emoca
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

from .lbs import lbs, batch_rodrigues, vertices2landmarks


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config, flame_full=False):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, 'rb') as f:
            # flame_model = Struct(**pickle.load(f, encoding='latin1'))
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype)) 
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        if not flame_full:
            shapedirs = torch.cat([shapedirs[:, :, :config.n_shape], shapedirs[:, :, 300:300 + config.n_exp]], 2)
        else:
            shapedirs = torch.cat([shapedirs[:, :, :300], shapedirs[:, :, 300:400]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype)) 
        #
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(config.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.tensor(lmk_embeddings['static_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_embeddings['static_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx',
                             torch.tensor(lmk_embeddings['dynamic_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords',
                             torch.tensor(lmk_embeddings['dynamic_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.tensor(lmk_embeddings['full_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('full_lmk_bary_coords',
                             torch.tensor(lmk_embeddings['full_lmk_bary_coords'], dtype=self.dtype))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

        #----------------------------------
        # clip rotation angles within a narrow range if needed
        eye_limits=((-50, 50), (-50, 50), (-0.1, 0.1))
        neck_limits=((-90, 90), (-60, 60), (-80, 80))
        jaw_limits=((-5, 60), (-0.1, 0.1), (-0.1, 0.1))
        global_limits=((-20, 20), (-90, 90), (-20, 20))

        global_limits = torch.tensor(global_limits).float() / 180 * np.pi
        self.register_buffer('global_limits', global_limits)
        neck_limits = torch.tensor(neck_limits).float() / 180 * np.pi
        self.register_buffer('neck_limits', neck_limits)
        jaw_limits = torch.tensor(jaw_limits).float() / 180 * np.pi
        self.register_buffer('jaw_limits', jaw_limits)
        eye_limits = torch.tensor(eye_limits).float() / 180 * np.pi
        self.register_buffer('eye_limits', eye_limits)


    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _apply_rotation_limit(self, rotation, limit):
        r_min, r_max = limit[:, 0].view(1, 3), limit[:, 1].view(1, 3)
        diff = r_max - r_min
        return r_min + (torch.tanh(rotation) + 1) / 2 * diff

    def apply_rotation_limits(self, neck=None, jaw=None):
        """
        method to call for applying rotation limits. Don't use _apply_rotation_limit() in other methods as this
        might cause some bugs if we change which poses are affected by rotation limits. For this reason, in this method,
        all affected poses are limited within one function so that if we add more restricted poses, they can just be
        updated here
        :param neck:
        :param jaw:
        :return:
        """
        neck = self._apply_rotation_limit(neck, self.neck_limits) if neck is not None else None
        jaw = self._apply_rotation_limit(jaw, self.jaw_limits) if jaw is not None else None

        ret = [i for i in [neck, jaw] if i is not None]

        return ret[0] if len(ret) == 1 else ret

    def _revert_rotation_limit(self, rotation, limit):
        """
        inverse function of _apply_rotation_limit()
        from rotation angle vector (rodriguez) -> scalars from -inf ... inf
        :param rotation: tensor of shape N x 3
        :param limit: tensor of shape 3 x 2 (min, max)
        :return:
        """
        r_min, r_max = limit[:, 0].view(1, 3), limit[:, 1].view(1, 3)
        diff = r_max - r_min
        rotation = rotation.clone()
        for i in range(3):
            rotation[:, i] = torch.clip(rotation[:, i],
                                        min=r_min[0, i] + diff[0, i] * .01,
                                        max=r_max[0, i] - diff[0, i] * .01)
        return torch.atanh((rotation - r_min) / diff * 2 - 1)

    def revert_rotation_limits(self, neck, jaw):
        """
        inverse function of apply_rotation_limits()
        from rotation angle vector (rodriguez) -> scalars from -inf ... inf
        :param rotation:
        :param limit:
        :return:
        """
        neck = self._revert_rotation_limit(neck, self.neck_limits)
        jaw = self._revert_rotation_limit(jaw, self.jaw_limits)
        return neck, jaw

    def get_neutral_joint_rotations(self):
        res = {}
        for name, limit in zip(['neck', 'jaw', 'global', 'eyes'],
                               [self.neck_limits, self.jaw_limits,
                                self.global_limits, self.eye_limits]):
            r_min, r_max = limit[:, 0], limit[:, 1]
            diff = r_max - r_min
            res[name] = torch.atanh(-2 * r_min / diff - 1)
            # assert (r_min + (torch.tanh(res[name]) + 1) / 2 * diff) < 1e-7
        return res

    def _pose2rot(self, pose):
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=pose.dtype).view([pose.shape[0], 3, 3])
        
        return rot_mats

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                         self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None, use_rotation_limits=False):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)

        if use_rotation_limits:
            neck_pose, jaw_pose = self.apply_rotation_limits(neck=pose_params[:, :3], jaw=pose_params[:, 3:])
            pose_params = torch.cat([neck_pose,jaw_pose],dim=-1)

            eye_pose_params = torch.cat([self._apply_rotation_limit(eye_pose_params[:, :3], self.eye_limits),
                        self._apply_rotation_limit(eye_pose_params[:, 3:], self.eye_limits)], dim=1)
            
        # set global rotation to zero
        full_pose = torch.cat(
            [torch.zeros_like(pose_params[:, :3]), pose_params[:, :3], pose_params[:, 3:], eye_pose_params], dim=1)
        # full_pose = torch.cat(
        #     [pose_params[:, :3], torch.zeros_like(pose_params[:, :3]), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, J_transformed = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(bz, 1),
                                         self.full_lmk_bary_coords.repeat(bz, 1, 1))

        return vertices, landmarks2d, landmarks3d, J_transformed



class FLAMETex(nn.Module):
    """
    current FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/albedoModel2020_FLAME_albedoPart.npz'
    ## adapted from BFM
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/FLAME_albedo_from_BFM.npz'
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)

        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.flame_tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.

        else:
            print('texture type ', config.tex_type, 'not exist!')
            exit()

        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture

class FLAMETex_trainable(nn.Module):
    """
    current FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/albedoModel2020_FLAME_albedoPart.npz'
    ## adapted from BFM
    tex_path: '/ps/scratch/yfeng/Data/FLAME/texture/FLAME_albedo_from_BFM.npz'
    """

    def __init__(self, config):
        super(FLAMETex_trainable, self).__init__()
        tex_params = config.tex_params
        texture_model = np.load(config.tex_path)

        num_tex_pc = texture_model['PC'].shape[-1]
        tex_shape = texture_model['MU'].shape

        MU = torch.from_numpy(np.reshape(texture_model['MU'], (1, -1))).float()[None, ...]
        PC = torch.from_numpy(np.reshape(texture_model['PC'], (-1, num_tex_pc))[:, :tex_params]).float()[None, ...]
        self.register_buffer('MU', MU)
        self.register_buffer('PC', PC)

        if 'specMU' in texture_model.files:
            specMU = torch.from_numpy(np.reshape(texture_model['specMU'], (1, -1))).float()[None, ...]
            specPC = torch.from_numpy(np.reshape(texture_model['specPC'], (-1, num_tex_pc)))[:, :tex_params].float()[
                None, ...]
            self.register_buffer('specMU', specMU)
            self.register_buffer('specPC', specPC)
            self.isspec = True
        else:
            self.isspec = False

        self.register_parameter('PC_correction', nn.Parameter(torch.zeros_like(PC)))

    def forward(self, texcode):
        diff_albedo = self.MU + (self.PC * texcode[:, None, :]).sum(-1) + (
                    self.PC_correction * texcode[:, None, :]).sum(-1)
        if self.isspec:
            spec_albedo = self.specMU + (self.specPC * texcode[:, None, :]).sum(-1)
            texture = (diff_albedo + spec_albedo)  # torch.pow(0.6*(diff_albedo + spec_albedo), 1.0/2.2)
        else:
            texture = diff_albedo
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture
