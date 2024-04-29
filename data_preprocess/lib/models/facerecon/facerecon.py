"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .cropping import align_img
from .bfm import ParametricFaceModel
from .util.pytorch3d import MeshRenderer
import trimesh

class FaceReconModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        )

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, is_train=self.isTrain)

        fov = 2 * np.arctan(112 / 1015) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=0.1, zfar=50, rasterize_size=int(2 * 112)
        )
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def preproces_img(self, im, lm, to_tensor=True):
        # to RGB
        stand_index = np.array([96, 97, 54, 76, 82])
        W,H = im.size
        lm = lm.reshape([-1, 2])
        lm = lm[stand_index,:]
        lm[:, -1] = H - 1 - lm[:, -1]
        trans_params, im, lm, _ = align_img(im, lm)
        if to_tensor:
            im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            lm = torch.tensor(lm).unsqueeze(0)
        return im, lm, trans_params

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        input_img = input['imgs']
        lmdks = input['lms']

        align_img, align_lmdks, trans_params = self.preproces_img(input_img, lmdks)
        align_img = align_img.to(self.device) 

        return align_img, trans_params
    
    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }
    
    def optimize_parameters(self):
        return None

    def forward(self, input):
        self.input_img, trans_params = self.set_input(input)
        output_coeff = self.net_recon(self.input_img)
        pred_coeffs_dict = self.split_coeff(output_coeff)
        pred_coeffs = {key:pred_coeffs_dict[key].cpu().numpy() for key in pred_coeffs_dict}

        self.facemodel.to(self.device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)

        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)

        return pred_coeffs, trans_params


    def save_mesh(self, name):

        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        # mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri)
        mesh.export(name)
    
    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            output_vis_numpy = np.concatenate((input_img_numpy, 
                                output_vis_numpy_raw), axis=-2)
            
            output_vis_numpy = np.clip(output_vis_numpy, 0, 255)

            # self.output_vis = torch.tensor(
            #         output_vis_numpy / 255., dtype=torch.float32
            #     ).permute(0, 3, 1, 2).to(self.device)
            
            self.output_vis = output_vis_numpy



