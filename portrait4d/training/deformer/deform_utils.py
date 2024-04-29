# utils for deformation field
import torch
from torch.nn.functional import normalize
import pytorch3d
from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
import numpy as np

_DEFAULT_MIN_TRIANGLE_AREA = 5e-3

# return inverse transformation for c2w or w2c matrix
def cam_world_matrix_transform(RT):

    #RT (B,4,4) cam2world matrix or world2cam matrix

    rot = RT[:,:3,:3]
    trans = RT[:,:3,3:]

    inverse_RT = torch.eye(4,device=RT.device).unsqueeze(0).repeat(RT.shape[0], 1, 1)
    inverse_rot = rot.permute(0,2,1)
    inverse_trans = - inverse_rot @ trans
    inverse_RT[:,:3,:3] = inverse_rot
    inverse_RT[:,:3,3:] = inverse_trans

    return inverse_RT

# return new triangle lists for vertices in mask
def get_face_tri(mask, face):
    mask_dict = dict()
    for i in range(mask.shape[0]):
        mask_dict[mask[i]] = i
    face_res = []
    for i in range(face.shape[0]):
        f = face[i]
        if f[0].item() in mask_dict \
            and f[1].item() in mask_dict \
            and f[2].item() in mask_dict:

            face_res.append([
                mask_dict[f[0].item()],
                mask_dict[f[1].item()],
                mask_dict[f[2].item()],
            ])
    return torch.from_numpy(np.array(face_res))

# calculate barycentric coordinates
@torch.no_grad()
def get_barycentric(points,closest_faces):
    # points (P,3) point clouds
    # closest_faces (P,3,3) clostest faces
    v1 = closest_faces[:,0] # (P,3)
    v2 = closest_faces[:,1]
    v3 = closest_faces[:,2]

    vq = points - v1

    r31r31 = torch.sum((v3-v1)**2,dim=-1) #(P,)
    r21r21 = torch.sum((v2-v1)**2,dim=-1)
    r21r31 = torch.sum((v2-v1)*(v3-v1),dim=-1)
    r31vq = torch.sum((v3-v1)*vq,dim=-1)
    r21vq = torch.sum((v2-v1)*vq,dim=-1)

    d = r31r31*r21r21 - r21r31**2
    d = torch.clamp(d, 1e-12)
    bary3 = torch.div(r21r21*r31vq - r21r31*r21vq,d)
    bary2 = torch.div(r31r31*r21vq - r21r31*r31vq,d)
    bary1 = 1. - bary2 - bary3

    bary = torch.stack([bary1,bary2,bary3],dim=-1) #(P,3)

    return bary

# calculate face normal
@torch.no_grad()
def get_face_normal(tris):
    # tris (B,T,3,3) Face vertices
    v1 = tris[:,:,0]
    v2 = tris[:,:,1]
    v3 = tris[:,:,2]
    normals = torch.cross(v2-v1,v3-v1,dim=-1)
    normals = normalize(normals,dim=-1)

    return normals

# calculate vertex normal
@torch.no_grad()
def get_vertex_normal(vts,faces):
    meshes = Meshes(verts=vts,faces=faces)
    # print(meshes.verts_normals_list())
    normals = meshes.verts_normals_list()[0]

    return normals

@torch.no_grad()
def point_to_face_coord(pts,vts,faces):
    # pts: (B,P,3) point clouds in the 3D space
    # vts: (B,N,3) mesh vertices
    # faces: (B,T,3) face indices

    # return:
    # dists： (B,P) point to face distances
    # idxs: (B,P) indices of the closest faces
    # baries: (B,P,3) barycentric coordinates of the projections on the closest faces
    pcls = Pointclouds(points=pts)
    meshes = Meshes(verts=vts,faces=faces)

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    B = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (B*P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()    

    dists, idxs = _C.point_face_dist_forward(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        _DEFAULT_MIN_TRIANGLE_AREA,
    )

    baries = get_barycentric(points,tris[idxs]) # (B*P,3)

    dists = dists.reshape(B,-1) # (B,P)
    idxs = idxs.reshape(B,-1) # (B,P)
    idxs = idxs - tris_first_idx.unsqueeze(-1)
    baries = baries.reshape(B,-1,3) # (B,P,3)

    return dists, idxs, baries


@torch.no_grad()
def point_to_edge_coord(pts,vts,faces):
    # pts: (B,P,3) point clouds in the 3D space
    # vts: (B,N,3) mesh vertices
    # faces: (B,T,3) face indices

    # return:
    # dists： (B,P) point to face distances
    # idxs: (B,P) indices of the closest faces
    # baries: (B,P,3) barycentric coordinates of the projections on the closest faces
    pcls = Pointclouds(points=pts)
    meshes = Meshes(verts=vts,faces=faces)

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    B = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (B*P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for egdes
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.edges_packed()
    segms = verts_packed[edges_packed]  # (S, 2, 3)
    segms_first_idx = meshes.mesh_to_edges_packed_first_idx()
    max_segms = meshes.num_edges_per_mesh().max().item() 

    # print(tris_first_idx)
    dists, idxs = _C.edge_point_dist_forward(
        points, 
        points_first_idx, 
        segms, 
        segms_first_idx, 
        max_segms
    )

    dists = dists.reshape(B,-1) # (B,P)
    idxs = idxs.reshape(B,-1) # (B,P)
    idxs = idxs - segms_first_idx.unsqueeze(-1)
    baries = baries.reshape(B,-1,3) # (B,P,3)

    return dists, idxs

    

