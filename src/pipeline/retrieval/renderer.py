import numpy as np
import pyrender
import torch
import trimesh
from pyrender.constants import RenderFlags
from scipy.spatial.transform import Rotation as Rot

from src.utils.bbox_utils import CropResizePad


class MeshRenderer:
    def __init__(self, n_poses, resolution=420):
        phi = np.sqrt(2.0)
        psi = 1.533751168755204288118041
        
        Q = np.empty(shape=(n_poses,4), dtype=float)
        self.rotations = [] # opencv
        self.mesh_poses = [] # opengl
        for i in range(n_poses):
            s = i+0.5
            r = np.sqrt(s/n_poses)
            R = np.sqrt(1.0-s/n_poses)
            alpha = 2.0 * np.pi * s / phi
            beta = 2.0 * np.pi * s / psi
            Q[i,0] = r*np.sin(alpha)
            Q[i,1] = r*np.cos(alpha)
            Q[i,2] = R*np.sin(beta)
            Q[i,3] = R*np.cos(beta)

            self.rotations.append(Rot.from_quat(Q[i]).as_matrix())

            mesh_pose = np.eye(4)
            mesh_pose[:3, 3] = np.array([0, 0, 1.1])
            mesh_pose[:3, :3] = self.rotations[-1]
            self.mesh_poses.append(mesh_pose)

        self.camera = pyrender.IntrinsicsCamera(fx=600, fy=600, cx=resolution/2, cy=resolution/2)
        self.renderer = pyrender.OffscreenRenderer(resolution, resolution)
        self.opencv2opengl = np.eye(4)
        self.opencv2opengl[1, 1] = -1
        self.opencv2opengl[2, 2] = -1
    
    def render(self, mesh, cull_faces=False):
        if isinstance(mesh, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(mesh)
        elif isinstance(mesh, trimesh.PointCloud):
            colors = mesh.colors

            if colors.size == 0:
                colors = np.ones_like(mesh.vertices) * 255
            mesh = pyrender.Mesh.from_points(mesh.vertices, colors=colors)

        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]), ambient_light=np.array([2.0, 2.0, 2.0, 2.0])
        )

        scene.add(self.camera, pose=self.opencv2opengl)
        cad_node = scene.add(mesh, pose=np.eye(4), name="cad")

        results = []
        for mesh_pose in self.mesh_poses:
            scene.set_pose(cad_node, pose=mesh_pose)
            if cull_faces:
                rgb, depth = self.renderer.render(scene)
            else:
                rgb, depth = self.renderer.render(scene, flags=RenderFlags.SKIP_CULL_FACES)
            results.append((rgb, depth, mesh_pose[:3, :3]))
        return results
    
    def render_from_poses(self, mesh, poses, cull_faces=False):
        if isinstance(mesh, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(mesh)
        elif isinstance(mesh, trimesh.PointCloud):
            colors = mesh.colors

            if colors.size == 0:
                colors = np.ones_like(mesh.vertices) * 255
            mesh = pyrender.Mesh.from_points(mesh.vertices, colors=colors)
        
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]), ambient_light=np.array([2.0, 2.0, 2.0, 2.0])
        )

        scene.add(self.camera, pose=self.opencv2opengl)
        cad_node = scene.add(mesh, pose=np.eye(4), name="cad")

        results = []
        for mesh_pose in poses:
            scene.set_pose(cad_node, pose=mesh_pose)
            if cull_faces:
                rgb, depth = self.renderer.render(scene)
            else:
                rgb, depth = self.renderer.render(scene, flags=RenderFlags.SKIP_CULL_FACES)
            results.append((rgb, depth, mesh_pose))
        return results

    @staticmethod
    def mask_to_bbox(mask):
        # Get the indices of non-zero elements
        y_indices, x_indices = np.nonzero(mask)
        
        # Get the min and max indices for each dimension
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Return the coordinates of the bounding box
        return np.array([x_min, y_min, x_max, y_max])
    
    @staticmethod
    def generate_proposals(res, resolution=420, bbox_extend=0):
        templates, boxes, poses, masks = [], [], [], []
        rgb_proposal_processor = CropResizePad(resolution, (420, 420), bbox_extend=bbox_extend)
        for img, depth, pose in res:
            mask = depth > 0

            if mask.sum() < 100:
                mask[105:315, 105:315] = True

            bbox = MeshRenderer.mask_to_bbox(mask)

            image = torch.from_numpy(img / 255).float()
            templates.append(image)
            boxes.append(bbox)
            poses.append(pose)
            masks.append(mask)

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        templates_croped = rgb_proposal_processor(templates, boxes)
        return templates_croped, poses, masks