import numpy as np
from bop_toolkit_lib import inout
import unittest
from pathlib import Path
from typing import Dict

"""
Compare inout.load_ply with inout.load_mesh (trimesh based). 
To run test_load_ply_vs_obj, export your test .ply file as an .obj using meshlab or trimesh.
"""

EXAMPLE_PLY_PATH = Path('replace/with/proper/absolute/path/obj_000002.ply')
EXAMPLE_OBJ_PATH = Path('replace/with/proper/absolute/path/obj_000002.obj')


def compare_models(m_ply: Dict, m_tri: Dict):
    """
    Model contain potentially:

     - 'pts' (nx3 ndarray)
     - 'normals' (nx3 ndarray), optional
     - 'colors' (nx3 ndarray), optional
     - 'faces' (mx3 ndarray), optional
     - 'texture_uv' (nx2 ndarray), optional
     - 'texture_uv_face' (mx6 ndarray), optional
     - 'texture_file' (string), optional
    """

    assert np.all(np.isclose(m_ply['pts'], m_tri['pts']))
    if 'normals' in m_ply:
        assert np.all(np.isclose(m_ply['normals'], m_tri['normals'], atol=1e-6))
    if 'colors' in m_ply:
        assert np.all(np.isclose(m_ply['colors'], m_tri['colors']))
    if 'faces' in m_ply:
        assert np.all(np.isclose(m_ply['faces'], m_tri['faces']))
    if 'texture_uv' in m_ply:
        assert np.all(np.isclose(m_ply['texture_uv'], m_tri['texture_uv']))
    if 'texture_uv_face' in m_ply:
        assert np.all(np.isclose(m_ply['texture_uv_face'], m_tri['texture_uv_face']))
    if 'texture_file' in m_ply:
        assert m_ply['texture_file'] == m_tri['texture_file']

class TestInout(unittest.TestCase):

    def test_load_ply_non_regression(self):
        model_ply = inout.load_ply(EXAMPLE_PLY_PATH)
        model_tri = inout.load_mesh(EXAMPLE_PLY_PATH)
        compare_models(model_ply, model_tri)

    def test_load_ply_vs_obj(self):
        model_ply = inout.load_ply(EXAMPLE_PLY_PATH)
        model_tri = inout.load_mesh(EXAMPLE_OBJ_PATH)
        compare_models(model_ply, model_tri)


if __name__ == '__main__':
    unittest.main()