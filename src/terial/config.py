from pathlib import Path


"""
Database Configuration
"""
DB_NAME = 'photoshape_db'
DB_HOST = 'drell.cs.washington.edu' 
DB_USER = 'photoshape_user'
DB_PASS = 'GcW7pbMaxxb2n6BwjvpPrTTw8vdZkWMUNPd7io'
DB_PORT = 15432
DB_DSN = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'


"""
Web configuration
"""
WEB_ROOT = 'http://localhost/static'


"""
Directory configuration
"""
LOCAL_ROOT = Path('/data/photoshape')
REMOTE_ROOT = Path('/projects/grail/kparnb/data')

# The directory where blob data associated with DB rows will be stored.
BLOB_ROOT = Path(LOCAL_ROOT, 'blobs')

# The directory where ShapeNet stuff lives.
SHAPENET_DIR = Path('/projects/grail/kparnb/data/shapenet/')

SHAPENET_META_DIR = SHAPENET_DIR / 'metadata'
SHAPENET_CORE_DIR = SHAPENET_DIR / 'ShapeNetCore.v2'
SHAPENET_TAXONOMY_PATH = SHAPENET_DIR / 'taxonomy.json'


"""
Alignment parameters
"""
ALIGN_BIN_SIZE = 8
ALIGN_IM_SHAPE = (100, 100)
# ALIGN_DIST_THRES = 8.0
ALIGN_DIST_THRES = 10.0
ALIGN_DIST_THRES_GEN = 100.0
ALIGN_TOP_K = 7


"""
Artifact names
"""
SHAPE_ALIGN_DATA_NAME = f'align_hog_{ALIGN_BIN_SIZE}.npz'
EXEMPLAR_ALIGN_DATA_NAME = f'align_hog_{ALIGN_BIN_SIZE}.npz'
EXEMPLAR_SUBST_MAP_NAME_OLD = 'substance_map_minc_vgg16.npz'
EXEMPLAR_SUBST_VIS_NAME_OLD = 'substance_map_minc_vgg16.png'
EXEMPLAR_SUBST_MAP_NAME = 'substance_map_minc_vgg16.map.v2.png'
EXEMPLAR_SUBST_VIS_NAME = 'substance_map_minc_vgg16.vis.v2.png'

RADMAP_DIR = Path('/projects/grail/kpar/data/envmaps2')
SHAPE_REND_RADMAP_NAME = 'studio027.cross.exr'
SHAPE_REND_RADMAP_PATH = RADMAP_DIR / SHAPE_REND_RADMAP_NAME
SHAPE_REND_SHAPE = (500, 500)
SHAPE_REND_SHAPE_STR = f'{SHAPE_REND_SHAPE[0]}x{SHAPE_REND_SHAPE[1]}'

SHAPE_REND_PHONG_NAME = f'shape_rend_phong_{SHAPE_REND_SHAPE_STR}.png'
SHAPE_REND_SUBSTANCE_NAME = f'shape_rend_substance_preview_{SHAPE_REND_SHAPE_STR}.png'
SHAPE_REND_PREVIEW_NAME = f'shape_rend_preview_{SHAPE_REND_SHAPE_STR}.png'
SHAPE_REND_SEGMENT_MAP_NAME = f'shape_rend_segments_{SHAPE_REND_SHAPE_STR}.map.png'
SHAPE_REND_SEGMENT_VIS_NAME = f'shape_rend_segments_{SHAPE_REND_SHAPE_STR}.vis.png'
SHAPE_REND_NORMALS_NAME = f'shape_rend_normals_{SHAPE_REND_SHAPE_STR}.v5.png'
SHAPE_REND_UV_U_NAME = f'shape_rend_uv_u_{SHAPE_REND_SHAPE_STR}.npz'
SHAPE_REND_UV_V_NAME = f'shape_rend_uv_v_{SHAPE_REND_SHAPE_STR}.npz'
SHAPE_TANGENTS_NAME = f'shape_rend_tangents_{SHAPE_REND_SHAPE_STR}.npz'
SHAPE_BITANGENTS_NAME = f'shape_rend_bitangents_{SHAPE_REND_SHAPE_STR}.npz'
SHAPE_WORLD_COORDS_NAME = f'shape_rend_world_coords_{SHAPE_REND_SHAPE_STR}.npz'
SHAPE_DEPTH_NAME = f'shape_rend_depth_{SHAPE_REND_SHAPE_STR}.npz'

# SHAPE_WARPED_PHONG_NAME = f'shape_warped_phong_{SHAPE_REND_SHAPE_STR}'
# SHAPE_WARPED_SEGMENTS_NAME = f'shape_warped_segments_{SHAPE_REND_SHAPE_STR}'
# SHAPE_CLEAN_SEGMENTS_NAME = f'shape_clean_segments_{SHAPE_REND_SHAPE_STR}'
PAIR_FG_BBOX_NAME = f'shape_fg_bbox_{SHAPE_REND_SHAPE_STR}.png'
PAIR_RAW_SEGMENT_MAP_NAME = f'shape_segment_map_raw_{SHAPE_REND_SHAPE_STR}.png'
PAIR_SHAPE_WARPED_PHONG_NAME = f'shape_warped_phong_{SHAPE_REND_SHAPE_STR}_3.png'
PAIR_SHAPE_WARPED_SEGMENT_VIS_NAME = f'shape_warped_segments_{SHAPE_REND_SHAPE_STR}.vis.v2.png'
PAIR_SHAPE_WARPED_SEGMENT_MAP_NAME = f'shape_warped_segments_{SHAPE_REND_SHAPE_STR}.map.v2.png'
PAIR_SHAPE_CLEAN_SEGMENT_VIS_NAME = f'shape_clean_segments_{SHAPE_REND_SHAPE_STR}.vis.v2.png'
PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME = f'shape_clean_segments_{SHAPE_REND_SHAPE_STR}.map.v2.png'
PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME_OLD = f'shape_clean_segments_{SHAPE_REND_SHAPE_STR}.map.png'
PAIR_SEGMENT_OVERLAY_NAME = f'shape_segments_overlay.v2.png'
PAIR_PROXY_SUBST_MAP_NAME = f'pair_proxy_substances.map.png'
PAIR_PROXY_SUBST_VIS_NAME = f'pair_proxy_substances.vis.png'
PAIR_SHAPE_SUBST_MAP_NAME = f'pair_shape_substances.map.png'
PAIR_SHAPE_SUBST_VIS_NAME = f'pair_shape_substances.vis.png'

PAIR_SHAPE_WARPED_NORMALS_NAME = f'shape_warped_normals_{SHAPE_REND_SHAPE_STR}.npz'
PAIR_SHAPE_WARPED_UV_U_NAME = f'shape_warped_uv_u_{SHAPE_REND_SHAPE_STR}.npz'
PAIR_SHAPE_WARPED_UV_V_NAME = f'shape_warped_uv_v_{SHAPE_REND_SHAPE_STR}.npz'

FLOW_DATA_NAME = f'exemplar_rend_flow_silhouette_{SHAPE_REND_SHAPE_STR}.npz'
FLOW_VIS_DATA_NAME = f'exemplar_rend_flow_silhouette_{SHAPE_REND_SHAPE_STR}.png'
FLOW_EXEMPLAR_SILHOUETTE_VIS = f'flow_exemplar_silhouette_{SHAPE_REND_SHAPE_STR}.vis.png'
FLOW_SHAPE_SILHOUETTE_VIS = f'flow_shape_silhouette_{SHAPE_REND_SHAPE_STR}.vis.png'
# FLOW_DATA_NAME = f'exemplar_rend_flow_phong_{SHAPE_REND_SHAPE_STR}'

MATERIAL_DIR = Path(LOCAL_ROOT, 'materials-500x500')
MATERIAL_DIR_POLIIGON = MATERIAL_DIR / 'poliigon'
MATERIAL_DIR_AITTALA = MATERIAL_DIR / 'aittala-beckmann'
MATERIAL_DIR_VRAY = MATERIAL_DIR / 'vray-materials-de'
MATERIAL_DIR_ADOBE_STOCK = MATERIAL_DIR / 'adobe-stock'

MATERIAL_CAND_VGG16_WHOLE_NAME = 'material_matches/seg_mat_candidates_minc_vgg16_2.npz'
MATERIAL_CAND_VGG16_PATCH_NAME = 'material_matches/seg_mat_candidates_minc_vgg16_patch_2.npz'
MATERIAL_CAND_MEDCOL_NAME = 'material_matches/seg_mat_candidates_medcol_256x256.npz'
MATERIAL_CAND_EXEMPLAR_NAME = 'material_matches/seg_mat_candidates_exemplar.npz'

MATERIAL_PREVIEW_MONKEY_NAME = 'previews/monkey.studio021.png'

MINC_VGG16_WEIGHTS_PATH = Path('/projects/grail/kparnb/prim/weights/minc_vgg16.npy')

# BRDF_CLASSIFIER_DIR = Path(DATA_ROOT, 'terial/brdf-classifier')
BRDF_CLASSIFIER_DIR_REMOTE = Path(REMOTE_ROOT, 'terial/brdf-classifier')
BRDF_CLASSIFIER_DIR_LOCAL = Path(LOCAL_ROOT, 'terial/brdf-classifier')
BRDF_CLASSIFIER_DATASETS_DIR = Path(BRDF_CLASSIFIER_DIR_LOCAL, 'datasets')
BRDF_CLASSIFIER_SNAPSHOTS_DIR = Path(BRDF_CLASSIFIER_DIR_REMOTE, 'snapshots')
BRDF_CLASSIFIER_DATA_PATH = Path(LOCAL_ROOT, 'terial/brdf-classifier/raw-20180306')



"""
Inference parameters
"""
INFERENCE_TOPK = 12
INFERENCE_MAX_DIST = 20


BRUTEFORCE_PAIR_MATERIAL_REND_DIR = Path(
    LOCAL_ROOT, 'terial/brdf-classifier/bruteforce/pair-material-rends')

SUBSTANCES = [
    'fabric',
    'leather',
    'wood',
    'metal',
    'plastic',
    'background',
]
