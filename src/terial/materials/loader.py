from pathlib import Path

import brender.material
from terial import models, config


def material_to_brender(material: models.Material, **kwargs):
    if material.type == models.MaterialType.AITTALA_BECKMANN:
        base_dir = Path(config.MATERIAL_DIR_AITTALA, material.substance,
                        material.name)
        return brender.material.AittalaMaterial.from_path(base_dir, **kwargs)
    elif material.type == models.MaterialType.POLIIGON:
        base_dir = Path(config.MATERIAL_DIR_POLIIGON, material.substance,
                        material.name)
        return brender.material.PoliigonMaterial.from_path(base_dir, **kwargs)
    elif material.type == models.MaterialType.VRAY:
        base_dir = Path(config.MATERIAL_DIR_VRAY, material.substance,
                        material.params['raw_name'], )
        return brender.material.VRayMaterial.from_path(base_dir, **kwargs)
    elif (material.type == models.MaterialType.MDL
          and material.source == 'adobe_stock'):
        mdl_path = Path(config.MATERIAL_DIR_ADOBE_STOCK,
                        material.substance,
                        f'AdobeStock_{material.source_id}',
                        f'{material.name}.mdl')
        return brender.material.MDLMaterial.from_path(mdl_path, **kwargs)
    elif material.type == models.MaterialType.PRINCIPLED:
        return brender.material.PrincipledMaterial(
            diffuse_color=material.params['diffuse_color'],
            specular=material.params['specular'],
            metallic=material.params['metallic'],
            roughness=material.params['roughness'],
            anisotropy=material.params['anisotropy'],
            anisotropic_rotation=material.params['anisotropic_rotation'],
            clearcoat=material.params['clearcoat'],
            clearcoat_roughness=material.params['clearcoat_roughness'],
            ior=material.params['ior'],
            **kwargs
        )
    elif material.type == models.MaterialType.BLINN_PHONG:
        return brender.material.BlinnPhongMaterial(
            diffuse_albedo=[float(c) for c in material.params['diffuse']],
            specular_albedo=[float(c) for c in material.params['specular']],
            shininess=float(material.params['shininess']),
            **kwargs)
    else:
        raise ValueError(f'Unsupported material type {material.type}')
