from terial import models
from terial.database import session_scope
from terial.materials import loader


def main():
    with session_scope() as sess:
        materials = (sess.query(models.Material)
                     .filter_by(type=models.MaterialType.MDL)
                     .all())

        for material in materials:
            bmat = loader.material_to_brender(material)
            print(bmat)
            del bmat


if __name__ == '__main__':
    main()
