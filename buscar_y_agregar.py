"""
Busca todas las carpetas (en todos los FIDs) que correspondan a una fecha
especifica de imagen, y las agrega a imagenes_duplicadas.txt (sin duplicar
lineas que ya esten).

Como usarlo:
    python buscar_y_agregar.py 20230907

Esto busca cualquier ruta dentro de embeddings/joint/ que contenga
"20230907" en el nombre de la carpeta de la imagen, y agrega su ruta
relativa (formato s2_l2a/fid_X/window/s2id) al txt.
"""

import sys
import glob
from pathlib import Path

EMB_ROOT = "embeddings"          # ajusta si tu carpeta de embeddings tiene otro nombre/ruta
TXT_PATH = "imagenes_duplicadas.txt"

def main(fecha):
    # busca en todas las carpetas de embeddings que tengan esa fecha en el nombre
    patron = str(Path(EMB_ROOT) / "joint" / "fid_*" / "*" / f"*{fecha}*" / "tile_*.npy")
    encontrados = glob.glob(patron)

    if not encontrados:
        print(f"No se encontro ninguna carpeta con la fecha {fecha} en {EMB_ROOT}/joint/")
        return

    # arma el set de rutas unicas en formato s2_l2a/fid_X/window/s2id
    nuevas = set()
    for npy in encontrados:
        p = Path(npy)
        pair = p.parent.name                    # "<s2id>__<s1id>"
        window = p.parent.parent.name
        fid = p.parent.parent.parent.name.replace("fid_", "")
        s2_id = pair.split("__")[0]
        nuevas.add(f"s2_l2a/fid_{fid}/{window}/{s2_id}")

    # lee lo que ya esta en el txt, para no duplicar
    txt_path = Path(TXT_PATH)
    existentes = set()
    if txt_path.exists():
        existentes = {l.strip() for l in txt_path.read_text().splitlines() if l.strip()}

    a_agregar = sorted(nuevas - existentes)

    if not a_agregar:
        print(f"Las {len(nuevas)} rutas encontradas con fecha {fecha} ya estaban en {TXT_PATH}")
        return

    with open(txt_path, "a") as f:
        for r in a_agregar:
            f.write(r + "\n")

    print(f"Encontradas {len(nuevas)} rutas con fecha {fecha}.")
    print(f"Agregadas {len(a_agregar)} nuevas a {TXT_PATH}:")
    for r in a_agregar:
        print(f"  {r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python buscar_y_agregar.py <fecha, ej 20230907>")
        sys.exit(1)
    main(sys.argv[1])
