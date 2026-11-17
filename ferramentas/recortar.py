"""Villani - recorte inteligente de fotos para o Instagram.

Acha o ponto de interesse da foto (smartcrop: detalhe, cor de pele, saturação)
e protege rostos (OpenCV), depois recorta nos formatos do Instagram.
Também descarta foto tremida ou escura demais (só avisa, não apaga).

Instalar:  pip install smartcrop "opencv-python-headless<5" pillow
Uso:       python3 recortar.py <pasta_entrada> <pasta_saida> [feed|story|ambos]
Saída:     <nome>_feed.jpg (1080x1350, 4:5)  e/ou  <nome>_story.jpg (1080x1920, 9:16)
           + triagem.csv com nitidez, brilho e aviso por foto.
O Claude sempre confere cada recorte antes de usar numa peça.
"""
import csv, os, sys
import cv2, numpy as np, smartcrop
from PIL import Image, ImageOps

FORMATOS = {"feed": (1080, 1350), "story": (1080, 1920)}
EXT = (".jpg", ".jpeg", ".png", ".heic", ".webp")
NITIDEZ_MIN, BRILHO_MIN, BRILHO_MAX = 60, 45, 225
_XML = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""), "haarcascade_frontalface_default.xml")
_rostos = cv2.CascadeClassifier(_XML) if os.path.exists(_XML) else None  # OpenCV 5 tirou os arquivos: usar <5


def abrir(p):
    if p.lower().endswith(".heic"):
        from pillow_heif import register_heif_opener  # pip install pillow-heif
        register_heif_opener()
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def triagem(im):
    g = cv2.cvtColor(np.array(im.resize((800, int(800 * im.height / im.width)))), cv2.COLOR_RGB2GRAY)
    nit, bri = cv2.Laplacian(g, cv2.CV_64F).var(), g.mean()
    avisos = []
    if nit < NITIDEZ_MIN: avisos.append("tremida/desfocada")
    if bri < BRILHO_MIN: avisos.append("escura")
    if bri > BRILHO_MAX: avisos.append("estourada")
    return round(nit), round(bri), "; ".join(avisos)


def rostos(im):
    if _rostos is None or _rostos.empty(): return []
    esc = 800 / im.width
    g = cv2.cvtColor(np.array(im.resize((800, int(im.height * esc)))), cv2.COLOR_RGB2GRAY)
    return [tuple(int(v / esc) for v in r) for r in _rostos.detectMultiScale(g, 1.1, 6, minSize=(40, 40))]


def recortar(im, alvo):
    tw, th = alvo
    sc = smartcrop.SmartCrop()
    r = sc.crop(im, tw, th)["top_crop"]
    x, y, w, h = r["x"], r["y"], r["width"], r["height"]
    # garante que nenhum rosto fique cortado: desloca a janela para cobrir todos, se couber
    fs = rostos(im)
    if fs:
        x0 = min(f[0] for f in fs); x1 = max(f[0] + f[2] for f in fs)
        y0 = min(f[1] for f in fs); y1 = max(f[1] + f[3] for f in fs)
        if x1 - x0 <= w: x = min(max(x, x1 - w), x0)
        if y1 - y0 <= h: y = min(max(y, y1 - h), y0)
        x = max(0, min(x, im.width - w)); y = max(0, min(y, im.height - h))
    return im.crop((x, y, x + w, y + h)).resize(alvo, Image.LANCZOS), len(fs)


def main():
    ent, sai = sys.argv[1], sys.argv[2]
    modo = sys.argv[3] if len(sys.argv) > 3 else "ambos"
    fmts = list(FORMATOS) if modo == "ambos" else [modo]
    os.makedirs(sai, exist_ok=True)
    linhas = []
    for nome in sorted(os.listdir(ent)):
        if not nome.lower().endswith(EXT): continue
        im = abrir(os.path.join(ent, nome))
        nit, bri, aviso = triagem(im)
        base = os.path.splitext(nome)[0]
        n_rostos = 0
        for f in fmts:
            out, n_rostos = recortar(im, FORMATOS[f])
            out.save(os.path.join(sai, f"{base}_{f}.jpg"), quality=92)
        linhas.append([nome, im.width, im.height, nit, bri, n_rostos, aviso])
        print(f"{nome}: nitidez {nit}, brilho {bri}, rostos {n_rostos} {('⚠ ' + aviso) if aviso else ''}")
    with open(os.path.join(sai, "triagem.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arquivo", "largura", "altura", "nitidez", "brilho", "rostos", "aviso"]); w.writerows(linhas)


if __name__ == "__main__":
    main()
