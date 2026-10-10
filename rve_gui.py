#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Удобный GUI для сборки RVE500 из микро-КТ лёгкого.

Пайплайн (см. REPORT_human_lung_RVE.md):
  JP2-zip (999 срезов, 4.9 мкм)
    -> [1] монтаж превью -> выбор ROI
    -> [2] вырезка куба 512^3 (extract_cube.py)
    -> [3] сегментация: гаусс + Отсу + closing -> маска (mesh_cube_v2.py)
    -> [4] RVE ~500 мкм: поиск позиции + STL обеих фаз (make_rve500.py)

Запуск:
  Windows:  двойной клик по run_gui.bat  (использует .venv если есть)
  Linux/WSL: bash run_gui.sh  или  python3 rve_gui.py

GUI зависит ТОЛЬКО от tkinter (stdlib) и запускается даже без numpy/Pillow.
Тяжёлые зависимости (numpy, Pillow, scipy, scikit-image, trimesh, matplotlib)
нужны только для самих расчётов; кнопка "Проверить зависимости" покажет чего нет.
"""
import io
import os
import queue
import subprocess
import sys
import threading
import traceback
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
PREVIEW_DIR = DATA / "preview"
RVE_DIR = DATA / "rve500"
DEFAULT_ZIP_NAME = "4.9um_LADAF-2020-27_lung-left_06_pag-0.02_0.04_jp2_.zip"

# Дефолты из отчёта (REPORT, п.4): проверенный чистый участок паренхимы.
D_X0, D_Y0, D_Z0, D_N = 295, 1044, 400, 512
D_VOXEL = 4.9
D_SIGMA = 0.8
D_RVE_UM = 500.0
D_SEARCH_STEP = 51
D_TAUBIN = 25

for _d in (DATA, PREVIEW_DIR, RVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- pipeline ---
# Все функции принимают log=callable(str) для вывода в GUI-лог.
# Импорты numpy/PIL/scipy/skimage/trimesh — ленивые, внутри функций.

def find_zip(explicit=""):
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    p = PROJECT / DEFAULT_ZIP_NAME
    if p.is_file():
        return p
    zips = sorted(PROJECT.glob("*.zip"))
    return zips[0] if zips else p


def step_preview_montage(zip_path, step=66, log=print):
    import numpy as np
    from PIL import Image
    zpath = Path(zip_path)
    if not zpath.is_file():
        raise FileNotFoundError(f"ZIP не найден: {zpath}")
    log(f"Открываю {zpath.name} ...")
    z = zipfile.ZipFile(zpath)
    names = sorted(n for n in z.namelist() if n.endswith(".jp2"))
    log(f"Срезов JP2 в архиве: {len(names)}")
    idxs = list(range(0, len(names), step))
    tiles, vals = [], []
    for i in idxs:
        a = np.array(Image.open(io.BytesIO(z.read(names[i]))))
        vals.append(a[::4, ::4])
        lo, hi = np.percentile(a, [1, 99])
        img = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        tiles.append(Image.fromarray((img * 255).astype(np.uint8)).resize((476, 476)))
    v = np.concatenate([x.ravel() for x in vals])
    log("Процентили интенсивностей: " + ", ".join(
        f"p{p}={np.percentile(v, p):.0f}" for p in (0.1, 1, 10, 50, 90, 99, 99.9)))
    cols, rows = 4, (len(tiles) + 3) // 4
    montage = Image.new("L", (cols * 476, rows * 476))
    for k, t in enumerate(tiles):
        montage.paste(t, ((k % cols) * 476, (k // cols) * 476))
    out = PREVIEW_DIR / "jp2_4.9um_slices_montage.png"
    montage.save(out)
    log(f"OK: {out} ({len(tiles)} срезов)")
    return out


def step_extract_cube(zip_path, x0, y0, z0, n, log=print):
    import numpy as np
    from PIL import Image
    zpath = Path(zip_path)
    out = DATA / f"cube_4.9um_{n}.npy"
    if out.name != "cube_4.9um_512.npy" and n == 512:
        out = DATA / "cube_4.9um_512.npy"
    log(f"Читаю срезы [{z0}:{z0+n}] ROI x=[{x0}:{x0+n}] y=[{y0}:{y0+n}] ...")
    z = zipfile.ZipFile(zpath)
    names = sorted(nn for nn in z.namelist() if nn.endswith(".jp2"))
    if z0 + n > len(names):
        raise ValueError(f"Z0+N={z0+n} > срезов в архиве ({len(names)})")
    probe = np.array(Image.open(io.BytesIO(z.read(names[0]))))
    H, W = probe.shape
    if x0 + n > W or y0 + n > H:
        raise ValueError(f"ROI выходит за кадр {W}x{H}: x0+n={x0+n}, y0+n={y0+n}")
    arr = np.lib.format.open_memmap(str(out), mode="w+", dtype=np.uint16, shape=(n, n, n))
    for k, zi in enumerate(range(z0, z0 + n)):
        a = np.array(Image.open(io.BytesIO(z.read(names[zi]))))
        arr[k] = a[y0:y0 + n, x0:x0 + n]
        if k % 64 == 0:
            log(f"  срез {k}/{n}")
    arr.flush()
    log(f"OK: {out} ({out.stat().st_size/1e6:.0f} МБ)")
    return out


def step_segment(cube_path, sigma=0.8, thr_manual=None, do_closing=True, log=print):
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from PIL import Image
    cube_path = Path(cube_path)
    log(f"Загружаю куб {cube_path.name} ...")
    data = np.load(str(cube_path), mmap_mode="r")
    data = np.asarray(data)
    log(f"  shape={data.shape} dtype={data.dtype} range={data.min()}..{data.max()}")
    if thr_manual is not None:
        thr = float(thr_manual)
        log(f"Порог вручную: {thr:.0f}")
    else:
        thr = float(threshold_otsu(data[::4, ::4, ::4].ravel()))
        log(f"Порог Отсу (по выборке [::4]): {thr:.0f}")
    log(f"Гаусс σ={sigma} по срезам ...")
    smooth = np.empty(data.shape, dtype=np.float32)
    for k in range(data.shape[0]):
        smooth[k] = ndi.gaussian_filter(data[k].astype(np.float32), sigma=sigma)
        if k % 128 == 0:
            log(f"  сглажено {k}/{data.shape[0]}")
    f0 = float((smooth > thr).mean())
    log(f"Доля ткани до closing: {f0:.3f}")
    mask = smooth > thr
    del smooth
    if do_closing:
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=1)
    mask[0, :, :] = mask[-1, :, :] = False
    mask[:, 0, :] = mask[:, -1, :] = False
    mask[:, :, 0] = mask[:, :, -1] = False
    log(f"Доля ткани после closing+грани: {mask.mean():.3f}")
    out = DATA / f"{cube_path.stem}_mask.npz"
    # каноническое имя для 512, чтобы make_rve500 находил без настроек
    if cube_path.name == "cube_4.9um_512.npy":
        out = DATA / "cube_4.9um_512_mask.npz"
    np.savez_compressed(str(out), mask=mask)
    log(f"OK: маска {out} ({out.stat().st_size/1e6:.1f} МБ)")
    # оверлеи для контроля
    for k, tag in [(mask.shape[0]//4, "z25"), (mask.shape[0]//2, "z50"),
                   (3*mask.shape[0]//4, "z75")]:
        lo, hi = np.percentile(np.asarray(data[k]), [1, 99])
        g = np.clip((data[k].astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.dstack([g, g, g])
        edge = mask[k] ^ ndi.binary_erosion(mask[k])
        rgb[edge] = [1.0, 0.2, 0.2]
        Image.fromarray((rgb * 255).astype(np.uint8)).save(PREVIEW_DIR / f"mask_overlay_{tag}.png")
    log(f"Оверлеи: {PREVIEW_DIR}/mask_overlay_z*.png")
    return out, float(thr)


def _mesh_from_mask(mask, voxel_um, out_path, taubin_iters, log):
    import numpy as np
    from skimage.measure import marching_cubes
    from trimesh import Trimesh
    from trimesh.exchange.export import export_mesh
    from trimesh.smoothing import filter_taubin
    padded = np.pad(mask, 1, constant_values=False)
    verts, faces, _, _ = marching_cubes(padded.astype(np.float32), level=0.5,
                                        spacing=(voxel_um,) * 3)
    m = Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    filter_taubin(m, lamb=0.5, nu=0.53, iterations=taubin_iters)
    m.fix_normals()
    export_mesh(m, str(out_path))
    log(f"{out_path.name}: {len(m.faces)} тр., {out_path.stat().st_size/1e6:.1f} МБ")
    return m


def _save_overlay(gray, mask, path):
    import numpy as np
    from scipy import ndimage as ndi
    from PIL import Image
    lo, hi = np.percentile(gray, [1, 99])
    g = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([g, g, g])
    edge = mask ^ ndi.binary_erosion(mask)
    rgb[edge] = [1.0, 0.25, 0.25]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def _render_mesh_preview(mesh, out_png, log=print):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        if len(faces) > 120_000:
            sel = np.random.default_rng(0).choice(len(faces), 120_000, replace=False)
            faces = faces[sel]
        tri = verts[faces]
        fig = plt.figure(figsize=(8, 8), dpi=120)
        ax = fig.add_subplot(projection="3d")
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = np.clip(np.abs(n @ np.array([0.4, -0.5, 0.77])), 0.25, 1.0)
        cols = np.dstack([shade*0.55, shade*0.75, shade*0.45, np.ones_like(shade)])[0]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolor="none"))
        r = verts.max(0) - verts.min(0)
        c = (verts.max(0) + verts.min(0)) / 2
        ax.set_xlim(c[0]-r.max()/2, c[0]+r.max()/2)
        ax.set_ylim(c[1]-r.max()/2, c[1]+r.max()/2)
        ax.set_zlim(c[2]-r.max()/2, c[2]+r.max()/2)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=-60)
        fig.tight_layout(pad=0)
        fig.savefig(out_png)
        plt.close(fig)
        log(f"Рендер: {out_png.name}")
    except Exception as e:
        log(f"Рендер пропущен (нет matplotlib?): {e}")


def step_build_rve(mask_path, gray_path, voxel_um=4.9, size_um=500.0,
                   search_step=51, taubin_iters=25, log=print):
    import numpy as np
    RVE_DIR.mkdir(exist_ok=True)
    n = int(round(size_um / voxel_um))
    log(f"RVE: {size_um:.0f} мкм / {voxel_um} мкм = {n} вокселей (ребро {n*voxel_um:.1f} мкм)")
    mask_full = np.load(str(mask_path))["mask"]
    if mask_full.ndim != 3 or not (mask_full.shape[0] == mask_full.shape[1] == mask_full.shape[2]):
        raise ValueError(f"Маска должна быть кубом, а shape={mask_full.shape}")
    S = mask_full.shape[0]
    if n > S:
        raise ValueError(f"RVE ({n}) больше куба ({S})")
    glob = float(mask_full.mean())
    log(f"Средняя доля ткани по кубу: {glob:.3f}; поиск позиции шагом {search_step} ...")
    best, best_score = None, 1e9
    for ox in range(0, S - n + 1, search_step):
        for oy in range(0, S - n + 1, search_step):
            for oz in range(0, S - n + 1, search_step):
                frac = float(mask_full[oz:oz+n, oy:oy+n, ox:ox+n].mean())
                s = abs(frac - glob)
                if s < best_score:
                    best, best_score = (ox, oy, oz), s
    ox, oy, oz = best
    loc = float(mask_full[oz:oz+n, oy:oy+n, ox:ox+n].mean())
    log(f"Позиция: x={ox} y={oy} z={oz}, доля ткани локально {loc:.3f} (Δ={best_score:.4f})")
    gray = np.load(str(gray_path), mmap_mode="r")
    sub_gray = np.asarray(gray[oz:oz+n, oy:oy+n, ox:ox+n])
    sub_mask = mask_full[oz:oz+n, oy:oy+n, ox:ox+n].copy()
    np.save(str(RVE_DIR / "cube_gray.npy"), sub_gray)
    np.savez_compressed(str(RVE_DIR / "mask_tissue.npz"), mask=sub_mask)
    vt = sub_mask.sum() * voxel_um**3
    va = (1 - sub_mask.mean()) * n**3 * voxel_um**3
    log(f"Объёмы: ткань {vt/1e9:.4f} мм³, воздух {va/1e9:.4f} мм³")
    log("Меш ткани (marching cubes + Taubin) ...")
    m_t = _mesh_from_mask(sub_mask, voxel_um, RVE_DIR / "lung_rve500_tissue.stl",
                          taubin_iters, log)
    log("Меш воздуха ...")
    _mesh_from_mask(~sub_mask, voxel_um, RVE_DIR / "lung_rve500_air.stl",
                    taubin_iters, log)
    k = n // 2
    _save_overlay(sub_gray[k], sub_mask[k], RVE_DIR / "overlay_mid.png")
    try:
        from PIL import Image
        im = Image.open(RVE_DIR / "overlay_mid.png")
        im.resize((im.width*4, im.height*4), Image.NEAREST).save(RVE_DIR / "overlay_x4.png")
    except Exception:
        pass
    log("Оверлей: overlay_mid.png (+overlay_x4.png)")
    _render_mesh_preview(m_t, RVE_DIR / "rve500_render.png", log)
    log("ГОТОВО: data/rve500/ — tissue/air STL, mask, gray, картинки.")
    return {"offset": (ox, oy, oz), "frac": loc, "n": n}


def check_deps():
    rows = []
    for mod, want in [("numpy", "массивы"), ("PIL", "картинки/JP2"),
                      ("scipy", "фильтры/морфология"),
                      ("skimage", "Отсу/marching cubes"),
                      ("trimesh", "STL"), ("matplotlib", "3D-превью (необяз.)")]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            rows.append((mod, v, "OK"))
        except Exception as e:
            rows.append((mod, "—", f"НЕТ ({e})"))
    return rows


# ------------------------------------------------------------------- GUI ---
def _tk():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    return tk, ttk, filedialog, messagebox


def main_gui():
    tk, ttk, filedialog, messagebox = _tk()
    root = tk.Tk()
    root.title("MicroCT Lungs — RVE500")
    root.geometry("1060x720")
    try:
        style = ttk.Style()
        for th in ("clam", "vista", "xpnative", "default"):
            if th in style.theme_names():
                style.theme_use(th)
                break
    except Exception:
        pass

    # --- переменные ---
    v_zip = tk.StringVar(value=str(find_zip()))
    v_x0 = tk.StringVar(value=str(D_X0)); v_y0 = tk.StringVar(value=str(D_Y0))
    v_z0 = tk.StringVar(value=str(D_Z0)); v_n = tk.StringVar(value=str(D_N))
    v_vox = tk.StringVar(value=str(D_VOXEL)); v_sig = tk.StringVar(value=str(D_SIGMA))
    v_thr_auto = tk.BooleanVar(value=True); v_thr = tk.StringVar(value="25519")
    v_rve = tk.StringVar(value=str(D_RVE_UM)); v_step = tk.StringVar(value=str(D_SEARCH_STEP))
    v_taubin = tk.StringVar(value=str(D_TAUBIN))
    v_mont_step = tk.StringVar(value="66")

    log_q = queue.Queue()
    busy = {"on": False}

    # --- layout: слева параметры+кнопки, справа лог+статус+превью ---
    main = ttk.Panedwindow(root, orient="horizontal")
    main.pack(fill="both", expand=True, padx=8, pady=8)
    left = ttk.Frame(main); right = ttk.Frame(main)
    main.add(left, weight=1); main.add(right, weight=2)

    # ===== левая панель =====
    r = 0
    ttk.Label(left, text="1. Данные", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 2)); r += 1
    ttk.Label(left, text="JP2-zip:").grid(row=r, column=0, sticky="w"); r += 1
    e_zip = ttk.Entry(left, textvariable=v_zip, width=42)
    e_zip.grid(row=r, column=0, columnspan=2, sticky="ew")
    def browse_zip():
        p = filedialog.askopenfilename(title="Выберите JP2 zip",
            filetypes=[("ZIP", "*.zip"), ("Все", "*.*")],
            initialdir=str(PROJECT))
        if p: v_zip.set(p)
    ttk.Button(left, text="…", width=3, command=browse_zip).grid(row=r, column=2); r += 1

    ttk.Label(left, text="2. Куб 512³ (extract_cube)", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 2)); r += 1
    f_cube = ttk.Frame(left); f_cube.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
    for i, (lab, var, w) in enumerate([("X0", v_x0, 7), ("Y0", v_y0, 7), ("Z0", v_z0, 7),
                                       ("N", v_n, 6), ("мкм/вокс", v_vox, 6)]):
        ttk.Label(f_cube, text=lab).grid(row=0, column=2*i, padx=(0 if i == 0 else 6, 2))
        ttk.Entry(f_cube, textvariable=var, width=w).grid(row=0, column=2*i+1)

    ttk.Label(left, text="3. Сегментация (маска)", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 2)); r += 1
    f_seg = ttk.Frame(left); f_seg.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
    ttk.Label(f_seg, text="σ гаусса").grid(row=0, column=0, padx=(0, 2))
    ttk.Entry(f_seg, textvariable=v_sig, width=6).grid(row=0, column=1)
    ttk.Checkbutton(f_seg, text="Отсу авто", variable=v_thr_auto).grid(row=0, column=2, padx=(8, 2))
    ttk.Entry(f_seg, textvariable=v_thr, width=8).grid(row=0, column=3)

    ttk.Label(left, text="4. RVE (make_rve500)", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 2)); r += 1
    f_rve = ttk.Frame(left); f_rve.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
    for i, (lab, var, w) in enumerate([("Размер, мкм", v_rve, 7), ("Шаг поиска", v_step, 6), ("Taubin N", v_taubin, 5)]):
        ttk.Label(f_rve, text=lab).grid(row=0, column=2*i, padx=(0 if i == 0 else 6, 2))
        ttk.Entry(f_rve, textvariable=var, width=w).grid(row=0, column=2*i+1)

    ttk.Label(left, text="Монтаж: каждый K-й срез").grid(row=r, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(left, textvariable=v_mont_step, width=6).grid(row=r, column=1, sticky="w"); r += 1

    left.grid_columnconfigure(0, weight=1)
    btns = {}
    def mkbtn(txt, cmd, row):
        b = ttk.Button(left, text=txt, command=cmd)
        b.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        return b
    # кнопки создаются ниже после объявления run-функций; плейсхолдеры:
    brow = r

    # ===== правая панель =====
    ttk.Label(right, text="Статус файлов", font=("", 10, "bold")).pack(anchor="w")
    status_txt = tk.StringVar(value="…")
    ttk.Label(right, textvariable=status_txt, justify="left").pack(anchor="w", pady=(0, 4))

    prog = ttk.Progressbar(right, mode="indeterminate")
    prog.pack(fill="x", pady=2)

    ttk.Label(right, text="Лог", font=("", 10, "bold")).pack(anchor="w")
    log_box = tk.Text(right, height=20, wrap="word", font=("Consolas", 9))
    log_box.pack(fill="both", expand=True)
    scr = ttk.Scrollbar(log_box, command=log_box.yview)
    log_box.configure(yscrollcommand=scr.set)

    f_prev = ttk.Frame(right); f_prev.pack(fill="x", pady=4)
    img_label1 = ttk.Label(f_prev, text="(превью появится после расчёта)")
    img_label1.pack(side="left", padx=4)
    img_label2 = ttk.Label(f_prev, text="")
    img_label2.pack(side="left", padx=4)
    photo_refs = []

    def log(msg):
        log_q.put(str(msg))

    def flush_log():
        try:
            while True:
                line = log_q.get_nowait()
                log_box.insert("end", line + "\n")
                log_box.see("end")
        except queue.Empty:
            pass
        root.after(150, flush_log)

    def refresh_status():
        def sz(p):
            p = Path(p)
            return f"{p.stat().st_size/1e6:.1f} МБ" if p.is_file() else "—"
        try:
            n = int(v_n.get())
        except Exception:
            n = D_N
        cube = DATA / f"cube_4.9um_{n}.npy"
        if n == 512 and not cube.is_file():
            cube = DATA / "cube_4.9um_512.npy"
        mask = DATA / f"cube_4.9um_{n}_mask.npz"
        if n == 512 and not mask.is_file():
            mask = DATA / "cube_4.9um_512_mask.npz"
        lines = [
            f"{'✅' if Path(v_zip.get()).is_file() else '⬜'} ZIP: {sz(v_zip.get())}",
            f"{'✅' if cube.is_file() else '⬜'} куб: {cube.name} {sz(cube)}",
            f"{'✅' if mask.is_file() else '⬜'} маска: {mask.name} {sz(mask)}",
            f"{'✅' if (RVE_DIR/'lung_rve500_tissue.stl').is_file() else '⬜'} RVE ткань: {sz(RVE_DIR/'lung_rve500_tissue.stl')}"
            f" | воздух: {sz(RVE_DIR/'lung_rve500_air.stl')}",
        ]
        status_txt.set("\n".join(lines))

    def refresh_preview():
        photo_refs.clear()
        for path, lab in [(RVE_DIR / "overlay_mid.png", img_label1),
                          (RVE_DIR / "rve500_render.png", img_label2)]:
            if not path.is_file():
                continue
            try:
                try:
                    from PIL import Image, ImageTk
                    im = Image.open(path)
                    im.thumbnail((330, 260))
                    ph = ImageTk.PhotoImage(im)
                except Exception:
                    ph = tk.PhotoImage(file=str(path))
                    # даунсемпл до ~330px по большей стороне
                    w, h = ph.width(), ph.height()
                    f = max(1, max(w, h) // 330)
                    if f > 1:
                        ph = ph.subsample(f, f)
                lab.configure(image=ph, text="")
                photo_refs.append(ph)  # держать ссылку!
            except Exception as e:
                lab.configure(text=f"{path.name}: не показан ({e})")

    def set_busy(on):
        busy["on"] = on
        for b in btns.values():
            b.configure(state=("disabled" if on else "normal"))
        if on:
            prog.start(12)
        else:
            prog.stop()
            refresh_status(); refresh_preview()

    def run_in_thread(fn):
        if busy["on"]:
            return
        try:
            params = read_params()
        except ValueError as e:
            messagebox.showerror("Параметры", str(e))
            return
        set_busy(True)
        def worker():
            try:
                fn(params)
                log("— готово —")
            except Exception:
                log("ОШИБКА:\n" + traceback.format_exc())
                log_q.put("__ERROR__")
            finally:
                root.after(0, lambda: set_busy(False))
        threading.Thread(target=worker, daemon=True).start()

    def read_params():
        try:
            x0, y0, z0, n = int(v_x0.get()), int(v_y0.get()), int(v_z0.get()), int(v_n.get())
            vox, sig = float(v_vox.get()), float(v_sig.get())
            rve, sstep, tau = float(v_rve.get()), int(v_step.get()), int(v_taubin.get())
            mstep = int(v_mont_step.get())
        except ValueError:
            raise ValueError("Числовые поля заполнены некорректно.")
        thr = None if v_thr_auto.get() else float(v_thr.get())
        for name, val, lo, hi in [("N", n, 8, 1024), ("σ", sig, 0, 5),
                                  ("RVE мкм", rve, 50, 2500), ("шаг", sstep, 1, 512),
                                  ("Taubin", tau, 0, 200)]:
            if not (lo <= val <= hi):
                raise ValueError(f"{name}={val} вне [{lo}..{hi}].")
        return dict(zip_path=v_zip.get(), x0=x0, y0=y0, z0=z0, n=n, vox=vox,
                    sig=sig, thr=thr, rve=rve, sstep=sstep, tau=tau, mstep=mstep)

    # --- действия ---
    def act_montage(p):
        log(f"[1] Монтаж срезов из {Path(p['zip_path']).name} ...")
        step_preview_montage(p["zip_path"], step=p["mstep"], log=log)

    def act_extract(p):
        log(f"[2] Куб x0={p['x0']} y0={p['y0']} z0={p['z0']} N={p['n']} ...")
        step_extract_cube(p["zip_path"], p["x0"], p["y0"], p["z0"], p["n"], log=log)

    def act_segment(p):
        cube = DATA / f"cube_4.9um_{p['n']}.npy"
        if p["n"] == 512 and not cube.is_file():
            cube = DATA / "cube_4.9um_512.npy"
        if not cube.is_file():
            # fallback: любой куб
            cands = sorted(DATA.glob("cube_4.9um_*.npy"))
            if cands:
                cube = cands[0]
        log(f"[3] Сегментация {cube.name} σ={p['sig']} thr={'авто' if p['thr'] is None else p['thr']} ...")
        step_segment(str(cube), sigma=p["sig"], thr_manual=p["thr"], log=log)

    def act_rve(p):
        mask = DATA / f"cube_4.9um_{p['n']}_mask.npz"
        if p["n"] == 512 and not mask.is_file():
            mask = DATA / "cube_4.9um_512_mask.npz"
        if not mask.is_file():
            cands = sorted(DATA.glob("*mask.npz"))
            if not cands:
                raise FileNotFoundError("Сначала сделайте шаг 3 (маска не найдена).")
            mask = cands[0]
        gray = DATA / f"cube_4.9um_{p['n']}.npy"
        if p["n"] == 512 and not gray.is_file():
            gray = DATA / "cube_4.9um_512.npy"
        log(f"[4] RVE {p['rve']:.0f} мкм из {mask.name} ...")
        step_build_rve(str(mask), str(gray), voxel_um=p["vox"], size_um=p["rve"],
                       search_step=p["sstep"], taubin_iters=p["tau"], log=log)

    def act_all(p):
        act_montage(p); act_extract(p); act_segment(p); act_rve(p)

    def act_deps(_p=None):
        log("Зависимости Python:")
        for mod, ver, st in check_deps():
            log(f"  {mod:10s} {ver:12s} {st}")
        log("Установка: pip install -r requirements.txt")

    def act_open(_p=None):
        RVE_DIR.mkdir(exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(RVE_DIR))  # noqa
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(RVE_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(RVE_DIR)])
        except Exception as e:
            log(f"Не открыть папку: {e} ({RVE_DIR})")

    def on_error_popup():
        try:
            while True:
                line = log_q.get_nowait()
                if line == "__ERROR__":
                    messagebox.showerror("Ошибка", "Шаг упал — подробности в логе.")
                else:
                    log_box.insert("end", line + "\n")
                    log_box.see("end")
        except queue.Empty:
            pass
        root.after(300, on_error_popup)

    brow0 = brow
    btns["m"] = mkbtn("1 · Монтаж срезов (выбор ROI)", lambda: run_in_thread(act_montage), brow0)
    btns["e"] = mkbtn("2 · Вырезать куб 512³", lambda: run_in_thread(act_extract), brow0+1)
    btns["s"] = mkbtn("3 · Сегментация → маска", lambda: run_in_thread(act_segment), brow0+2)
    btns["r"] = mkbtn("4 · Собрать RVE500 (STL обеих фаз)", lambda: run_in_thread(act_rve), brow0+3)
    btns["a"] = mkbtn("▶ Всё сразу (шаги 1–4)", lambda: run_in_thread(act_all), brow0+4)
    brow0 += 5
    f2 = ttk.Frame(left); f2.grid(row=brow0, column=0, columnspan=3, sticky="ew", pady=4); brow0 += 1
    b = ttk.Button(f2, text="Зависимости", command=lambda: run_in_thread(act_deps)); b.pack(side="left", expand=True, fill="x", padx=1); btns["d"] = b
    b = ttk.Button(f2, text="Папка RVE", command=lambda: act_open()); b.pack(side="left", expand=True, fill="x", padx=1)
    b = ttk.Button(f2, text="Обновить", command=lambda: (refresh_status(), refresh_preview())); b.pack(side="left", expand=True, fill="x", padx=1)

    ttk.Label(left, text="Подсказка: шаги идут по порядку.\nГалочки ✅ показывают, что уже готово.\nДвойной клик по run_gui.bat — запуск в Windows.",
              foreground="gray", wraplength=300, justify="left").grid(row=brow0, column=0, columnspan=3, sticky="w", pady=6)

    refresh_status(); refresh_preview()
    root.after(150, flush_log)
    root.after(300, on_error_popup)
    log_box.insert("end", f"Проект: {PROJECT}\nПроверьте шаги 1–4 слева. Кнопка «Зависимости» покажет, что доставить.\n")
    root.mainloop()


if __name__ == "__main__":
    main_gui()
