# Вертикальный фон приглашения 9:16 из горизонтального макета src.png.
# В фоне: конфетти, логотип, «ПРИГЛАШЕНИЕ», «в Ричи Клуб!», сцена с альбатросом.
import cv2, numpy as np
im = cv2.imread('src.png')[:, :, ::-1].astype(np.float32)
BG = np.array([205, 236, 252], np.float32)
diff = np.abs(im - BG).sum(2)
lum = im.mean(2)

# Области текста (y0, y1, x0, x1) в исходнике
R = {
    'head':  (92, 218, 380, 1548),
    'b1': (312, 428, 120, 1530), 'b2': (428, 545, 120, 1205),
    'b3': (545, 672, 120, 1458), 'b4': (672, 812, 120, 1185),
    'dl': (880, 990, 120, 485), 'tl': (992, 1105, 120, 485),
    'place': (1150, 1280, 120, 1110),
    'heart': (1295, 1475, 120, 1525),
}
mask = np.zeros(lum.shape, np.uint8)
for y0, y1, x0, x1 in R.values():
    sub = (diff[y0:y1, x0:x1] > 40) & (lum[y0:y1, x0:x1] < 215)
    mask[y0:y1, x0:x1] |= sub.astype(np.uint8) * 255
mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
clean = cv2.inpaint(im[:, :, ::-1].astype(np.uint8), mask, 9, cv2.INPAINT_TELEA)[:, :, ::-1].astype(np.float32)

def darkest(y0, y1, x0, x1):
    P = im[y0:y1, x0:x1].reshape(-1, 3); return P[P.mean(1).argmin()]

def matte(key, T):
    y0, y1, x0, x1 = R[key]
    P, B = im[y0:y1, x0:x1], clean[y0:y1, x0:x1]
    d = T - B
    a = ((P - B) * d).sum(2) / np.maximum((d * d).sum(2), 1)
    a = np.clip(a, 0, 1); a[a < 0.06] = 0
    return a

NAVY = np.array([30, 38, 92], np.float32)
GREY = darkest(*R['place']).astype(np.float32)
y0, y1, x0, x1 = R['heart']
P = im[y0:y1, x0:x1].reshape(-1, 3)
GREEN = P[(P[:, 1] - P[:, 0]).argmax()].astype(np.float32)
print('grey', GREY, 'green', GREEN)

W, H = 1600, 2844
out = np.ones((H, W, 3), np.float32) * BG

def paste_matte(a, T, x, y):
    h, w = a.shape
    reg = out[y:y + h, x:x + w]
    reg[:] = reg * (1 - a[..., None]) + T * a[..., None]

def paste_rect(x0, y0, x1, y1, x, y):
    out[y:y + y1 - y0, x:x + x1 - x0] = im[y0:y1, x0:x1]

DX = -30
# сцена снизу
sx0, sy0 = 1400, 60
scene = clean[sy0:, sx0:]
s = W / scene.shape[1]
scene = cv2.resize(scene, (W, int(round(scene.shape[0] * s))), interpolation=cv2.INTER_AREA)
SY = H - scene.shape[0]
out[SY:] = scene
fade = 330
t = np.linspace(1, 0, fade)[:, None] ** 1.4 * np.ones((1, W))
# шары не растворяем: маска по эллипсам (координаты исходника)
keep = np.zeros((H, W), np.float32)
for cx, cy, r in [(2790, 285, 190), (3000, 370, 190), (3085, 535, 175)]:
    cv2.circle(keep, (int((cx - sx0) * s), int(SY + (cy - sy0) * s)), int(r * s), 1, -1)
keep = cv2.GaussianBlur(keep, (0, 0), 18)
t = t * (1 - keep[SY:SY + fade])
out[SY:SY + fade] = out[SY:SY + fade] * (1 - t[..., None]) + BG * t[..., None]
# левый край сцены уже светлый; затемнения нет

paste_rect(20, 10, 140, 165, 20 + DX + 30, 10)         # конфетти
paste_rect(138, 76, 302, 272, 138 + DX, 76)             # логотип
paste_matte(matte('head', NAVY), NAVY, 380 + DX, 92)    # ПРИГЛАШЕНИЕ

LINE = (84, 140, 178)
def line(xa, xb, y): cv2.line(out, (xa, y), (xb, y), LINE, 3, cv2.LINE_AA)

# Остальной текст рисует страница (index.html). Здесь только «в Ричи Клуб!»:
# в исходнике базовая линия ≈ 775, верх области 672 → смещение 103
CLUB_BASE = 900
paste_matte(matte('b4', NAVY), NAVY, 120 + DX, CLUB_BASE - 103)
res = np.clip(out, 0, 255).astype(np.uint8)[:, :, ::-1]
cv2.imwrite('card_v2.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite('/private/tmp/claude-501/-Users-user-Desktop--------------/25c2935c-434c-4b0b-a4d2-af8c05f88adb/scratchpad/v.png', cv2.resize(res, (600, 1066), interpolation=cv2.INTER_AREA))
