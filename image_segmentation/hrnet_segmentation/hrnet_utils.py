import numpy as np
import cv2


def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def convert_label(label, inverse=False):
    ignore_label = -1
    label_mapping = {-1: ignore_label, 0: ignore_label, 
                     1: ignore_label, 2: ignore_label, 
                     3: ignore_label, 4: ignore_label, 
                     5: ignore_label, 6: ignore_label, 
                     7: 0, 8: 1, 9: ignore_label, 
                     10: ignore_label, 11: 2, 12: 3, 
                     13: 4, 14: ignore_label, 15: ignore_label, 
                     16: ignore_label, 17: 5, 18: ignore_label, 
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                     25: 12, 26: 13, 27: 14, 28: 15, 
                     29: ignore_label, 30: ignore_label, 
                     31: 16, 32: 17, 33: 18}

    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


def smooth_output(preds: np.ndarray, height: int, width: int) -> np.ndarray:
    result = np.zeros((1, 19, height, width))
    for i in range(19):
        result[0, i] = cv2.resize(
            preds[..., i].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    return result


def apply_palette_to_np_img(img, palette):
    h, w = img.shape[:2]
    result_img = np.zeros((h, w, 3)).astype(np.uint8)

    for i in range(256):
        r = palette[i*3]
        g = palette[i*3+1]
        b = palette[i*3+2]
        apply_idx = np.where(img == i)
        result_img[apply_idx] = [b, g, r]

    return result_img


def gen_preds_img_np(preds, height, width):
    palette = get_palette(256)
    preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
    # for now, we only expects one image per batch
    pred = convert_label(preds[0], inverse=True)
    pred = apply_palette_to_np_img(pred, palette)
    return pred


def gen_preds_img(preds, height, width):
    palette = get_palette(256)
    palette = np.array(palette, dtype=np.uint8).reshape(-1, 3)
    preds = np.argmax(preds, axis=1).astype(np.int32)
    pred = convert_label(preds[0], inverse=True).astype(np.int32)
    h, w = pred.shape
    gen_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        gen_img[:, :, i] = palette[np.clip(pred, 0, 255), 2 - i]
    gen_img = cv2.resize(gen_img, (width, height), interpolation=cv2.INTER_NEAREST)
    return gen_img



def save_pred(preds, sv_fname, height, width):
    save_img = gen_preds_img(preds, height, width)
    cv2.imwrite(sv_fname, save_img)
