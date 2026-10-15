"""阶段 5 失败案例诊断：把一个不好的预测按「体素来源」拆开着色。

一个不好的预测长什么样，决定下一步该修什么，所以每一类单独一种颜色：

- ``CAT_CORE``     绿：落在它最好的那个 GT 实例里的体素，真正画对的部分；
- ``CAT_OTHER_GT`` 橙：落在另一个 thing GT 实例里的体素，「没拆分」的直接证据；
- ``CAT_STUFF``    土黄：落在墙 / 地板 / 天花板的体素，「膨胀」的直接证据；
- ``CAT_UNLABELED``灰：落在没有实例标注的体素；
- ``CAT_MISSING``  蓝灰：GT 有、预测没有覆盖到的体素。

图上只画 ``pred | (best GT)`` 这些体素，所以一块「椅子 + 地板」看起来是 绿芯 + 土黄壳。
``object_categories`` / ``object_stats`` 是纯函数（CPU 单测见 ``tests/test_phase5_failures.py``），
作图与 PLY 由 ``tools/phase5_failures.py`` 驱动。算法模块不 import 本模块（AGENTS.md §2）。
"""
import numpy as np

CAT_UNLABELED, CAT_STUFF, CAT_CORE, CAT_OTHER_GT, CAT_MISSING = 0, 1, 2, 3, 4
CAT_NAMES = ('unlabeled', 'stuff (wall/floor/ceiling)', 'core (inside best GT)',
             'into another thing GT', 'missing (GT not covered)')
CAT_COLORS = np.array([[0.78, 0.78, 0.75],    # unlabeled
                       [0.94, 0.78, 0.45],    # stuff
                       [0.18, 0.62, 0.31],    # core
                       [0.92, 0.41, 0.20],    # into another object
                       [0.55, 0.63, 0.75]],   # missing
                      np.float32)
_RANK = np.array([0, 1, 4, 2, 3])             # 画图顺序：核心最后画，压在最上面


def thing_iou(pred, gt_vox, keep):
    """(K,) 预测与每个 thing GT 的体素 IoU（只数有实例标注的体素，与评估同口径）。"""
    gt_vox = np.asarray(gt_vox)
    n = float(np.count_nonzero(pred))
    out = np.zeros(len(keep), np.float64)
    for k, g in enumerate(keep):
        m = gt_vox == int(g)
        inter = float(np.count_nonzero(pred & m))
        out[k] = inter / max(n + float(np.count_nonzero(m)) - inter, 1.0)
    return out


def shares_of(pred, gt_vox, keep):
    """(K,) 预测的体素里有多大比例落在每个 thing GT 里（「混合」的判据）。"""
    gt_vox = np.asarray(gt_vox)
    n = max(int(np.count_nonzero(pred)), 1)
    return np.array([np.count_nonzero(pred & (gt_vox == int(g))) / n for g in keep], np.float64)


def best_thing_gt(pred, gt_vox, keep):
    """IoU 最大的 thing GT id（没有覆盖任何 thing 时返回 -1）。"""
    if len(keep) == 0 or not np.any(pred):
        return -1
    iou = thing_iou(pred, gt_vox, keep)
    if iou.max() <= 0:
        return -1
    return int(np.asarray(keep)[int(iou.argmax())])


def object_categories(pred, gt_vox, sem, keep, stuff, best=None):
    """每个体素的类别（见模块 docstring）；返回 ``(cat (V,) int8, best id)``。"""
    gt_vox, sem = np.asarray(gt_vox), np.asarray(sem)
    best = best_thing_gt(pred, gt_vox, keep) if best is None else int(best)
    is_thing = np.isin(gt_vox, np.asarray(keep, np.int64))
    stuff_ids = np.asarray(sorted(int(v) for v in stuff), np.int64)   # 允许传入 set
    cat = np.where(np.isin(sem, stuff_ids), CAT_STUFF, CAT_UNLABELED).astype(np.int8)
    cat[is_thing & (gt_vox != best)] = CAT_OTHER_GT
    cat[gt_vox == best] = CAT_CORE
    cat[(gt_vox == best) & ~np.asarray(pred)] = CAT_MISSING
    return cat, best


def object_stats(pred, gt_vox, sem, keep, stuff, best=None):
    """一个预测的体素来源拆解（诊断表的一行，只数这个预测自己的体素）。纯函数。

    前四类只在 ``pred`` 内计数（「这个预测吃进了什么」）；``missing`` 只数 GT 有而预测
    没有覆盖的体素（「这个物体漏掉了多少」）。
    """
    gt_vox = np.asarray(gt_vox)
    pred = np.asarray(pred)
    cat, best = object_categories(pred, gt_vox, sem, keep, stuff, best=best)
    n_gt = int(np.count_nonzero(gt_vox == best)) if best >= 0 else 0
    sh = shares_of(pred, gt_vox, keep)
    n = int(np.count_nonzero(pred))
    out = dict(best=int(best), n_pred=n, n_gt=n_gt, size_ratio=n / max(n_gt, 1),
               mixed=int(np.count_nonzero(sh >= 0.2)), max_other_share=float(np.sort(sh)[-2]) if len(sh) > 1 else 0.0)
    for cat_id, name, sel in ((CAT_CORE, 'core', pred), (CAT_OTHER_GT, 'other_gt', pred),
                              (CAT_STUFF, 'stuff', pred), (CAT_UNLABELED, 'unlabeled', pred),
                              (CAT_MISSING, 'missing', ~pred)):
        out[name] = int(np.count_nonzero(sel & (cat == cat_id)))
        out[name + '_share'] = out[name] / max(n, 1)
    return out


def select_and_subsample(cat, sel, max_points=20000, seed=0):
    """要画的体素下标：核心全留，其余随机抽；按类别排序让核心画在最上面。"""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return idx
    core = idx[cat[idx] == CAT_CORE]
    rest = np.setdiff1d(idx, core, assume_unique=False)
    if len(rest) > max(0, max_points - len(core)):
        rng = np.random.default_rng(seed)
        rest = rng.choice(rest, max(0, max_points - len(core)), replace=False)
    idx = np.concatenate([core, rest])
    return idx[np.argsort(_RANK[cat[idx]], kind='stable')]


def write_category_ply(path, xyz, cat, sel):
    """同一套着色的 PLY，便于在 MeshLab / CloudCompare 里转着看。"""
    from .pointcloud import write_ply
    idx = np.flatnonzero(sel)
    write_ply(path, np.asarray(xyz)[idx], CAT_COLORS[cat[idx]])


def object_png(path, xyz, cat, sel, title, note=None, max_points=20000, seed=0):
    """单个预测的诊断图：左 3D 视角，右俯视；按体素来源染色，附类别图例。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    xyz = np.asarray(xyz)
    idx = select_and_subsample(cat, sel, max_points=max_points, seed=seed)
    if len(idx) == 0:
        return
    col = CAT_COLORS[cat[idx]]
    fig = plt.figure(figsize=(12.0, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], c=col, s=5, linewidths=0, depthshade=False)
    ext = np.ptp(xyz[idx], 0); ax.set_box_aspect(ext / max(ext.max(), 1e-6))
    ax.view_init(elev=16, azim=-62)
    ax.set_axis_off(); ax.set_title('3D view', fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(xyz[idx, 0], xyz[idx, 1], c=col, s=3, linewidths=0)
    ax2.set_aspect('equal'); ax2.set_axis_off(); ax2.set_title('top-down (z up)', fontsize=10)
    present = [c for c in range(5) if np.any(cat[idx] == c)]
    ax2.legend(handles=[Line2D([], [], marker='o', ls='', ms=6, color=CAT_COLORS[c], label=CAT_NAMES[c])
                        for c in present], loc='upper right', fontsize=7.5, frameon=False)
    fig.suptitle(title if note is None else f'{title}\n{note}', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
