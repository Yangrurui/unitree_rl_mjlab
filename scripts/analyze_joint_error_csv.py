"""读取并分析 play_joint_error_log 导出的 CSV，统计每个关节的误差。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析关节误差 CSV，输出各关节统计信息。"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="CSV 文件路径（来自 play_with_error_log.py）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="将统计结果保存到指定文件（默认仅打印）",
    )
    parser.add_argument(
        "--unit",
        "-u",
        choices=["rad", "deg"],
        default="rad",
        help="统计输出单位：rad（弧度）或 deg（度），默认 rad",
    )
    return parser.parse_args()


def load_csv(csv_path: Path) -> tuple[np.ndarray, list[str], dict[str, tuple[int, int]]]:
    """加载 CSV，返回 (data, all_columns, joint_columns)。

    joint_columns: {joint_name: (rad_col_idx, deg_col_idx)}
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    # 识别关节列：*_rad 和 *_deg 成对
    joint_columns: dict[str, tuple[int, int]] = {}
    seen_rad = set()

    for i, col in enumerate(header):
        if col.endswith("_rad") and col != "step":
            base = col[:-4]  # 去掉 _rad
            deg_col = f"{base}_deg"
            if deg_col in header:
                j = header.index(deg_col)
                joint_columns[base] = (i, j)
                seen_rad.add(col)

    return data, header, joint_columns


def compute_joint_statistics(
    data: np.ndarray,
    joint_columns: dict[str, tuple[int, int]],
    unit: str,
) -> list[tuple[str, float, float, float, float, float]]:
    """计算每个关节的统计量：mean, std, max, rmse。

    返回 [(joint_name, mean, std, max, min, rmse), ...]
    """
    results = []
    rad_to_deg = np.degrees(1.0) if unit == "deg" else 1.0

    for joint_name, (rad_idx, deg_idx) in joint_columns.items():
        # 用 rad 列计算（更精确）
        vals = data[:, rad_idx].astype(np.float64)
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0:
            continue

        mean_val = np.mean(vals) * rad_to_deg
        std_val = np.std(vals) * rad_to_deg
        max_val = np.max(vals) * rad_to_deg
        min_val = np.min(vals) * rad_to_deg
        rmse_val = np.sqrt(np.mean(vals**2)) * rad_to_deg

        results.append(
            (joint_name, mean_val, std_val, max_val, min_val, rmse_val)
        )

    return results


def format_table(
    results: list[tuple[str, float, float, float, float, float]],
    unit: str,
    n_steps: int = 0,
    n_joints: int = 0,
) -> str:
    """格式化输出为表格字符串。"""
    u = "deg" if unit == "deg" else "rad"
    lines = [
        "",
        "=" * 85,
        "各关节误差统计",
        "=" * 85,
        f"{'关节名':<35} {'平均(MAE)':<12} {'标准差':<12} {'最大':<12} {'最小':<12} {'RMSE':<12}",
        f"{'':35} ({u}){'':<8} ({u}){'':<8} ({u}){'':<8} ({u}){'':<8} ({u})",
        "-" * 85,
    ]

    for name, mean_v, std_v, max_v, min_v, rmse_v in results:
        lines.append(
            f"{name:<35} {mean_v:<12.6f} {std_v:<12.6f} {max_v:<12.6f} "
            f"{min_v:<12.6f} {rmse_v:<12.6f}"
        )

    # 全关节平均
    n = len(results)
    avg_mean = sum(r[1] for r in results) / n if n else 0
    avg_std = sum(r[2] for r in results) / n if n else 0
    avg_max = sum(r[3] for r in results) / n if n else 0
    avg_min = sum(r[4] for r in results) / n if n else 0
    avg_rmse = sum(r[5] for r in results) / n if n else 0

    lines.extend([
        "-" * 85,
        f"{'全关节平均':<35} {avg_mean:<12.6f} {avg_std:<12.6f} {avg_max:<12.6f} "
        f"{avg_min:<12.6f} {avg_rmse:<12.6f}",
        "=" * 85,
        f"\n总步数: {n_steps} | 关节数: {n_joints}",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_file).expanduser().resolve()

    if not csv_path.exists():
        print(f"[ERROR] 文件不存在: {csv_path}", file=sys.stderr)
        return 1

    try:
        data, header, joint_columns = load_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] 读取 CSV 失败: {e}", file=sys.stderr)
        return 1

    n_steps = data.shape[0]
    n_joints = len(joint_columns)

    if n_joints == 0:
        print("[WARN] 未检测到关节列（*_rad, *_deg）", file=sys.stderr)
        return 1

    results = compute_joint_statistics(data, joint_columns, args.unit)
    table = format_table(results, args.unit, n_steps=n_steps, n_joints=n_joints)

    print(table)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"\n[INFO] 统计结果已保存到: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
