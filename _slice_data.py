"""临时脚本：按行数切片数据集到指定比例，保持 jsonl 格式。"""
import sys


def slice_file(src, dst, fraction=0.05):
    total = 0
    with open(src, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    n = max(int(total * fraction), 1)
    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i >= n:
                break
            g.write(line)
    size_mb = round(100 * n / total, 1)
    print(f"{src}: {total} 行 -> {n} 行（{size_mb}%），输出 {dst}")


if __name__ == "__main__":
    base = "dataset"
    slice_file(f"{base}/pretrain_t2t_mini.jsonl", f"{base}/pretrain_t2t_mini_sub.jsonl", 0.05)
    slice_file(f"{base}/sft_t2t_mini.jsonl", f"{base}/sft_t2t_mini_sub.jsonl", 0.05)
