import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data.data import load_data, make_test_split
from evaluate import metrics
from llm.prompts import SYSTEM, USER_TEMPLATE
from llm.qwen3_back import ask_label_with_raw  # ← новая функция

def row_to_prompt(row: pd.Series) -> str:
    def v(x):
        if pd.isna(x): return "29999"
        try: return str(int(x))
        except Exception: return str(x)
    return USER_TEMPLATE.format(
        rr_interval=v(row["rr_interval"]), p_onset=v(row["p_onset"]),
        p_end=v(row["p_end"]), qrs_onset=v(row["qrs_onset"]),
        qrs_end=v(row["qrs_end"]), t_end=v(row["t_end"]),
        p_axis=v(row["p_axis"]), qrs_axis=v(row["qrs_axis"]), t_axis=v(row["t_axis"])
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Путь к CSV. Если не указан — ecg/data/ecg_data.csv")
    ap.add_argument("--logs_csv", default=None, help="Куда сохранить логи (CSV), напр. results/logs.csv")
    ap.add_argument("--with_prompts", action="store_true", help="Сохранять текст промптов в логи")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(__file__).parent / "data" / "ecg_data.csv"
    df = load_data(str(csv_path))
    X_test, y_test = make_test_split(df)

    rows = []
    t0 = time.perf_counter()
    for idx, row in X_test.reset_index(drop=True).iterrows():
        prompt = row_to_prompt(row)
        t1 = time.perf_counter()
        label, raw = ask_label_with_raw(SYSTEM, prompt)   # метка и сырой ответ
        dt = time.perf_counter() - t1
        rec = {
            "index": idx,
            "y_true": int(y_test.iloc[idx]),
            "y_pred": int(label),
            "latency_sec": round(dt, 4),
        }
        if args.with_prompts:
            rec["prompt"] = prompt
            rec["raw_response"] = raw
        rows.append(rec)
    total_dt = time.perf_counter() - t0

    preds = np.array([r["y_pred"] for r in rows])
    result = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "n_test": int(len(y_test)),
        "total_time_sec": round(total_dt, 2),
        "avg_latency_sec": round(total_dt / max(1, len(rows)), 4),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.logs_csv:
        out_path = Path(args.logs_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        print(f"[saved] logs -> {out_path}")

if __name__ == "__main__":
    main()
