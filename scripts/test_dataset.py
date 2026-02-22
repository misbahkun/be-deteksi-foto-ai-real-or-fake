"""Test API against CIFAKE dataset samples."""
import requests
import os
import json

API_URL = "http://127.0.0.1:8000/predict"
DATASET_BASE = "dataset/CIFAKE_RIL_OR_FAKE/test"

def test_folder(label, folder, count=5):
    path = os.path.join(DATASET_BASE, folder)
    files = sorted(os.listdir(path))[:count]
    correct = 0
    
    print(f"\n{'='*60}")
    print(f"  Testing {label} images ({folder}/)")
    print(f"{'='*60}")
    
    for fname in files:
        fpath = os.path.join(path, fname)
        with open(fpath, "rb") as f:
            resp = requests.post(API_URL, files={"file": (fname, f, "image/jpeg")})
        
        if resp.status_code != 200:
            print(f"  {fname:25s} -> ERROR {resp.status_code}: {resp.text[:80]}")
            continue
        
        d = resp.json()
        pred_label = d["label"]
        conf = d["confidence"]
        real_prob = d["probabilities"]["real"]
        time_ms = d["inference_time_ms"]
        
        # For REAL folder: correct if label == "real"
        # For FAKE folder: correct if label != "real" (any AI source)
        is_correct = (pred_label == "real") if label == "REAL" else (pred_label != "real")
        correct += int(is_correct)
        mark = "✓" if is_correct else "✗"
        
        print(f"  {mark} {fname:25s} -> {pred_label:18s} conf={conf:.3f}  real={real_prob:.3f}  {time_ms}ms")
    
    accuracy = correct / len(files) * 100
    print(f"\n  Accuracy: {correct}/{len(files)} ({accuracy:.0f}%)")
    return correct, len(files)

r_correct, r_total = test_folder("REAL", "REAL", count=10)
f_correct, f_total = test_folder("FAKE", "FAKE", count=10)

total_correct = r_correct + f_correct
total = r_total + f_total
print(f"\n{'='*60}")
print(f"  OVERALL: {total_correct}/{total} ({total_correct/total*100:.0f}%)")
print(f"{'='*60}")
