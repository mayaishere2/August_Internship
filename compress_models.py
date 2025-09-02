import os, pickle, joblib

IN_DIR = "models"          # folder that has your .pkl models
OUT_DIR = "models_joblib"  # compressed models will go here

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isdir(IN_DIR):
    raise FileNotFoundError(
        f"'{IN_DIR}' folder not found. Create it and put your .pkl models inside it."
    )

count = 0
for fn in os.listdir(IN_DIR):
    if not fn.lower().endswith(".pkl"):
        continue
    in_path = os.path.join(IN_DIR, fn)
    out_path = os.path.join(OUT_DIR, fn[:-4] + ".joblib")
    with open(in_path, "rb") as f:
        model = pickle.load(f)
    joblib.dump(model, out_path, compress=3)  # try 3..5 for size vs speed
    print("Wrote", out_path)
    count += 1

if count == 0:
    print("No .pkl files found in 'models' — nothing to compress.")
else:
    print(f"Done. Compressed {count} model(s) into '{OUT_DIR}'.")
