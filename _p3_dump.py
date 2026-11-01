import json
p = "/mnt/c/Users/drk20/Downloads/homework-practice-03-linclass.ipynb"
nb = json.load(open(p, encoding="utf-8"))
print("cells:", len(nb["cells"]), "| nbformat:", nb.get("nbformat"), nb.get("nbformat_minor"))
print("kernel:", nb.get("metadata", {}).get("kernelspec"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"]).strip()
    head = src.splitlines()[0][:95] if src else "(empty)"
    print(f"{i:3d} {c['cell_type'][:4]:5s} lines={len(src.splitlines()):3d} out={len(c.get('outputs',[])):2d} | {head}")