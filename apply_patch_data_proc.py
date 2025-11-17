from pathlib import Path
import re, sys

p = Path("src/data_processing.py")
if not p.exists():
    print("ERROR: src/data_processing.py not found"); sys.exit(1)
text = p.read_text(encoding="utf8")

# If already importing clr_transform from transforms, do nothing
if "from src.utils.transforms import clr_transform" in text:
    print("data_processing.py already imports clr_transform — no change made.")
    sys.exit(0)

# Find the def clr_transform(...) ... return clr block
m = re.search(r"def\\s+clr_transform\\s*\\([^\\)]*\\)\\s*:\\s*([\\s\\S]*?)\\n(?=def\\s+get_preprocessor|def\\s+main\\(|class\\s+)", text)
if not m:
    # fallback: try to find start and last 'return clr'
    start = text.find("def clr_transform(")
    if start == -1:
        print("Could not find clr_transform block. No changes made.")
        sys.exit(0)
    end = text.find("return clr", start)
    if end == -1:
        print("Could not find end of clr_transform block. No changes made.")
        sys.exit(1)
    end = text.find("\\n", end) + 1
    new_text = text[:start] + "from src.utils.transforms import clr_transform\\n\\n" + text[end:]
else:
    start = m.start()
    end = m.end()
    new_text = text[:start] + "from src.utils.transforms import clr_transform\\n\\n" + text[end:]

p.write_text(new_text, encoding="utf8")
print("Patched src/data_processing.py — now imports clr_transform from src.utils.transforms")
