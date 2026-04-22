import sys
from pathlib import Path

student_dir = Path("student_code")
base_dir = Path("original_main") # The competition root directory

# 1. Check files that must be EXACT matches
exact_files = ["scripts/evaluate.py", "scripts/sim.py"]

for file_path in exact_files:
    student_file = student_dir / file_path
    base_file = base_dir / file_path
    
    if not student_file.exists():
        print(f"🚨 CHEAT DETECTED: {file_path} was deleted!")
        sys.exit(1)
        
    if student_file.read_text() != base_file.read_text():
        print(f"🚨 CHEAT DETECTED: {file_path} has been modified!")
        sys.exit(1)

# 2. Check the TOML file (Only lines 7 and 43 allowed to change)
toml_path = "config/level2.toml"
student_toml_file = student_dir / toml_path
base_toml_file = base_dir / toml_path

if not student_toml_file.exists():
    print(f"🚨 CHEAT DETECTED: {toml_path} was deleted!")
    sys.exit(1)

student_lines = student_toml_file.read_text().splitlines()
base_lines = base_toml_file.read_text().splitlines()

# If they added or removed lines, the line numbers would shift, which is a violation
if len(student_lines) != len(base_lines):
    print(f"🚨 CHEAT DETECTED: {toml_path} has a different number of lines!")
    sys.exit(1)

# Allowed lines are 1-indexed (7 and 48), so their list indices are 6 and 47
allowed_indices = {6, 47} 

for i, (s_line, b_line) in enumerate(zip(student_lines, base_lines)):
    if i not in allowed_indices and s_line != b_line:
        print(f"🚨 CHEAT DETECTED: {toml_path} was modified on line {i + 1}!")
        print(f"Expected: {b_line.strip()}")
        print(f"Found:    {s_line.strip()}")
        sys.exit(1)  # Ends the action

print("✅ Anti-cheat checks passed. No unauthorized modifications detected.")