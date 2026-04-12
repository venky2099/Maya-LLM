import verify_provenance  # Maya Research Series -- Nexus Learning Labs, Bengaluru
verify_provenance.stamp()  # logs canary + ORCID on every run
"""
run_ablation_llm.py — Maya-LLM Ablation Study v2
Nexus Learning Labs — Maya Research Series

Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha
"""
CONDITIONS = {
    "A": {"bhaya": False, "vairagya": False, "karma": False, "prana": False},
    "B": {"bhaya": True,  "vairagya": False, "karma": False, "prana": False},
    "C": {"bhaya": False, "vairagya": True,  "karma": False, "prana": False},
    "D": {"bhaya": False, "vairagya": False, "karma": True,  "prana": False},
    "E": {"bhaya": True,  "vairagya": True,  "karma": False, "prana": False},
    "F": {"bhaya": True,  "vairagya": True,  "karma": True,  "prana": True},  # canonical
}
if __name__ == "__main__":
    for cond, flags in CONDITIONS.items():
        star = " ★ canonical" if cond == "F" else ""
        print(f"  Condition {cond}: {flags}{star}")