REGION_COLORS = {
    "R0": "#3c8f38",   # Green
    "R1": "#37a0f1",   # Blue
    "R2": "#f5181c",   # Red
    "R3": "#c7b214",   # Yellow/Gold
    "R4": "#ecb5ee",   # Light Pink
    "R5": "#883edd",   # Purple
    "R6": "#ff8419",   # Orange
    "R7": "#e750e4",   # Magenta
    "R8": "#a56a2f",   # Brown
    "R9": "#1ae910",   # Bright Green
    "R10": "#2c39a8",  # Dark Blue
    "R11": "#08d2a6",  # Teal
    "R12": "#dfff11",  # Yellow
}

COMBINED_REGION_MAPPING = {
    "background": 0,
    "region0": 1,
    "region1": 2,
    "region2": 3,
    "region3": 4,
    "region45": [5, 6],
    "region6": 7,
    "region78": [8, 9],
    "region9_12": [10, 11, 12, 13],
}

COARSE_REGION_GROUPS = {
    "region45": [5, 6],
    "region78": [8, 9],
    "region9_12": [10, 11, 12, 13],
}

REGION9_12_CORONAL_ORDER = ["region12", "region11", "region10", "region9"]
PCI_REGION_MAPPING = {
    "background": 0,
    "region0": 1,
    "region1": 2,
    "region2": 3,
    "region3": 4,
    "region4": 5,
    "region5": 6,
    "region6": 7,
    "region7": 8,
    "region8": 9,
    "region9": 10,
    "region10": 11,
    "region11": 12,
    "region12": 13,
}

TOTALSEG_STRUCTURES = {
    "duodenum",
    "hip_left",
    "hip_right",
    "small_bowel",
}

COARSE_SPLIT_STRUCTURES = {
    "region45": "hip_left",
    "region78": "hip_right",
}

PCI_REGION_NAMING = {"Background", "R0: Central", "R1: Right upper", "R2: Epigastrium", "R3: Left upper", "R4: Left flank", 
        "R5: Left lower", "R6: Pelvis", "R7: Right lower", "R8: Right flank", 
        "R9: Upper jejunum", "R10: Lower jejunum", "R11: Upper ileum", "R12: Lower ileum"}