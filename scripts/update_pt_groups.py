import torch
import json
from pathlib import Path
import sys

# Add src to path so we can import gnn_vuln
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_vuln.data.dataset_lm import CWE_GROUP_MAP, GROUP_VOCAB

def update_multiclass_pt():
    pt_file = PROJECT_ROOT / "data" / "processed" / "lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.pt"
    vocab_file = PROJECT_ROOT / "data" / "raw" / "bigvul" / "cwe_vocab.json"
    
    if not pt_file.exists():
        print(f"Error: {pt_file} not found")
        return
        
    print(f"\\n--- Updating {pt_file.name} ---")
    print(f"Loading vocab from {vocab_file}...")
    with open(vocab_file, "r") as f:
        cwe_vocab = json.load(f)
        
    id_to_cwe = {v: k for k, v in cwe_vocab.items()}
    
    print(f"Loading dataset...")
    result = torch.load(pt_file, weights_only=False)
    if len(result) == 3:
        data, slices, class_names = result
    else:
        data, slices = result
        class_names = None
        
    new_group_ids = []
    for c_id in data.y:
        c_id_val = c_id.item()
        
        if c_id_val == 0:
            cwe_str = id_to_cwe.get(c_id_val, "benign")
        else:
            cwe_str = id_to_cwe.get(c_id_val, "")
            
        if cwe_str == "benign":
            group_id = GROUP_VOCAB["benign"]
        elif cwe_str:
            group_name = CWE_GROUP_MAP.get(cwe_str, "")
            group_id = GROUP_VOCAB.get(group_name, -1) if group_name else -1
        else:
            group_id = -1
            
        new_group_ids.append(group_id)
        
    data.group_id = torch.tensor(new_group_ids, dtype=torch.long)
    slices['group_id'] = slices['y'].clone()
    
    unknown_count = new_group_ids.count(-1)
    print(f"Updated group_ids. Total: {len(new_group_ids)}, UNKNOWN (-1): {unknown_count}")
    
    print(f"Saving updated dataset...")
    if class_names is not None:
        torch.save((data, slices, class_names), pt_file)
    else:
        torch.save((data, slices), pt_file)
    print("Done multiclass pt!")


def update_group_pt():
    pt_file = PROJECT_ROOT / "data" / "processed" / "lm_dataset_megavul_group_unixcoder-base_ft_s3500r42.pt"
    if not pt_file.exists():
        print(f"\\nError: {pt_file} not found")
        return
        
    print(f"\\n--- Updating {pt_file.name} ---")
    print(f"Loading dataset...")
    result = torch.load(pt_file, weights_only=False)
    if len(result) == 3:
        data, slices, class_names = result
    else:
        print("No class names to update.")
        return
        
    replacements = {
        'access_control': 'broken_access_control',
        'configuration': 'security_misconfiguration',
        'cryptography': 'cryptographic_failures',
        'authentication': 'authentication_failures',
        'data_integrity': 'software_or_data_integrity_failures',
        'logging': 'logging_and_alerting_failures',
        'error_handling': 'mishandling_exceptional_conditions'
    }
    
    new_class_names = []
    for c in class_names:
        new_class_names.append(replacements.get(c, c))
        
    print(f"Old class names: {class_names}")
    print(f"New class names: {new_class_names}")
    
    print(f"Saving updated dataset...")
    torch.save((data, slices, new_class_names), pt_file)
    print("Done group pt!")


def main():
    update_multiclass_pt()
    update_group_pt()

if __name__ == "__main__":
    main()
