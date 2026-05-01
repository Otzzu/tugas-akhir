import torch
import json
from pathlib import Path
import sys

# Add src to path so we can import gnn_vuln
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_vuln.data.dataset_lm import CWE_GROUP_MAP, GROUP_VOCAB

def main():
    pt_file = PROJECT_ROOT / "data" / "processed" / "lm_dataset_bigvul_multiclass_unixcoder-base_ft_top10.pt"
    vocab_file = PROJECT_ROOT / "data" / "raw" / "bigvul" / "cwe_vocab.json"
    
    if not pt_file.exists():
        print(f"Error: {pt_file} not found")
        sys.exit(1)
        
    print(f"Loading vocab from {vocab_file}...")
    with open(vocab_file, "r") as f:
        cwe_vocab = json.load(f)
        
    # Create reverse lookup: cwe_id (int) -> cwe_str (e.g. "CWE-119")
    id_to_cwe = {v: k for k, v in cwe_vocab.items()}
    
    print(f"Loading dataset from {pt_file}...")
    result = torch.load(pt_file, weights_only=False)
    if len(result) == 3:
        data, slices, class_names = result
    else:
        data, slices = result
        class_names = None
        
    print(f"Total nodes in data.y: {len(data.y)}")
    
    # We need to map data.y to data.group_id
    new_group_ids = []
    
    # The dataset merges all graphs into one giant 'data' object.
    # The 'y' tensor should have one element per graph.
    # We can just iterate over the y tensor.
    for c_id in data.y:
        c_id_val = c_id.item()
        
        # Determine group_id
        if c_id_val == 0: # Usually benign is 0, let's verify with id_to_cwe
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
        
    # Convert back to tensor
    data.group_id = torch.tensor(new_group_ids, dtype=torch.long)
    
    # CRITICAL: We also need to add group_id to the slices dictionary!
    # Since group_id has the exact same shape/granularity as y (one per graph),
    # we can just copy the slices from y.
    slices['group_id'] = slices['y'].clone()
    
    # Verify counts
    unknown_count = new_group_ids.count(-1)
    print(f"Updated group_ids. Total: {len(new_group_ids)}, UNKNOWN (-1): {unknown_count}")
    
    # Save back
    print(f"Saving updated dataset to {pt_file}...")
    if class_names is not None:
        torch.save((data, slices, class_names), pt_file)
    else:
        torch.save((data, slices), pt_file)
        
    print("Done!")

if __name__ == "__main__":
    main()
