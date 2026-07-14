"""Parameter-derived dataset names — a cache key for callers without their own versioning.
Callers that version datasets themselves pass ds_name and never touch this."""
from __future__ import annotations

import hashlib
import json


def filter_suffix(cwe_list, cwe_groups, filter_owasp: bool = False,
                  filter_top25_dangerous: bool = False) -> str:
    if not cwe_list and not cwe_groups and not filter_owasp and not filter_top25_dangerous:
        return ""
    key = json.dumps(
        {"l": sorted(cwe_list or []), "g": sorted(cwe_groups or []),
         "owasp": filter_owasp, "top25": filter_top25_dangerous},
        sort_keys=True,
    )
    return "_f" + hashlib.md5(key.encode()).hexdigest()[:8]


def derive_ds_name(*, source: str, mode: str, lm_short: str, func_short: str,
                   add_func_tokens: bool, func_max_length: int, top_cwe: int,
                   fsuffix: str, max_per_class: int, resample_seed: int,
                   ds_name_suffix: str = "") -> str:
    ft = "_ft" if add_func_tokens else ""
    ml = f"_ml{func_max_length}" if add_func_tokens and func_max_length != 512 else ""
    top = f"_top{top_cwe}" if top_cwe > 0 else ""
    samp = f"_s{max_per_class}r{resample_seed}" if max_per_class > 0 else ""
    live = f"_live_{func_short}" if func_short != lm_short else ""
    return (f"lm_dataset_{source}_{mode}_{lm_short}{live}"
            f"{ft}{ml}{top}{fsuffix}{samp}{ds_name_suffix}")
