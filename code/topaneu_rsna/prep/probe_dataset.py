"""
Step 0: inspect the real TopAneu data and emit topaneu_rsna/labels.json.

    python -m topaneu_rsna.prep.probe_dataset

Reads location_mapping.json / vessel_mapping.json / type_mapping.json, derives
the location -> vessel correspondence by name, assigns each vessel to one of
three coarse groups (the author's Model 1 predicts 3 vessel groups), and prints
anything it could not resolve so you can hand-fix labels.json once.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

POSTERIOR = re.compile(r"(basilar|\bba\b|vert|\bva\b|pca|pcom|sca|aica|pica|poster)", re.I)
MCA = re.compile(r"(\bmca\b|middle.?cerebral|m1|m2|m3)", re.I)


def _load_mapping(path: Path) -> dict:
    """Accept flat or dataset-style {'labels': {...}} mappings."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw.get("labels"), dict):
        raw = raw["labels"]
    out = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[int(v)] = str(k)
        elif isinstance(v, str) and str(k).lstrip("-").isdigit():
            out[int(k)] = v
        elif isinstance(v, dict) and "id" in v:
            out[int(v["id"])] = str(k)
        else:
            raise ValueError(f"Unrecognised mapping entry in {path}: {k!r}: {v!r}")
    return {k: v for k, v in sorted(out.items()) if k != 0}


def _ordered(mapping: dict) -> list:
    return [mapping[k] for k in sorted(mapping)]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def coarse_group(vessel_name: str) -> int:
    if POSTERIOR.search(vessel_name):
        return 1          # posterior + basilar
    if MCA.search(vessel_name):
        return 2          # MCA
    return 3              # other (ICA / ACA / AComm / ...)


def _location_vessel(location: str, vessels: list[str]) -> str | None:
    """Map the numbered TopAneu location taxonomy onto vessel regions."""
    vessel_set = set(vessels)
    side = "R" if location.startswith("R-") else "L" if location.startswith("L-") else None

    if "BA" in location:
        return "BA"
    if location == "4.1 Acom complex":
        return "Acom"
    if side is None:
        return None

    rules = (
        (r"VA-PICA|PICA", f"{side}-PICA"),
        (r"VA trunk", f"{side}-VA"),
        (r"AICA", f"{side}-AICA"),
        (r"SCA", f"{side}-SCA"),
        (r"P1P2", f"{side}-P1P2"),
        (r"P3P4", f"{side}-P3P4"),
        (r"infraclinoid C1-C5", f"{side}-ICA-C1-C5"),
        (r"C7-Pcom", f"{side}-Pcom"),
        (r"C7-AChA", f"{side}-AChA"),
        (r"ICA", f"{side}-ICA-C6-C7"),
        (r"A1|A2", f"{side}-A1A2"),
        (r"A3|Distal ACA", f"{side}-A3"),
        (r"M1 trunk|M1 early", f"{side}-M1"),
        (r"M1-M2|Distal-M2M3", f"{side}-M2"),
    )
    for pattern, vessel in rules:
        if re.search(pattern, location) and vessel in vessel_set:
            return vessel
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.LABELS_JSON)
    ap.add_argument("--n_probe", type=int, default=5)
    args = ap.parse_args()

    loc_map = _load_mapping(C.LOCATION_MAPPING)
    ves_map = _load_mapping(C.VESSEL_MAPPING)
    typ_map = _load_mapping(C.TYPE_MAPPING) if C.TYPE_MAPPING.exists() else {}

    locations = _ordered(loc_map)
    vessels = _ordered(ves_map)
    types = _ordered(typ_map)

    print(f"locations : {len(locations)}")
    print(f"vessels   : {len(vessels)}")
    print(f"types     : {len(types)}")

    # ---- location -> vessel: explicit taxonomy rules, then name matching
    vnorm = {_norm(v): v for v in vessels}
    l2v, unresolved = {}, []
    for loc in locations:
        mapped = _location_vessel(loc, vessels)
        if mapped is not None:
            l2v[loc] = mapped
            continue
        n = _norm(loc)
        if n in vnorm:
            l2v[loc] = vnorm[n]
            continue
        cands = [v for k, v in vnorm.items() if k and (k in n or n in k)]
        if len(cands) == 1:
            l2v[loc] = cands[0]
        elif cands:
            l2v[loc] = max(cands, key=lambda v: len(_norm(v)))
        else:
            unresolved.append(loc)

    groups = {v: coarse_group(v) for v in vessels}
    gc = Counter(groups.values())
    print(f"coarse groups: posterior={gc[1]} mca={gc[2]} other={gc[3]}")

    # ---- sanity-probe a few volumes
    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    print(f"cases     : {len(cases)}")
    for case in cases[: args.n_probe]:
        img, meta = uio.read(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}")
        line = [f"{case}: shape={img.shape} spacing={tuple(round(s,3) for s in meta['spacing'])}"]
        for tag, d in (("ves", C.VESSEL_MASKS), ("loc", C.LOCATION_MASKS), ("typ", C.TYPE_MASKS)):
            p = d / f"{case}{C.LABEL_SUFFIX}"
            if p.exists():
                a, _ = uio.read(p)
                u = np.unique(a)
                line.append(f"{tag}={u[u > 0].tolist()[:8]}")
            else:
                line.append(f"{tag}=MISSING")
        print("  " + "  ".join(line))

    jp = C.LOCATION_JSONS / f"{cases[0]}.json"
    if jp.exists():
        print(f"\nsample {jp.name}:\n{jp.read_text()[:600]}")

    spec = {
        "locations": locations,
        "vessels": vessels,
        "types": types,
        "location_to_vessel": l2v,
        "coarse_groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2))
    print(f"\nwrote {args.out}")

    if unresolved:
        print("\n!! could not map these locations onto a vessel class -- edit "
              "'location_to_vessel' in labels.json before continuing:")
        for u in unresolved:
            print(f"   {u}")
        print(f"   valid vessel names: {', '.join(vessels)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
