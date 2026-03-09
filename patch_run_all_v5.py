#!/usr/bin/env python3
"""
patch_run_all_v5.py
===================
Applies 4 targeted edits to run_all_v5.py to support oit367_final_dataset.csv.

Run from the OIT-367/ folder:
  python3 patch_run_all_v5.py

Backs up run_all_v5.py → run_all_v5.py.bak before patching.
Safe to re-run: checks whether each patch is already applied before modifying.
"""

import re, shutil, sys
from pathlib import Path

SRC = Path("run_all_v5.py")
BAK = Path("run_all_v5.py.bak")

if not SRC.exists():
    print("ERROR: run_all_v5.py not found. Run from OIT-367/ folder.")
    sys.exit(1)

shutil.copy(SRC, BAK)
print(f"Backup saved → {BAK}")

code = SRC.read_text()
original = code  # keep for diff at end


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — Add changelog entry for v6 dataset consolidation
# ─────────────────────────────────────────────────────────────────────────────
P1_MARKER = "v6 Add K"
if P1_MARKER not in code:
    # Insert after the last "v5 Add J" changelog line
    OLD = '    v5 Add J — lyric sentiment (VADER) added to longevity models'
    NEW = (
        '    v5 Add J — lyric sentiment (VADER) added to longevity models\n'
        '    v6 Add K — consolidated final dataset (build_final_dataset.py):\n'
        '               time_signature, is_male_artist, artist_age,\n'
        '               artist_scrobbles_log, artist_listeners_monthly_log,\n'
        '               is_mainstream_genre, artist_genre_count'
    )
    if OLD in code:
        code = code.replace(OLD, NEW, 1)
        print("PATCH 1 applied: changelog entry added")
    else:
        print("PATCH 1 skipped: changelog anchor not found (check manually)")
else:
    print("PATCH 1 already applied")


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — Load from oit367_final_dataset.csv when it exists,
#           fall back to rebuilding from source CSVs otherwise.
#           This preserves backward compatibility.
# ─────────────────────────────────────────────────────────────────────────────
P2_MARKER = "oit367_final_dataset.csv"
if P2_MARKER not in code:
    # Find the line that reads oit367_base_dataset.csv
    OLD = 'df = pd.read_csv(base_dataset_path)'
    NEW = (
        'final_dataset_path = data_dir / "oit367_final_dataset.csv"\n'
        'if final_dataset_path.exists():\n'
        '    print(f"Loading pre-built final dataset: {final_dataset_path}")\n'
        '    df = pd.read_csv(final_dataset_path)\n'
        'else:\n'
        '    print(f"Final dataset not found; rebuilding from base: {base_dataset_path}")\n'
        '    df = pd.read_csv(base_dataset_path)\n'
    )
    if OLD in code:
        code = code.replace(OLD, NEW, 1)
        print("PATCH 2 applied: final dataset loading logic added")
    else:
        # Try a slightly different known form
        OLD2 = "df = pd.read_csv(str(base_dataset_path))"
        if OLD2 in code:
            code = code.replace(OLD2, NEW.replace("base_dataset_path)", "base_dataset_path)"), 1)
            print("PATCH 2 applied (alt form): final dataset loading logic added")
        else:
            print("PATCH 2 skipped: read_csv anchor not found — add manually:")
            print("  Replace the line that reads oit367_base_dataset.csv with:")
            print("  " + NEW.replace("\n", "\n  "))
else:
    print("PATCH 2 already applied")


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 — Add new auto-detected feature candidates to the detection loop.
#           The existing loop already checks notna().mean() > 0.5, so new
#           features with <50% coverage are safely ignored.
# ─────────────────────────────────────────────────────────────────────────────
P3_MARKER = "is_male_artist"
if P3_MARKER not in code:
    # The loop currently checks: "lastfm_listeners_log", "is_us_artist"
    OLD = '    for col in ["lastfm_listeners_log", "is_us_artist"]:'
    NEW = (
        '    for col in ["lastfm_listeners_log", "is_us_artist",\n'
        '               "is_male_artist", "artist_age",\n'
        '               "artist_scrobbles_log", "artist_listeners_monthly_log",\n'
        '               "is_mainstream_genre", "time_signature"]:'
    )
    if OLD in code:
        code = code.replace(OLD, NEW, 1)
        print("PATCH 3 applied: new features added to auto-detection loop")
    else:
        print("PATCH 3 skipped: auto-detection loop anchor not found — add manually:")
        print('  Extend the for col in [...] list with:')
        print('  "is_male_artist", "artist_age", "artist_scrobbles_log",')
        print('  "artist_listeners_monthly_log", "is_mainstream_genre", "time_signature"')
else:
    print("PATCH 3 already applied")


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4 — Add new features to the NaN fill list so they don't silently
#           drop rows in models (conservative: unknown → 0 / median).
# ─────────────────────────────────────────────────────────────────────────────
P4_MARKER = "is_male_artist"
# Only apply if patch 3 worked (otherwise the features aren't in FEATURES yet)
if P4_MARKER in code:
    # Find the NaN fill block — it sets NaN to 0 for is_us_artist etc.
    OLD_FILL = '    for col in ["artist_peak_popularity", "artist_track_count", "lastfm_listeners_log", "is_us_artist"]:'
    NEW_FILL = (
        '    for col in ["artist_peak_popularity", "artist_track_count",\n'
        '               "lastfm_listeners_log", "is_us_artist",\n'
        '               "is_male_artist", "is_mainstream_genre",\n'
        '               "artist_scrobbles_log", "artist_listeners_monthly_log"]:'
    )
    if OLD_FILL in code:
        code = code.replace(OLD_FILL, NEW_FILL, 1)
        print("PATCH 4 applied: NaN fill list extended")
    else:
        # Try alt: fill with 0 for binary/log cols; median fill for artist_age
        print("PATCH 4 skipped: NaN fill anchor not found — add manually:")
        print('  Extend the NaN fill loop with:')
        print('  "is_male_artist", "is_mainstream_genre",')
        print('  "artist_scrobbles_log", "artist_listeners_monthly_log"')
        print('  Also add: df["artist_age"].fillna(df["artist_age"].median(), inplace=True)')


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 5 — Update the end-of-run print block to reflect v6 additions
# ─────────────────────────────────────────────────────────────────────────────
P5_MARKER = "v6 Add K"
if P5_MARKER not in code:
    OLD_PRINT = (
        '  print(f"  ✓ v5 Add J — lyric sentiment added to longevity models'
    )
    NEW_PRINT = (
        '  print(f"  ✓ v5 Add J — lyric sentiment added to longevity models'
        " (VADER; 41.9% charted coverage)\")\n"
        '  new_v6_feats = [c for c in ["is_male_artist","artist_age","artist_scrobbles_log",\n'
        '                               "artist_listeners_monthly_log","is_mainstream_genre",\n'
        '                               "time_signature"] if c in FEATURES]\n'
        '  if new_v6_feats:\n'
        '      print(f"  ✓ v6 Add K — consolidated dataset features active: {new_v6_feats}")'
    )
    # This patch is complex — just report what to add
    print("PATCH 5: manually add after the Add J print line:")
    print('  new_v6_feats = [c for c in ["is_male_artist","artist_age","artist_scrobbles_log",')
    print('                              "artist_listeners_monthly_log","is_mainstream_genre",')
    print('                              "time_signature"] if c in FEATURES]')
    print('  if new_v6_feats:')
    print('      print(f"  ✓ v6 Add K — consolidated dataset features active: {new_v6_feats}")')
else:
    print("PATCH 5 already applied")


# ─────────────────────────────────────────────────────────────────────────────
# WRITE PATCHED FILE
# ─────────────────────────────────────────────────────────────────────────────
if code != original:
    SRC.write_text(code)
    print(f"\nPatched file written → {SRC}")
    # Count lines changed
    orig_lines = original.splitlines()
    new_lines  = code.splitlines()
    print(f"Lines before: {len(orig_lines)}  Lines after: {len(new_lines)}")
else:
    print("\nNo changes made (all patches already applied or not found).")

print("\nDone. Verify with: python3 run_all_v5.py")
