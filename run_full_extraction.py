"""
Run full feature extraction (safe-by-default).
Creates per-batch CSVs in <PROJECT_ROOT>/features_parts/

To run full extraction, edit RUN_ALL = True below or pass an env var.
"""
import os
import math
import pandas as pd
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project paths (must match the notebook)
PROJECT_ROOT = r"c:\Users\mubva\Downloads\Nlp"
MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'sesotho_tone_manifest.csv')
FULL_FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features_parts')
FULL_FEATURES_INDEX = os.path.join(PROJECT_ROOT, 'features_parts_index.csv')

os.makedirs(FULL_FEATURES_DIR, exist_ok=True)

# Config
BATCH_SIZE = 200  # files per batch
N_WORKERS = min(8, (os.cpu_count() or 4))
# Safety: default to not run ALL batches. Set RUN_ALL=True to run the entire manifest.
RUN_ALL = False
MAX_BATCHES = 2  # when RUN_ALL is False, run at most MAX_BATCHES batches

print('Using Python:', os.sys.executable)
print('PROJECT_ROOT:', PROJECT_ROOT)
print('Manifest:', MANIFEST_PATH)
print('Output parts dir:', FULL_FEATURES_DIR)
print('BATCH_SIZE:', BATCH_SIZE, 'N_WORKERS:', N_WORKERS)
print('RUN_ALL:', RUN_ALL, 'MAX_BATCHES(when not running all):', MAX_BATCHES)

if not os.path.exists(MANIFEST_PATH):
    raise SystemExit(f"Manifest not found at: {MANIFEST_PATH}")

manifest_df = pd.read_csv(MANIFEST_PATH)
file_list = manifest_df['filepath'].tolist()
n_files = len(file_list)
print(f'Total files in manifest: {n_files}')


# local helpers

def _estimate_f0_local(y, sr):
    try:
        # try using estimate_f0 from notebook if present (not available here), so use pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        if f0 is not None and not np.all(np.isnan(f0)):
            return float(np.nanmedian(f0)), float(np.nanmean(f0))
    except Exception:
        pass
    # fallback
    try:
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        mag_thresh = np.median(mags[mags>0]) if np.any(mags>0) else 0
        pitched = pitches[mags > mag_thresh]
        if pitched.size > 0:
            return float(np.median(pitched)), float(np.mean(pitched))
    except Exception:
        pass
    return None, None


def _extract_features_single(fp):
    try:
        y, sr = librosa.load(fp, sr=16000, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        median_f0, mean_f0 = _estimate_f0_local(y, sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = float(np.mean(spec_cent)) if spec_cent.size else None
        row = {'filepath': fp, 'sr': int(sr), 'duration': duration, 'median_f0': median_f0, 'mean_f0': mean_f0, 'spec_cent_mean': spec_cent_mean}
        for i, v in enumerate(mfcc_mean, start=1):
            row[f'mfcc{str(i)}_mean'] = float(v)
        for i, v in enumerate(mfcc_std, start=1):
            row[f'mfcc{str(i)}_std'] = float(v)
        return (row, None)
    except Exception as e:
        return (None, (fp, str(e)))

# Run batches
n_batches = math.ceil(n_files / BATCH_SIZE) if n_files else 0
if not RUN_ALL:
    n_batches = min(n_batches, MAX_BATCHES)

print(f'Planned batches to run: {n_batches} (total manifest batches would be {math.ceil(n_files / BATCH_SIZE)})')

total_extracted = 0
errors = []
for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, n_files)
    batch_files = file_list[start:end]
    print(f'\nStarting batch {batch_idx+1}/{n_batches}: files {start}..{end-1} (count={len(batch_files)})')
    rows_out = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(_extract_features_single, fp): fp for fp in batch_files}
        for fut in as_completed(futures):
            row, err = fut.result()
            if row is not None:
                rows_out.append(row)
            if err is not None:
                errors.append(err)
    # Save batch
    if rows_out:
        batch_df = pd.DataFrame(rows_out)
        part_name = f'features_part_{batch_idx+1:03d}.csv'
        out_path = os.path.join(FULL_FEATURES_DIR, part_name)
        batch_df.to_csv(out_path, index=False)
        total_extracted += len(batch_df)
        print(f'  -> Saved {len(batch_df)} extracted rows to: {out_path}')
    else:
        print('  -> No rows extracted in this batch.')

# Write index
parts = sorted([p for p in os.listdir(FULL_FEATURES_DIR) if p.startswith('features_part_') and p.endswith('.csv')])
idx_rows = []
for p in parts:
    pth = os.path.join(FULL_FEATURES_DIR, p)
    try:
        dfp = pd.read_csv(pth)
        idx_rows.append({'part_file': pth, 'rows': len(dfp)})
    except Exception:
        idx_rows.append({'part_file': pth, 'rows': None})
if idx_rows:
    pd.DataFrame(idx_rows).to_csv(FULL_FEATURES_INDEX, index=False)
    print(f'\nWritten parts index to: {FULL_FEATURES_INDEX}')

print(f'\nExtraction complete. Total extracted rows (across parts): {total_extracted}')
if errors:
    print(f'Encountered {len(errors)} errors during extraction. Sample:')
    for e in errors[:10]:
        print(' -', e)
    print('\nYou can inspect per-part CSVs in the features_parts directory and combine them later.')
else:
    print('No extraction errors reported.')

print('\nTo run the full extraction set RUN_ALL = True at the top of this script and re-run it.')
