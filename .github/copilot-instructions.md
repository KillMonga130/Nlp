# Sesotho Tone Extraction Project - AI Agent Instructions

## Project Overview
This is a machine learning research project for **automatic tone extraction from Sesotho language audio recordings**. The goal is to classify tone patterns (Lexical vs Subject Marker) using acoustic features extracted from WAV files.

**Key Context**: This is an academic NLP project (course: ALP9X02) with a tight deadline (Oct 24, 2025). The codebase is primarily Jupyter notebook-based with supporting Python scripts for batch processing.

## Architecture & Data Flow

### Data Organization (Critical!)
The project has a **specific hierarchical structure** with hashed folder names that must be preserved:

```
c:\Users\mubva\Downloads\Nlp\
├── Minimal Pairs Recordings_6a9c3d7b8b9da1b7ea304bf2a54f32ae/  # Primary dataset (208 files)
│   └── Minimal Pairs/
│       ├── Lexical/           # Tone category 1
│       └── Subject Marker/    # Tone category 2
├── Processed Recordings_1c1ebd100cbabdb5d39320b816b39f42/  # Secondary dataset (844 files)
│   └── Processed Recordings/
│       ├── Free State/        # Regional groupings
│       ├── Soweto/
│       └── Vaal/
├── Tone Marked Dictionary_d0539665bcbf3b53c3391d54b7653569/  # Reference PDFs
└── sesotho_tone_extraction_project/  # Code & docs
    ├── sesotho_tone_extraction.ipynb  # Main workflow
    ├── README.md                       # Operational playbook
    └── PROJECT_ROADMAP.md             # High-level plan
```

**File Naming Convention**: `{SPEAKER}_{LOCATION}_{DATE}_{SEGMENT}.wav`
- Example: `KM_FS_19_04_S11.wav` → Speaker KM, Free State, April 2019, Segment 11
- This metadata is parsed and stored in the manifest for speaker-independent splitting

### Pipeline Flow
1. **Manifest Creation** → `sesotho_tone_manifest.csv` (2106 rows): Catalogs all audio files with parsed metadata
2. **Feature Extraction** → `features_parts/features_part_*.csv`: Batched acoustic feature extraction (F0, MFCC, spectral)
3. **Model Training** → Supervised classification (Random Forest, SVM, or deep learning)
4. **Evaluation** → Per-speaker, per-region, cross-validation metrics

## Critical Librosa Patterns (Common Pitfalls!)

### ⚠️ Keyword-Only Arguments
**ALL** `librosa.feature.*` functions require keyword arguments (Python 3.11+):

```python
# ❌ WRONG - Will fail with "takes 0 positional arguments"
mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)

# ✅ CORRECT - Use keyword arguments
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
```

### F0 Extraction Strategy (2-Tier Fallback)
Pitch extraction can fail on unvoiced segments. Use this robust pattern:

```python
def estimate_f0(y, sr):
    # Try 1: pyin (better for voiced speech)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                      fmax=librosa.note_to_hz('C7'))
        if f0 is not None and not np.all(np.isnan(f0)):
            return float(np.nanmedian(f0)), float(np.nanmean(f0))
    except Exception:
        pass
    
    # Fallback: piptrack
    try:
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        mag_thresh = np.median(mags[mags>0]) if np.any(mags>0) else 0
        pitched = pitches[mags > mag_thresh]
        if pitched.size > 0:
            return float(np.median(pitched)), float(np.mean(pitched))
    except Exception:
        pass
    
    return None, None  # Unvoiced or error
```

### Standard Preprocessing
Always resample to **16 kHz** for feature consistency:

```python
y, sr = librosa.load(filepath, sr=16000, mono=True)
duration = librosa.get_duration(y=y, sr=sr)
```

## Feature Extraction Standards

### Required Features Per File (26 dimensions + metadata)
```python
{
    'filepath': str,
    'sr': int,
    'duration': float,
    'median_f0': float,        # Primary tone indicator
    'mean_f0': float,
    'spec_cent_mean': float,
    'mfcc1_mean': float, ..., 'mfcc13_mean': float,   # 13 MFCC means
    'mfcc1_std': float, ..., 'mfcc13_std': float      # 13 MFCC stds
}
```

### Batch Processing Pattern
The project uses **ThreadPoolExecutor** for parallel extraction to handle 2000+ files:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

BATCH_SIZE = 200  # Files per batch (memory consideration)
N_WORKERS = min(8, os.cpu_count() or 4)

with ThreadPoolExecutor(max_workers=N_WORKERS) as exe:
    futures = {exe.submit(_extract_features_single, fp): fp for fp in batch_files}
    for fut in as_completed(futures):
        row, err = fut.result()
        # Handle results...
```

**Output Convention**: Save batches as `features_parts/features_part_{batch_idx:03d}.csv` with an index file `features_parts_index.csv` tracking row counts.

## Machine Learning Workflow

### Label Assignment
Labels are derived from folder structure:
```python
# Manifest has 'category' column: 'Lexical' | 'Subject Marker' | None
manifest['label'] = manifest['category'].apply(
    lambda v: 'lexical' if v and 'lexical' in v.lower() 
              else ('subject_marker' if v else None)
)
```

### Train/Test Splitting (Speaker-Independent!)
**Critical**: Use `GroupShuffleSplit` by speaker to prevent data leakage:

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(gss.split(manifest, manifest['label'], 
                                      groups=manifest['speaker']))
```

This ensures test speakers are completely unseen during training, simulating real-world generalization.

### Baseline Model (Random Forest)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
```

**Persist scaler for inference**: `joblib.dump(scaler, 'models/scaler.joblib')`

## Development Workflows

### Environment Setup (Conda)
```powershell
C:/ProgramData/anaconda3/Scripts/conda.exe create -n sesotho-tone python=3.11 -y
conda activate sesotho-tone
C:/ProgramData/anaconda3/Scripts/conda.exe install -n sesotho-tone -c conda-forge librosa pysoundfile soxr numba -y
C:/ProgramData/anaconda3/Scripts/conda.exe install -n sesotho-tone numpy pandas scipy scikit-learn matplotlib seaborn -y
```

**Why conda-forge**: Audio libraries (`librosa`, `pysoundfile`) are better maintained on conda-forge channel.

### Running Full Extraction
```powershell
# From project root (c:\Users\mubva\Downloads\Nlp)
python run_full_extraction.py
```

**Default Safety**: Script only processes 2 batches (400 files). Set `RUN_ALL = True` to process all 2106 files.

### Notebook Execution Order
1. **Environment Setup** (Cell 2) → Set `PROJECT_ROOT` and verify paths
2. **Manifest Creation** (Cell 8) → Generates `sesotho_tone_manifest.csv`
3. **Feature Extraction** (Cells 9-14) → Sample/debug/full extraction
4. **Model Training** (Future cells) → Load features, split, train, evaluate

## Project-Specific Conventions

### Path Handling
- **Always use raw strings** for Windows paths: `r"c:\Users\mubva\Downloads\Nlp"`
- **Absolute paths required** in manifest and feature CSVs
- **Relative paths** stored separately in manifest for portability

### Error Handling in Extraction
Return tuples for success/failure:
```python
def _extract_features_single(fp):
    try:
        # ... extraction logic ...
        return (row_dict, None)  # Success
    except Exception as e:
        return (None, (fp, str(e)))  # Failure
```

Collect errors separately and log sample (first 10) at end of batch.

### Outputs Location
All artifacts go in project root (`c:\Users\mubva\Downloads\Nlp`), not inside notebook folder:
- `sesotho_tone_manifest.csv`
- `features_parts/` directory
- `features_sample.csv`, `features_sample_debug.csv`
- Future: `models/`, `reports/`

## Common Gotchas & Solutions

1. **"mfcc() takes 0 positional arguments"** → Use keyword args: `librosa.feature.mfcc(y=y, sr=sr)`
2. **All-NaN F0 values** → Unvoiced segments; use 2-tier fallback (pyin → piptrack)
3. **Memory issues on full extraction** → Batch processing (200 files), save incrementally
4. **Speaker data leakage** → Use `GroupShuffleSplit` by speaker, not random split
5. **Inconsistent sample rates** → Always resample to 16 kHz: `librosa.load(fp, sr=16000)`

## Key Files to Reference

- **`sesotho_tone_extraction_project/README.md`**: Operational playbook with commands, schema, troubleshooting
- **`run_full_extraction.py`**: Production-ready batch extraction script (thread-safe)
- **`sesotho_tone_manifest.csv`**: Source of truth for all audio file metadata
- **`PROJECT_ROADMAP.md`**: High-level research plan and timeline (4-week sprint)

## When Adding New Features

1. Check existing manifest schema before adding columns
2. Add new feature to `_extract_features_single()` function
3. Update feature column list in README documentation
4. Rerun extraction on sample batch first (`features_sample_debug.csv`)
5. Verify no NaN/Inf values in new feature before full run

## Model Architecture Progression (Next Phase Guidance)

### Phase 1: Baseline (CURRENT FOCUS) ✅
**Random Forest on aggregated features** (26 dimensions)
- Fast training (~5 minutes for 2000 samples)
- Interpretable feature importance
- Expected accuracy: 70-85%

### Phase 2: Sequence Modeling (PRIORITY - if time permits)

**LSTM for pitch contour sequences** - Best for tone extraction where F0 trajectory matters:

```python
# Extract frame-level features (not aggregated)
def extract_sequence_features(filepath, sr=16000, hop_length=512):
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    # F0 contour (time series)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                            fmax=librosa.note_to_hz('C7'), 
                            hop_length=hop_length)
    # MFCCs (13 coefficients × time frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    # Combine: shape (time_frames, 14 features)
    features = np.vstack([f0, mfcc]).T
    return features

# Model architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(None, 14)),
    Dropout(0.3),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')  # Binary: Lexical vs Subject Marker
])
```
**Expected improvement**: +5-10% accuracy over Random Forest
**Time required**: 1-2 days for data pipeline + training

### Phase 3: CNN on Spectrograms (MEDIUM PRIORITY)

**CNN for mel-spectrogram classification** - Good if spectral patterns differ between tone types:

```python
# Extract mel-spectrogram
def extract_melspectrogram(filepath, sr=16000, n_mels=128):
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db  # Shape: (128, time_frames)

# CNN architecture
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(128, time_frames)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
```
**Expected improvement**: +8-12% accuracy (if spectral patterns are distinct)
**Time required**: 2-3 days

### Phase 4: Transfer Learning with wav2vec 2.0 (FUTURE WORK)

**Pre-trained speech model embeddings** - State-of-the-art approach:

```python
# Requires: pip install transformers torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def extract_wav2vec_embeddings(filepath):
    y, sr = librosa.load(filepath, sr=16000, mono=True)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state
    return embeddings.mean(dim=1).numpy()  # Aggregate to fixed size
```
**Expected improvement**: +15-20% accuracy (state-of-the-art)
**Why defer**: Requires PyTorch, large model downloads (~360MB), longer training time
**Time required**: 3-4 days

### Architecture Selection Guide

**Use Random Forest** if:
- < 24 hours remaining until deadline
- Need interpretability for academic report
- Limited computational resources

**Use LSTM** if:
- 1-2 days available
- Pitch contour trajectory is key discriminator for Sesotho tones
- Want moderate improvement with manageable complexity

**Use CNN** if:
- 2-3 days available
- Spectral differences are visually evident in spectrograms
- Have GPU available for faster training

**Use wav2vec2** if:
- > 3 days available
- Aiming for publication-quality results
- Computational resources available (GPU recommended)

## Sesotho Tone System (Linguistic Context)

Sesotho is a **tone language** with two primary tone levels:
- **High tone (H)**: Marked in dictionaries with acute accent (á, é)
- **Low tone (L)**: Unmarked or marked with grave accent (à, è)

**Project Scope**: Binary classification
- `Lexical` tone: Inherent word tone (e.g., distinguishing homophones)
- `Subject Marker` tone: Grammatical tone marking subjects

**Reference Materials**: 16 Tone Marked Dictionary PDFs in `Tone Marked Dictionary_.../Dictionary/` provide linguistic ground truth for validation.

**Future Extensions**: The dictionary PDFs support:
- Multi-level tone classification (H, L, Rising, Falling)
- Tone sandhi rules (tone changes in context)
- Word-level tone pattern analysis

## Evaluation Thresholds & Success Criteria

**Minimum Acceptable Performance** (for project completion):
- Overall accuracy: **≥ 70%** (baseline Random Forest)
- Macro F1-score: **≥ 0.65** (handles class imbalance)
- Speaker-independent accuracy: **≥ 60%** (unseen speakers)

**Target Performance** (strong academic result):
- Overall accuracy: **≥ 85%**
- Macro F1-score: **≥ 0.80**
- Speaker-independent accuracy: **≥ 75%**

**Excellence Indicators** (publishable quality):
- Overall accuracy: **≥ 90%**
- Macro F1-score: **≥ 0.85**
- Per-region consistency: Max 5% accuracy variance across Free State/Soweto/Vaal

**Critical Evaluation Requirements**:
1. **No speaker leakage**: Use `GroupShuffleSplit` by speaker
2. **Per-class metrics**: Report precision/recall for both Lexical and Subject Marker
3. **Confusion matrix**: Document systematic error patterns
4. **Qualitative analysis**: Listen to ≥10 misclassified samples for linguistic insights
5. **Per-region analysis**: Compare Free State vs Soweto vs Vaal performance
