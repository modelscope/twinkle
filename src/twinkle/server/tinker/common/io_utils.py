import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_SAVE_DIR = os.environ.get('TWINKLE_DEFAULT_SAVE_DIR', './outputs')

def get_base_dir() -> Path:
    return Path(DEFAULT_SAVE_DIR)

def get_model_dir(model_id: str) -> Path:
    return get_base_dir() / model_id

def get_dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob('*'):
            if p.is_file():
                total += p.stat().st_size
    return total

def save_train_info(model_id: str, info: Dict[str, Any]):
    model_dir = get_model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = model_dir / "tinker_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(info, f, indent=2)

def load_train_info(model_id: str) -> Optional[Dict[str, Any]]:
    metadata_path = get_model_dir(model_id) / "tinker_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def update_train_info(model_id: str, updates: Dict[str, Any]):
    info = load_train_info(model_id)
    if info:
        info.update(updates)
        save_train_info(model_id, info)

def save_checkpoint_info(model_id: str, checkpoint_info: Dict[str, Any]):
    info = load_train_info(model_id)
    if not info:
        return
    
    checkpoints = info.get("checkpoints", [])
    # Update existing or append new
    existing_idx = next((i for i, c in enumerate(checkpoints) if c['checkpoint_id'] == checkpoint_info['checkpoint_id']), -1)
    if existing_idx >= 0:
        checkpoints[existing_idx] = checkpoint_info
    else:
        checkpoints.append(checkpoint_info)
    
    info['checkpoints'] = checkpoints
    info['last_checkpoint'] = checkpoint_info
    save_train_info(model_id, info)

def load_checkpoint_info(model_id: str) -> List[Dict[str, Any]]:
    info = load_train_info(model_id)
    return info.get("checkpoints", []) if info else []

def list_training_runs(limit: int = 20, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    base_dir = get_base_dir()
    if not base_dir.exists():
        return [], 0
    
    # Filter directories that have metadata
    candidates = []
    for d in base_dir.iterdir():
        if d.is_dir() and (d / "tinker_metadata.json").exists():
            candidates.append(d)
    
    # Sort by metadata mtime (descending)
    candidates.sort(key=lambda d: (d / "tinker_metadata.json").stat().st_mtime, reverse=True)
    
    total = len(candidates)
    selected = candidates[offset : offset + limit]
    
    result = []
    for d in selected:
        info = load_train_info(d.name)
        if info:
            # Hide internal checkpoints list for list view if too large, but specs didn't restrict it
            result.append(info)
            
    return result, total

def delete_checkpoint_file(model_id: str, checkpoint_id: str) -> bool:
    if ".." in checkpoint_id:
        return False
    
    model_dir = get_model_dir(model_id)
    ckpt_full_path = model_dir / checkpoint_id
    
    # Remove files
    if ckpt_full_path.exists():
        if ckpt_full_path.is_dir():
            shutil.rmtree(ckpt_full_path)
        else:
            ckpt_full_path.unlink()
    
    # Update metadata
    info = load_train_info(model_id)
    if info:
        checkpoints = info.get("checkpoints", [])
        new_ckpts = [c for c in checkpoints if c['checkpoint_id'] != checkpoint_id]
        info['checkpoints'] = new_ckpts
        
        # If we deleted the "last_checkpoint", reset it
        if info.get('last_checkpoint') and info['last_checkpoint'].get('checkpoint_id') == checkpoint_id:
            info['last_checkpoint'] = new_ckpts[-1] if new_ckpts else None
            
        save_train_info(model_id, info)
        return True
    return False