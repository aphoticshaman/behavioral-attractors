"""Utility functions for Paper B experiment."""

import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_text(path: str) -> str:
    """Load text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save_jsonl(records: List[Dict], path: str, mode: str = 'a') -> None:
    """Append records to JSONL file."""
    with open(path, mode, encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_jsonl(path: str) -> List[Dict]:
    """Load all records from JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_run_id() -> str:
    """Generate unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return f"run_{timestamp}_{random_suffix}"


def create_manifest(config: Dict, models: List[str], run_id: str) -> Dict:
    """Create experiment manifest."""
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config_snapshot": config,
        "models": models,
        "version": "1.0.0"
    }


def format_telemetry(turn: int, thermal: float, damage: float,
                     mem_err: float, policy_lock: float) -> str:
    """Format telemetry packet string."""
    return f"[telemetry t={turn:03d} thermal={thermal:.2f} damage={damage:.2f} mem_err={mem_err:.2f} policy_lock={policy_lock:.2f}]"


def parse_telemetry(text: str) -> Optional[Dict[str, float]]:
    """Parse telemetry packet from text. Returns None if not found."""
    import re
    pattern = r'\[telemetry t=(\d+) thermal=([\d.]+) damage=([\d.]+) mem_err=([\d.]+) policy_lock=([\d.]+)\]'
    match = re.search(pattern, text)
    if match:
        return {
            "turn": int(match.group(1)),
            "thermal": float(match.group(2)),
            "damage": float(match.group(3)),
            "mem_err": float(match.group(4)),
            "policy_lock": float(match.group(5))
        }
    return None


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


class SeededRNG:
    """Seeded random number generator for reproducibility."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def random(self) -> float:
        return self.rng.random()

    def uniform(self, a: float, b: float) -> float:
        return self.rng.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def choice(self, seq: List) -> Any:
        return self.rng.choice(seq)

    def shuffle(self, seq: List) -> None:
        self.rng.shuffle(seq)

    def sample(self, seq: List, k: int) -> List:
        return self.rng.sample(seq, k)
