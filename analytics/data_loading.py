from pathlib import Path
import pandas as pd

def load_frame_events(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")
    df = pd.read_csv(csv_path)
    return df