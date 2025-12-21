from pathlib import Path
import pandas as pd

# Load frame-level event data from a validated CSV path
def load_frame_events(csv_path: str | Path) -> pd.DataFrame:
    # Normalize input to Path object for consistent filesystem handling
    csv_path = Path(csv_path)

    # Fail fast if the expected CSV file is missing
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    # Read structured frame events into a DataFrame
    df = pd.read_csv(csv_path)
    return df