from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class SplitConfig:
    test_size: float
    val_size: float
    stratify: bool
    random_state: int

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_data(
    df: pd.DataFrame,
    target: str,
    cfg: SplitConfig,
    id_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:

    if id_column is not None and id_column in df.columns:
        df = df.drop(columns=[id_column])

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    y = df[target]
    X = df.drop(columns=[target])

    strat = y if cfg.stratify else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size,
        random_state=cfg.random_state, stratify=strat
    )

    # val as fraction of remaining data
    val_fraction_of_trainval = cfg.val_size / (1.0 - cfg.test_size)
    strat_tv = y_trainval if cfg.stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=cfg.random_state,
        stratify=strat_tv
    )

    return X_train, X_val, X_test, y_train, y_val, y_test