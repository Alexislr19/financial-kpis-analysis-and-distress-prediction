import os
from typing import List, Tuple
from pandas import DataFrame, Series

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. Load RAW datasets (from Notebook 01)
# ============================================================

def load_compustat_raw(path: str) -> pd.DataFrame:
    """
    Load raw Compustat CSV (all columns read as strings).
    Equivalent to load_compustat_data from notebook 01.
    """
    return pd.read_csv(path, dtype=str)


def load_macro_raw(path: str) -> pd.DataFrame:
    """
    Load raw macro CSV using ';' separator, NA values = 'NA'.
    Equivalent to load_macro_data from notebook 01.
    """
    return pd.read_csv(path, sep=";", dtype=str, na_values=["NA"])


# ============================================================
# 2. Preprocessing Compustat RAW → cleaned numeric structure
# ============================================================

def preprocess_compustat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep useful columns only, convert strings to numeric where appropriate,
    convert date, cast integers into pandas nullable Int64.
    Equivalent to preprocess_compustat from notebook 01.
    """
    useful: List[str] = [
        "gvkey", "datadate", "fyear", "conm", "tic",
        "sic", "fyr",
        "act", "lct", "at", "lt",
        "seq", "teq", "ceq",
        "dlc", "dltt",
        "revt", "ebit", "xint", "oancf",
        "dlrsn",
    ]

    # Keep only columns present in dataset
    df = df[[c for c in useful if c in df.columns]]

    # Convert datadate to datetime
    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")

    # Convert integer-like columns to Int64
    for c in ["fyear", "fyr", "dlrsn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Convert other numeric fields
    num_cols = [
        "act", "lct", "at", "lt", "seq", "teq", "ceq",
        "dlc", "dltt", "revt", "ebit", "xint", "oancf",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ============================================================
# 3. Compute 5 KPIs
# ============================================================

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5 KPIs: ROA, Total Debt/Equity, Current Ratio,
    CFO Margin, Asset Turnover.
    Equivalent to compute_kpis from notebook 01.
    """
    d = df.copy()

    # Total debt
    total_debt = (
        d["dltt"] + d["dlc"]
        if {"dltt", "dlc"}.issubset(d.columns)
        else None
    )

    # Earnings before tax
    ebt = (
        d["ebit"] - d["xint"]
        if {"ebit", "xint"}.issubset(d.columns)
        else None
    )

    # Find an available equity measure
    equity = None
    for eq in ["seq", "ceq", "teq"]:
        if eq in d.columns:
            equity = d[eq]
            break

    # ROA
    if ebt is not None and "at" in d.columns:
        d["roa"] = ebt / d["at"]

    # Total Debt to Equity
    if total_debt is not None and equity is not None:
        d["total_debt_to_equity"] = total_debt / equity

    # Current Ratio
    if {"act", "lct"}.issubset(d.columns):
        d["current_ratio"] = d["act"] / d["lct"]

    # CFO Margin
    if {"oancf", "revt"}.issubset(d.columns):
        d["cfo_margin"] = d["oancf"] / d["revt"]

    # Asset Turnover
    if {"revt", "at"}.issubset(d.columns):
        d["asset_turnover"] = d["revt"] / d["at"]

    return d


def load_processed_kpis(path: str | None = None) -> DataFrame:
    """
    Load processed KPI dataset (cross-platform).
    Default:
        <project_root>/data/processed/compustat_kpis.csv
    """
    default_path = PROJECT_ROOT / "data" / "processed" / "compustat_kpis.csv"
    file_path = Path(path) if path else default_path

    if not file_path.exists():
        raise FileNotFoundError(f"Processed KPIs file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ============================================================
# 4. Complete pipeline RAW → processed files (Notebook 01 logic)
# ============================================================

def prepare_raw_datasets(
    base_dir: str = "/files/financial-kpis-analysis-and-distress-prediction/",
    comp_in: str = "data/raw/compustat_data.csv",
    comp_out: str = "data/processed/compustat_kpis.csv",
    macro_in: str = "data/raw/macro_data.csv",
    macro_out: str = "data/processed/macro_data.csv",
) -> Tuple[str, str]:
    """
    Replicates the code of the notebook's `if __name__ == "__main__"` block:
      - load raw Compustat
      - preprocess + compute KPIs → export compustat_kpis.csv
      - load raw Macro
      - drop 2025 and drop columns with ≥ 20% missing → export macro_data.csv

    Returns the full paths of the processed files.
    """

    base = base_dir.rstrip("/") + "/"

    comp_in_path = os.path.join(base, comp_in)
    comp_out_path = os.path.join(base, comp_out)
    macro_in_path = os.path.join(base, macro_in)
    macro_out_path = os.path.join(base, macro_out)

    os.makedirs(os.path.dirname(comp_out_path), exist_ok=True)

    # ---------- Compustat ----------
    df_raw = load_compustat_raw(comp_in_path)
    df_prep = preprocess_compustat(df_raw)
    df_kpis = compute_kpis(df_prep)
    df_kpis.to_csv(comp_out_path, index=False)
    print(f"Saved Compustat KPIs → {comp_out_path}")

    # ---------- Macro ----------
    df_macro = load_macro_raw(macro_in_path)

    # Drop year 2025 if present in column "Name"
    if "Name" in df_macro.columns:
        df_macro = df_macro[df_macro["Name"] != "2025"]

    # Keep only columns with < 20% missing
    missing_ratio = df_macro.isna().mean()
    cols_to_keep = missing_ratio[missing_ratio < 0.20].index.tolist()
    df_macro = df_macro[cols_to_keep]

    df_macro.to_csv(macro_out_path, index=False)
    print(f"Saved cleaned macro data → {macro_out_path}")

    return comp_out_path, macro_out_path



def load_processed_macro(path: str | None = None) -> DataFrame:
    """
    Load processed macro dataset (cross-platform).
    Default:
        <project_root>/data/processed/macro_data.csv
    """
    default_path = PROJECT_ROOT / "data" / "processed" / "macro_data.csv"
    file_path = Path(path) if path else default_path

    if not file_path.exists():
        raise FileNotFoundError(f"Processed macro file not found: {file_path}")

    return pd.read_csv(file_path)



def build_balanced_panel_with_deltas(
    kpis_path: str = "../data/processed/compustat_kpis.csv",
    macro_path: str = "../data/processed/macro_data.csv",
    out_panel_path: str = "../data/final/panel_balanced_with_deltas.csv",
    out_deltas_path: str = "../data/final/panel_only_deltas.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a balanced firm × year panel with:
      - firm-level KPIs and their year-on-year deltas
      - macro variables and their year-on-year deltas
      - a binary target y indicating bankrupt vs healthy firms
    Then save:
      - the full panel to out_panel_path
      - a delta-only version (only delta KPIs + delta macro) to out_deltas_path

    Returns:
        df_panel (DataFrame): full panel with KPIs, macro and target y
        df_deltas (DataFrame): panel with only gvkey, fyear and all delta_* columns
    """

    # ================================
    # 1. Load data
    # ================================
    df = pd.read_csv(kpis_path)
    df_macro = pd.read_csv(macro_path)

    print("Compustat KPIs shape:", df.shape)
    print("Macro data shape     :", df_macro.shape)

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # ================================
    # 2. Filter on dlrsn and basic typing
    # ================================
    df["dlrsn"] = pd.to_numeric(df["dlrsn"], errors="coerce")
    df["gvkey"] = df["gvkey"].astype(str)
    df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce").astype("Int64")

    # Keep only firms whose dlrsn ∈ {NaN, 2, 3} over their entire history
    def firm_dlrsn_valid(s: pd.Series) -> bool:
        return s.isna().all() or s.dropna().isin([2, 3]).all()

    firm_valid = df.groupby("gvkey")["dlrsn"].apply(firm_dlrsn_valid)
    valid_firms = firm_valid[firm_valid].index

    df = df[df["gvkey"].isin(valid_firms)].copy()
    print("\nAfter firm-level dlrsn filter (NaN/2/3 only):", df.shape)

    # ================================
    # 3. Keep firms with no missing KPI
    # ================================
    kpi_cols = [
        "roa",
        "total_debt_to_equity",
        "current_ratio",
        "cfo_margin",
        "asset_turnover",
    ]
    kpi_cols = [c for c in kpi_cols if c in df.columns]

    df[kpi_cols] = df[kpi_cols].replace([np.inf, -np.inf], np.nan)

    firm_has_missing = df.groupby("gvkey")[kpi_cols].apply(
        lambda g: g.isna().any().any()
    )
    firms_no_missing = firm_has_missing[~firm_has_missing].index

    print("\nFirms before KPI-missing filter:", df["gvkey"].nunique())
    print("Firms with full KPIs           :", len(firms_no_missing))

    df = df[df["gvkey"].isin(firms_no_missing)].copy()
    df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    print("Shape after removing firms with missing KPIs:", df.shape)

    # ================================
    # 4. Identify bankrupt vs healthy firms
    #    - bankrupt: all dlrsn ∈ {2,3}
    #    - healthy : all dlrsn NaN
    # ================================
    g = df.groupby("gvkey")["dlrsn"]

    is_bankrupt = g.apply(lambda s: s.notna().all() and s.isin([2, 3]).all())
    is_healthy = g.apply(lambda s: s.isna().all())

    bankrupt_firms = is_bankrupt[is_bankrupt].index.values
    healthy_firms = is_healthy[is_healthy].index.values

    print("\nFirm types BEFORE balancing:")
    print("  Bankrupt (all years 2/3):", len(bankrupt_firms))
    print("  Healthy  (all years NaN):", len(healthy_firms))

    if len(bankrupt_firms) == 0 or len(healthy_firms) == 0:
        raise ValueError("Not enough bankrupt or healthy firms to balance the sample.")

    # ================================
    # 5. Balance sample (same number of firms)
    # ================================
    np.random.seed(42)
    n_sample = min(len(bankrupt_firms), len(healthy_firms))

    sampled_bankrupt = np.random.choice(bankrupt_firms, size=n_sample, replace=False)
    sampled_healthy = np.random.choice(healthy_firms, size=n_sample, replace=False)

    selected_firms = np.concatenate([sampled_bankrupt, sampled_healthy])

    df_bal = df[df["gvkey"].isin(selected_firms)].copy()
    df_bal = df_bal.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    print("\nFirm types AFTER balancing:")
    print("  Bankrupt sampled:", len(sampled_bankrupt))
    print("  Healthy sampled :", len(sampled_healthy))
    print("Balanced firm-level shape:", df_bal.shape)

    # ================================
    # 6. KPI deltas by firm
    # ================================
    delta_kpi_cols = []
    for col in kpi_cols:
        dcol = f"delta_{col}"
        df_bal[dcol] = df_bal.groupby("gvkey")[col].diff()
        delta_kpi_cols.append(dcol)

    print("\nDelta KPI columns created:", delta_kpi_cols)

    before_rows = df_bal.shape[0]
    df_bal = df_bal.dropna(subset=delta_kpi_cols).copy()
    after_rows = df_bal.shape[0]

    print("Shape after dropping first-year rows (no KPI deltas):")
    print("  Before:", before_rows)
    print("  After :", after_rows)

    # ================================
    # 7. Prepare macro data + deltas
    # ================================
    df_macro.columns = (
        df_macro.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    if "name" not in df_macro.columns:
        raise ValueError("Macro data must contain a 'Name' column with years (now 'name').")

    df_macro["fyear"] = pd.to_numeric(df_macro["name"], errors="coerce")
    df_macro = df_macro.dropna(subset=["fyear"]).copy()
    df_macro["fyear"] = df_macro["fyear"].astype(int)
    df_macro = df_macro.drop(columns=["name"])

    macro_cols = [c for c in df_macro.columns if c != "fyear"]

    # Clean numeric formatting (e.g., "1 234,56")
    for col in macro_cols:
        df_macro[col] = (
            df_macro[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df_macro[col] = pd.to_numeric(df_macro[col], errors="coerce")

    # Median imputation
    for col in macro_cols:
        df_macro[col] = df_macro[col].fillna(df_macro[col].median())

    df_macro = df_macro.sort_values("fyear").reset_index(drop=True)

    # Macro deltas
    delta_macro_cols = []
    for col in macro_cols:
        dcol = f"delta_{col}"
        df_macro[dcol] = df_macro[col].diff()
        delta_macro_cols.append(dcol)

    print("\nDelta macro columns created:", delta_macro_cols)

    before_rows_macro = df_macro.shape[0]
    df_macro = df_macro.dropna(subset=delta_macro_cols).copy()
    after_rows_macro = df_macro.shape[0]

    print("Macro shape after dropping first year (no deltas):")
    print("  Before:", before_rows_macro)
    print("  After :", after_rows_macro)

    # ================================
    # 8. Merge firm + macro on fyear
    # ================================
    df_bal["fyear"] = df_bal["fyear"].astype(int)
    df_panel = df_bal.merge(df_macro, on="fyear", how="left")

    print("\nPanel shape after merge firm × macro:", df_panel.shape)

    # ================================
    # 9. Create target y (1 = bankrupt)
    # ================================
    bankrupt_set = set(sampled_bankrupt)
    df_panel["y"] = df_panel["gvkey"].isin(bankrupt_set).astype(int)

    print("\nTarget distribution (y):")
    print(df_panel["y"].value_counts())

    # ================================
    # 10. Save final panel
    # ================================
    os.makedirs(os.path.dirname(out_panel_path), exist_ok=True)
    df_panel.to_csv(out_panel_path, index=False)

    print("\nFinal panel saved to:", out_panel_path)
    print("Final shape:", df_panel.shape)
    print("\nPreview:")
    print(df_panel.head())

    # ================================
    # 11. Create file containing ONLY delta KPIs + delta MACRO
    # ================================
    delta_cols_only = [c for c in df_panel.columns if c.startswith("delta_")]

    df_deltas = df_panel[["gvkey", "fyear"] + delta_cols_only].copy()

    os.makedirs(os.path.dirname(out_deltas_path), exist_ok=True)
    df_deltas.to_csv(out_deltas_path, index=False)

    print("\nDelta-only file saved to:", out_deltas_path)
    print("Shape:", df_deltas.shape)
    print(df_deltas.head())

    return df_panel, df_deltas


def load_final_panel(path: str | None = None) -> DataFrame:
    """
    Load the final balanced firm-year panel (cross-platform).

    Default location:
        <project_root>/data/final/panel_balanced_with_deltas.csv
    """
    default_path = PROJECT_ROOT / "data" / "final" / "panel_balanced_with_deltas.csv"
    file_path = Path(path) if path else default_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Final panel file not found:\n  {file_path}\n\n"
            "Expected location:\n"
            f"  {default_path}\n\n"
            "Fix:\n"
            "- Ensure the file exists in data/final/\n"
            "- Or pass an explicit path to load_final_panel(path=...)"
        )

    return pd.read_csv(file_path)


def build_X_y_from_panel(df: DataFrame) -> Tuple[DataFrame, Series, List[str]]:
    """
    Build the feature matrix X and target vector y from the final panel.

    Uses:
      - KPIs
      - KPI deltas
      - macro variables
      - macro deltas
    """

    # Target
    y = df["y"].astype(int)

    # Feature columns explicitly used in notebook 04
    kpi_cols = [
        "roa",
        "total_debt_to_equity",
        "current_ratio",
        "cfo_margin",
        "asset_turnover",
    ]

    delta_kpi_cols = [
        "delta_roa",
        "delta_total_debt_to_equity",
        "delta_current_ratio",
        "delta_cfo_margin",
        "delta_asset_turnover",
    ]

    macro_cols = [
        "us_gdp_(ar)_cura",
        "us_cpi___all_urban:_all_items_sadj",
        "us_treasury_bill_rate___3_month_(ep)_nadj",
        "us_treasury_yield_adjusted_to_constant_maturity___20_year_nadj",
        "us_unemployment_rate_sadj",
    ]

    delta_macro_cols = [
        "delta_us_gdp_(ar)_cura",
        "delta_us_cpi___all_urban:_all_items_sadj",
        "delta_us_treasury_bill_rate___3_month_(ep)_nadj",
        "delta_us_treasury_yield_adjusted_to_constant_maturity___20_year_nadj",
        "delta_us_unemployment_rate_sadj",
    ]

    feature_cols = kpi_cols + delta_kpi_cols + macro_cols + delta_macro_cols
    X = df[feature_cols].copy()

    print("Feature matrix X shape:", X.shape)
    print("Any NaN in X? ->", X.isna().any().any())

    return X, y, feature_cols


def standard_train_test_split(
    X: DataFrame,
    y: Series,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
):
    """
    Standard sklearn train_test_split wrapper.
    If stratify=True, uses y to preserve class balance in train/test.
    """
    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    print("\nTrain size:", X_train.shape[0])
    print("Test size :", X_test.shape[0])
    print("\nClass distribution in y_train:")
    print(y_train.value_counts())
    print("\nClass distribution in y_test:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def impute_train_test_median(
    X_train: DataFrame,
    X_test: DataFrame,
):
    """
    Apply median imputation (sklearn SimpleImputer) on train and test sets.
    Returns imputed arrays and the fitted imputer.
    """
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    return X_train_imp, X_test_imp, imputer

