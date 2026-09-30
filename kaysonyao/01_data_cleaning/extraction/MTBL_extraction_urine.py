import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path


def natural_compound_order(columns):
    """
    Natural-sort compound IDs:
    c1, c2, p1, p2, ..., p10, p11
    """
    def sort_key(value):
        value = str(value).strip()
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", value)

        if match:
            prefix, number = match.groups()
            return (prefix.lower(), int(number))

        return (value.lower(), float("inf"))

    return sorted(columns, key=sort_key)


def get_sample_start(df):
    """
    Find the position of the Sample ID column, regardless of case,
    surrounding whitespace, or non-breaking spaces.
    """
    normalized = (
        pd.Index(df.columns)
        .fillna("")
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .str.lower()
    )

    matches = np.flatnonzero(normalized == "sample id")

    if len(matches) == 0:
        raise ValueError(
            "Could not find a Sample ID column. "
            f"Columns found: {df.columns.tolist()}"
        )

    return matches[0]


def start_at_sample_id(df):
    """
    Locate the first row containing 'Sample ID', make that row the header,
    and retain all rows below it.

    This creates the ordinary biological-sample intermediate dataframe.
    The original raw worksheet remains available separately for extraction
    of BatchNPoolN control values.
    """
    cleaned = (
        df.fillna("")
        .astype(str)
        .apply(
            lambda col: (
                col.str.replace("\xa0", " ", regex=False)
                .str.strip()
                .str.lower()
            )
        )
    )

    row_mask = cleaned.eq("sample id").any(axis=1)

    if not row_mask.any():
        raise ValueError("Could not find Sample ID anywhere in this sheet.")

    header_position = np.flatnonzero(row_mask.to_numpy())[0]
    header_row = df.iloc[header_position].copy()

    print(f"Found first Sample ID marker on Excel row {header_position + 1}.")

    result = df.iloc[header_position + 1:].copy()
    result.columns = header_row.tolist()

    result.columns = [
        "Sample ID"
        if str(col).replace("\xa0", " ").strip().lower() == "sample id"
        else col
        for col in result.columns
    ]

    return result.reset_index(drop=True)


def get_batch_urine(df):
    """
    Build Sample_ID / batch metadata for ordinary biological samples.

    The batch annotation is expected in dataframe row index 2 after
    start_at_sample_id() has created the intermediate dataframe.
    """
    sample_start = get_sample_start(df)
    batch_row_index = 2

    temp = df.iloc[
        batch_row_index:batch_row_index + 1,
        sample_start:
    ].copy()

    if temp.empty:
        raise ValueError(
            "Could not extract the ordinary-sample batch row. "
            "Check batch_row_index."
        )

    # Ensure the first selected header is consistently named.
    temp.columns = temp.columns.astype(object)
    temp.columns.values[0] = "Sample ID"

    temp = temp.set_index("Sample ID").transpose()
    temp.columns = ["batch"]

    def extract_batch(value):
        match = re.search(r"_(B\d+)_", str(value))
        return match.group(1) if match else pd.NA

    temp["batch"] = temp["batch"].map(extract_batch)
    temp.index.name = "Sample_ID"

    # Remove columns/rows that do not represent actual sample IDs.
    temp = temp.loc[
        temp.index.notna()
        & (temp.index.astype(str).str.strip() != "")
    ]

    return temp.reset_index()


def get_compounds_urine(df):
    """
    Extract urine compound metadata.

    In the clean biological-sample intermediate dataframe:
    - Compound metadata header is dataframe row index 3.
    - Compound records begin immediately below it.
    - The first two blank Export Order values are named c1 and c2.
    """
    sample_start = get_sample_start(df)
    compound_header_row = 3

    temp = df.iloc[compound_header_row:, :sample_start].copy()

    if temp.empty:
        raise ValueError(
            "Compound metadata slice is empty. Check compound_header_row "
            "and the Sample ID position."
        )

    temp.columns = temp.iloc[0].tolist()
    temp = temp.iloc[1:].copy()

    export_order_col = "Export Order"

    if export_order_col not in temp.columns:
        raise ValueError(
            f"Expected '{export_order_col}' in urine compound headers; "
            f"found: {temp.columns.tolist()}"
        )

    blank_rows = (
        temp[export_order_col].isna()
        | temp[export_order_col].astype(str).str.strip().eq("")
    )

    blank_positions = np.flatnonzero(blank_rows.to_numpy())

    if len(blank_positions) < 2:
        raise ValueError(
            "Expected at least two blank Export Order values for urine ISTDs."
        )

    temp.iloc[
        blank_positions[0],
        temp.columns.get_loc(export_order_col)
    ] = "c1"

    temp.iloc[
        blank_positions[1],
        temp.columns.get_loc(export_order_col)
    ] = "c2"

    temp = temp.set_index(export_order_col)
    temp.index.name = "Export Order"

    return temp


def get_expression_urine(df, comp):
    """
    Create the ordinary biological-sample expression matrix.

    Rows: Sample_ID
    Columns: c1, c2, p1, p2, etc.
    """
    sample_start = get_sample_start(df)
    first_compound_row = 4

    # The first column after "Sample ID" is not a biological sample intensity
    # column in this workbook layout, hence sample_start + 1.
    temp = df.iloc[first_compound_row:, sample_start + 1:].copy()

    if len(temp) != len(comp.index):
        raise ValueError(
            f"Expression has {len(temp)} compound rows but compound metadata "
            f"has {len(comp.index)} rows. Check first_compound_row."
        )

    temp.index = comp.index
    temp = temp.transpose()

    temp.index.name = "Sample_ID"
    temp.columns.name = "compound"

    return temp


def get_pos_pool_expression_raw(raw_df, comp):
    """
    Extract POS BatchNPoolN control expression values from the unchanged
    raw POS worksheet.

    Expected POS layout:
    - Pool labels: Excel row 4, columns AF:CA.
    - First biological Sample ID marker: Excel row 4, column CB.
    - First compound intensity row: Excel row 7.

    CumulativePool controls are intentionally excluded.
    """
    pool_header_row = 3
    first_value_row = 6

    first_pool_col = 31  # AF, zero-based position
    last_pool_col = 78   # CA, zero-based position

    pool_labels = raw_df.iloc[
        pool_header_row,
        first_pool_col:last_pool_col + 1
    ].tolist()

    pool_values = raw_df.iloc[
        first_value_row:,
        first_pool_col:last_pool_col + 1
    ].copy()

    # Keep only values such as Batch1Pool1, Batch2Pool10, etc.
    # CumulativePool is excluded.
    keep_positions = [
        i for i, label in enumerate(pool_labels)
        if pd.notna(label)
        and re.fullmatch(r"(?i)Batch\d+Pool\d+", str(label).strip())
    ]

    if not keep_positions:
        raise ValueError(
            "No BatchNPoolN labels were found in POS Excel row 4, columns AF:CA."
        )

    pool_labels = [
        str(pool_labels[i]).strip()
        for i in keep_positions
    ]

    pool_values = pool_values.iloc[:, keep_positions].copy()

    if len(pool_values) != len(comp.index):
        raise ValueError(
            f"POS pool expression has {len(pool_values)} compound rows, "
            f"but compound metadata has {len(comp.index)} rows. "
            "Check first_value_row."
        )

    pool_values.index = comp.index
    pool_expression = pool_values.transpose()

    pool_expression.index = pool_labels
    pool_expression.index.name = "Sample_ID"
    pool_expression.columns.name = "compound"

    return pool_expression


def get_neg_pool_expression_raw(raw_df, comp):
    """
    Extract NEG BatchNPoolN control expression values from the unchanged
    raw NEG worksheet.

    Expected NEG layout:
    - Pool labels: Excel row 4, columns AC:BY.
    - First biological Sample ID marker: Excel row 4, column BZ.
    - First compound intensity row: Excel row 7.

    CumulativePool controls are intentionally excluded.
    """
    pool_header_row = 3
    first_value_row = 6

    first_pool_col = 28  # AC, zero-based position
    last_pool_col = 76   # BY, zero-based position

    pool_labels = raw_df.iloc[
        pool_header_row,
        first_pool_col:last_pool_col + 1
    ].tolist()

    pool_values = raw_df.iloc[
        first_value_row:,
        first_pool_col:last_pool_col + 1
    ].copy()

    # Keep only BatchNPoolN controls; exclude CumulativePool.
    keep_positions = [
        i for i, label in enumerate(pool_labels)
        if pd.notna(label)
        and re.fullmatch(r"(?i)Batch\d+Pool\d+", str(label).strip())
    ]

    if not keep_positions:
        raise ValueError(
            "No BatchNPoolN labels were found in NEG Excel row 4, columns AC:BY."
        )

    pool_labels = [
        str(pool_labels[i]).strip()
        for i in keep_positions
    ]

    pool_values = pool_values.iloc[:, keep_positions].copy()

    if len(pool_values) != len(comp.index):
        raise ValueError(
            f"NEG pool expression has {len(pool_values)} compound rows, "
            f"but compound metadata has {len(comp.index)} rows. "
            "Check first_value_row."
        )

    pool_values.index = comp.index
    pool_expression = pool_values.transpose()

    pool_expression.index = pool_labels
    pool_expression.index.name = "Sample_ID"
    pool_expression.columns.name = "compound"

    return pool_expression


def get_batch_pools_urine(raw_df, polarity):
    """
    Create batch metadata for BatchNPoolN controls from the raw worksheet.

    CumulativePool controls are excluded.
    """
    pool_header_row = 3

    if polarity == "pos":
        first_pool_col = 31  # AF
        last_pool_col = 78   # CA
    elif polarity == "neg":
        first_pool_col = 28  # AC
        last_pool_col = 76   # BY
    else:
        raise ValueError("polarity must be 'pos' or 'neg'.")

    pool_ids = (
        raw_df.iloc[
            pool_header_row,
            first_pool_col:last_pool_col + 1
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    records = []

    for sample_id in pool_ids:
        match = re.fullmatch(r"(?i)Batch(\d+)Pool\d+", sample_id)

        if match:
            records.append(
                {
                    "Sample_ID": sample_id,
                    "batch": f"B{match.group(1)}"
                }
            )

    if not records:
        raise ValueError(
            f"No BatchNPoolN IDs found in the {polarity.upper()} pool block."
        )

    return pd.DataFrame(records)


def generate_files(df, raw_df, file_output, polarity):
    """
    Generate ordinary biological-sample files and BatchNPoolN metadata.

    Keep compound metadata in source-row order while mapping raw intensity
    rows. Export a separately natural-sorted metadata table afterward.
    """
    compounds = get_compounds_urine(df)

    batch_pools = get_batch_pools_urine(raw_df, polarity)
    batch_samples = get_batch_urine(df)

    batch = pd.concat(
        [batch_pools, batch_samples],
        axis=0,
        ignore_index=True
    )

    expression = get_expression_urine(df, compounds)
    compound_order = natural_compound_order(compounds.index)
    expression = expression.reindex(columns=compound_order)
    compounds_export = compounds.reindex(compound_order)

    batch.to_csv(
        f"{file_output}/{polarity}_batch.csv",
        index=False
    )

    compounds_export.to_csv(
        f"{file_output}/{polarity}_compounds.csv",
        index_label="Export Order"
    )

    expression.to_csv(
        f"{file_output}/{polarity}_expression.csv",
        index_label="Sample_ID"
    )

    return compounds, expression


def extract_data(file_input, file_output):
    """
    Extract urine POS and NEG sheets.

    Workflow:
    1. Keep raw sheets for fixed-range BatchNPoolN extraction.
    2. Build clean intermediate biological-sample dataframes beginning at the
       first Sample ID row.
    3. Extract compounds, sample metadata, and biological expressions from
       clean intermediates.
    4. Extract only BatchNPoolN control expressions from raw sheets.
    5. Combine pool controls followed by biological samples.
    """
    raw_pos = pd.read_excel(
        file_input,
        sheet_name="POS Compounds",
        header=None
    ).dropna(how="all")

    raw_neg = pd.read_excel(
        file_input,
        sheet_name="NEG Compounds",
        header=None
    ).dropna(how="all")

    file_pos = start_at_sample_id(raw_pos)
    file_neg = start_at_sample_id(raw_neg)

    pos_compounds, pos_sample_expression = generate_files(
        file_pos,
        raw_pos,
        file_output,
        "pos"
    )

    neg_compounds, neg_sample_expression = generate_files(
        file_neg,
        raw_neg,
        file_output,
        "neg"
    )

    # NEG pool controls: extract from raw data and exclude CumulativePool.
    neg_pool_expression = get_neg_pool_expression_raw(
        raw_neg,
        neg_compounds
    )

    neg_pool_expression = neg_pool_expression.reindex(
        columns=neg_sample_expression.columns
    )

    neg_expression = pd.concat(
        [neg_pool_expression, neg_sample_expression],
        axis=0
    )

    neg_expression = neg_expression.reindex(
        columns=natural_compound_order(neg_expression.columns)
    )

    neg_expression.index.name = "Sample_ID"
    neg_expression.columns.name = "compound"

    # Pool-only matrix is also naturally ordered.
    neg_pool_expression = neg_pool_expression.reindex(
        columns=natural_compound_order(neg_pool_expression.columns)
    )

    neg_pool_expression.to_csv(
        f"{file_output}/neg_batch_pool_expression.csv",
        index_label="Sample_ID"
    )

    neg_expression.to_csv(
        f"{file_output}/neg_expression.csv",
        index_label="Sample_ID"
    )

    # POS pool controls: extract from raw data and exclude CumulativePool.
    pos_pool_expression = get_pos_pool_expression_raw(
        raw_pos,
        pos_compounds
    )

    pos_pool_expression = pos_pool_expression.reindex(
        columns=pos_sample_expression.columns
    )

    pos_expression = pd.concat(
        [pos_pool_expression, pos_sample_expression],
        axis=0
    )

    pos_expression = pos_expression.reindex(
        columns=natural_compound_order(pos_expression.columns)
    )

    pos_expression.index.name = "Sample_ID"
    pos_expression.columns.name = "compound"

    # Pool-only matrix is also naturally ordered.
    pos_pool_expression = pos_pool_expression.reindex(
        columns=natural_compound_order(pos_pool_expression.columns)
    )

    pos_pool_expression.to_csv(
        f"{file_output}/pos_batch_pool_expression.csv",
        index_label="Sample_ID"
    )

    pos_expression.to_csv(
        f"{file_output}/pos_expression.csv",
        index_label="Sample_ID"
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage:\n"
            "python MTBL_extraction.py <input_excel_file> <output_directory>"
        )

    file_input = sys.argv[1]
    file_output = sys.argv[2]

    Path(file_output).mkdir(parents=True, exist_ok=True)

    extract_data(file_input, file_output)


if __name__ == "__main__":
    main()
