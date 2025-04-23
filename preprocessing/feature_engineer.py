import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ast # To safely evaluate string representations of lists
import warnings

def process_genres(df: pd.DataFrame, genres_col: str = 'Genres') -> pd.DataFrame:
    """
    Processes the 'Genres' column. It expects genres to be in a string
    representation of a list (e.g., "['Fiction', 'Sci-Fi']").
    It converts this string into an actual list of genres.

    Args:
        df (pd.DataFrame): The input DataFrame.
        genres_col (str): The name of the column containing genres.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'Genres' column (contains lists).
    """
    if genres_col not in df.columns:
        warnings.warn(f"Column '{genres_col}' not found. Skipping genre processing.")
        return df

    def parse_genre_list(genres_str):
        try:
            if pd.isna(genres_str) or not isinstance(genres_str, str):
                return [] 
            genres = ast.literal_eval(genres_str)
            if isinstance(genres, list) and all(isinstance(g, str) for g in genres):
                return genres
            else:
                return [] # Return empty list if structure is not list of strings
        except (ValueError, SyntaxError, TypeError):
            print(f"Could not parse genres: {genres_str}") # Debug print
            return [] # Return empty list on parsing error

    df[genres_col] = df[genres_col].apply(parse_genre_list)
    print(f"Processed '{genres_col}' column.")
    return df

def normalize_numerical_features(df: pd.DataFrame, cols_to_normalize: list = ['Avg_Rating', 'Num_Ratings']) -> pd.DataFrame:
    """
    Normalizes specified numerical columns using Min-Max scaling.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols_to_normalize (list): A list of column names to normalize.

    Returns:
        pd.DataFrame: The DataFrame with normalized columns.
    """
    scaler = MinMaxScaler()
    present_cols = [col for col in cols_to_normalize if col in df.columns]

    if not present_cols:
        warnings.warn("No columns specified for normalization were found in the DataFrame.")
        return df

    for col in present_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
             warnings.warn(f"Column '{col}' contains NaN values. MinMaxScaler might raise errors or produce NaNs.")

    try:
        normalized_col_names = [f"{col}_normalized" for col in present_cols]
        df[normalized_col_names] = scaler.fit_transform(df[present_cols])
        print(f"Normalized columns: {present_cols} -> {normalized_col_names}")
    except ValueError as e:
        print(f"Error during normalization, possibly due to NaN values or non-numeric data in columns: {present_cols}. Error: {e}")
        for col in normalized_col_names:
             if col in df.columns:
                 df.drop(columns=[col], inplace=True)

    return df

# Example usage (optional, for testing purposes)
if __name__ == '__main__':
    data = {
        'Book': ['Book A', 'Book B', 'Book C', 'Book D'],
        'Author': ['Author X', 'Author Y', 'Author Z', 'Author W'],
        'Description': ['Desc A', 'Desc B', 'Desc C', 'Desc D'],
        'Genres': ["['Fiction', 'Sci-Fi']", "['Fiction', 'Mystery']", "['Non-Fiction']", "Invalid Genre String"],
        'Avg_Rating': [4.5, 4.0, 3.5, None],
        'Num_Ratings': [100, 200, 50, 150],
        'URL': ['url_a', 'url_b', 'url_c', 'url_d']
    }
    sample_df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(sample_df)

    processed_df = process_genres(sample_df.copy())
    print("\nDataFrame after processing genres:")
    print(processed_df[['Book', 'Genres']])
    print(processed_df['Genres'].iloc[0]) # Check type of first element
    print(processed_df['Genres'].iloc[3]) # Check type of invalid element

    # Normalize numerical features
    normalized_df = normalize_numerical_features(processed_df.copy()) # Use copy
    print("\nDataFrame after normalizing numerical features:")
    print(normalized_df[['Book', 'Avg_Rating', 'Num_Ratings', 'Avg_Rating_normalized', 'Num_Ratings_normalized']])

    print("\nFeature engineer module executed successfully (for testing).")