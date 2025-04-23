import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads book data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the book data.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        Exception: For other potential loading errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    try:
        df = pd.read_csv(file_path, index_col=0)
        print(f"Data loaded successfully from {file_path}")
        print(f"Dataset shape: {df.shape}")
        # Display first few rows and info to verify
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nDataFrame Info:")
        df.info()
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise

# Example usage (optional, for testing purposes)
if __name__ == '__main__':
    # Useing dummy file
    if not os.path.exists('../data'):
        os.makedirs('../data')
    dummy_file = '../data/short_goodreads_data.csv'
    if not os.path.exists(dummy_file):
        dummy_data = {
            'Unnamed: 0': [0, 1],
            'Book': ['Book A', 'Book B'],
            'Author': ['Author X', 'Author Y'],
            'Description': ['Description for A.', 'Description for B.'],
            'Genres': ["['Fiction', 'Sci-Fi']", "['Fiction', 'Mystery']"],
            'Avg_Rating': [4.5, 4.0],
            'Num_Ratings': [100, 200],
            'URL': ['url_a', 'url_b']
        }
        pd.DataFrame(dummy_data).to_csv(dummy_file, index=False)
        print(f"Created dummy file: {dummy_file}")


    try:
        data_path = '../data/goodreads_data.csv'
        if not os.path.exists(data_path):
             print(f"Warning: {data_path} not found, using dummy data for testing.")
             data_path = dummy_file

        dataframe = load_data(data_path)
        print("\nLoader module executed successfully (for testing).")
        print(dataframe.head())

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An error occurred during testing: {e}")