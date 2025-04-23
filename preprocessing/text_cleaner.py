import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # Using word_tokenize for better handling

# WARNING!
# Ensure you have downloaded nltk stopwords:
# import nltk
# nltk.download('stopwords')
# Load stop words once
# WARNING!

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Please run: import nltk; nltk.download('stopwords')")
    stop_words = set()

def clean_text(text: str) -> str:
    """
    Cleans a single text string by:
    1. Converting to lowercase.
    2. Removing punctuation.
    3. Removing numbers (optional, kept for now as some might be relevant).
    4. Removing English stop words.
    5. Removing extra whitespace.

    Args:
        text (str): The text string to clean.

    Returns:
        str: The cleaned text string.
    """
    if not isinstance(text, str):
        return "" # Return empty string if input is not a string (e.g., NaN)

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove numbers (using regex) 
    # text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words and len(word) > 1]
    text = " ".join(filtered_words)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Example usage (optional, for testing purposes)
if __name__ == '__main__':
    sample_text = " This is a Sample Description with Punctuation! And numbers 123. Check it out.  "
    cleaned = clean_text(sample_text)
    print(f"Original: '{sample_text}'")
    print(f"Cleaned:  '{cleaned}'")

    sample_nan = float('nan')
    cleaned_nan = clean_text(sample_nan)
    print(f"Original: {sample_nan}")
    print(f"Cleaned: '{cleaned_nan}'")

    print("\nText cleaner module executed successfully (for testing).")