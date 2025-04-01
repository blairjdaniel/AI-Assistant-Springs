import re
import yaml
import string
import json

def load_yaml_config(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def clean_text(text: str, lower_case: bool = True, remove_punctuation: bool = True) -> str:
    if lower_case:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

def count_words(text: str) -> int:
    return len(text.split())

def extract_emails(text: str) -> list:
    pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    return re.findall(pattern, text)

def format_phone_number(phone: str) -> str:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone

def format_phone_numbers_in_text(text: str) -> str:
    """
    Searches the text for phone number patterns and replaces them with a formatted version.
    The regex pattern below matches numbers like 604-557-6168, 6045576168, or 604.557.6168.
    Adjust the pattern as necessary.
    """
    phone_pattern = re.compile(r'\b(\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4})\b')
    matches = phone_pattern.findall(text)
    for match in matches:
        formatted = format_phone_number(match)
        text = text.replace(match, formatted)
    return text

def clean_text_for_bert(text: str, cleaning_conf: dict) -> str:
    """
    Cleans text for training BERT based on cleaning configuration.
    
    Parameters:
        text (str): Input text to clean.
        cleaning_conf (dict): Cleaning configuration (e.g., lower_case, remove_punctuation).
        
    Returns:
        str: Cleaned text.
    """
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Remove specific unwanted characters/substrings
    text = text.replace("**", "")
    text = text.replace("#", "")
    text = text.replace("-", "")
    
    # Apply lower-casing if enabled
    if cleaning_conf.get("lower_case", False):
        text = text.lower()
    
    # Remove punctuation if enabled
    if cleaning_conf.get("remove_punctuation", False):
        text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Format phone numbers found in the text
    text = format_phone_numbers_in_text(text)
    
    # Optionally, add any additional cleaning (like removing stopwords) here.
    
    return text

# Sample usage for testing (can be removed or placed under '__main__')
if __name__ == "__main__":
    sample_text = "Hello, World! This is a sample text. Contact us at example@test.com. Call us at 604-557-6168."
    cleaning_conf = {
        "lower_case": True,
        "remove_punctuation": False  # Set False so demo phone number isn't stripped before formatting
    }
    print("Clean Text for BERT:", clean_text_for_bert(sample_text, cleaning_conf))


def convert_insta_txt_to_jsonl(input_path: str, output_path: str):
    """
    Reads a file with one JSON object per line (insta.txt) and writes a new JSON Lines file.
    
    Each line should be a valid JSON string which will be re-serialized into output_path.
    """
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                # Try to parse each line as JSON
                data = json.loads(line)
                # Write the JSON object as a single line in the output file
                outfile.write(json.dumps(data) + "\n")
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)

# Sample usage for testing (can be removed or placed under '__main__')
if __name__ == "__main__":
    # Test cleaning function
    sample_text = "Hello, World! This is a sample text. Contact us at example@test.com. Call us at 604-557-6168."
    cleaning_conf = {
        "lower_case": True,
        "remove_punctuation": False  # Set False so demo phone number isn't stripped before formatting
    }
    print("Clean Text for BERT:", clean_text_for_bert(sample_text, cleaning_conf))
    
    # Test conversion of Instagram posts
    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/cache/insta/insta.txt"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/insta.jsonl"
    convert_insta_txt_to_jsonl(input_path, output_path)