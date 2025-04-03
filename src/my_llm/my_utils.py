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

def remove_empty_body_entries(input_path: str, output_path: str) -> None:
    """
    Reads a JSON file containing a list of entries and writes a new JSON file 
    with any entry removed where the "body" key is an empty string.
    
    Parameters:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    import json
    with open(input_path, "r") as infile:
        data = json.load(infile)
    
    # Filter entries: remove those where "body" is exactly ""
    filtered_data = [entry for entry in data if entry.get("body", None) != ""]
    
    with open(output_path, "w") as outfile:
        json.dump(filtered_data, outfile, indent=4)

def add_form_field(input_path: str, output_path: str, form_type: str) -> None:
    """
    Reads a JSON file containing a list of entries, adds a "form" key with the provided form_type
    to each entry, and writes the updated list to a new JSON file.
    
    Parameters:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
        form_type (str): The form type to add (e.g., "waitlist", "contact", "phase3_sales").
    """
    import json
    with open(input_path, "r") as infile:
        data = json.load(infile)
    
    # Add the "form" field to each entry
    for entry in data:
        entry["form"] = form_type
    
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)

def add_company_voice(input_path: str, output_path: str, company_voice: str) -> None:
    """
    Reads a JSONL file, adds a "company_voice" key with the given value to each entry,
    and writes the modified entries to a new JSONL file.
    
    Parameters:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to the output JSONL file.
        company_voice (str): The metadata value for the company's voice.
    """
    import json
    with open(input_path, "r") as infile:
        data = json.load(infile)
    
    # Add the "company_voice" field to each entry
    for entry in data:
        entry["company_voice"] = company_voice
    
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)

def load_forms_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def enrich_email_with_contact_response(email_entry: dict, forms_config: dict) -> dict:
    form_type = email_entry.get("form", "").lower()
    baseline = {}
    forms = forms_config.get("forms", {})
    
     # Create a lower-cased key mapping for forms
    lower_forms = { key.lower(): value for key, value in forms.items() }
    
    # Retrieve the corresponding baseline response (if any)
    email_entry["baseline_response"] = lower_forms.get(form_type, lower_forms.get("contact", {}))
    return email_entry



def process_emails(emails_jsonl_path: str, forms_yaml_path: str, output_path: str) -> None:
    forms_config = load_forms_config(forms_yaml_path)
    enriched_entries = []
    with open(emails_jsonl_path, "r") as infile:
        # Each line is a JSON object
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                enriched_entry = enrich_email_with_contact_response(entry, forms_config)
                enriched_entries.append(enriched_entry)
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)
    # Write the enriched emails back into a JSONL file
    with open(output_path, "w") as outfile:
        for entry in enriched_entries:
            outfile.write(json.dumps(entry) + "\n")

import json
import yaml

def load_socials_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def enrich_instagram_post(instagram_entry: dict, socials_config: dict) -> dict:
    # Retrieve the instagram template from the socials config
    instagram_template = socials_config.get("socials", {}).get("instagram", {})
    
    # Get the response template string
    response_template = instagram_template.get("response_template", "")
    
    # Format the template if needed by inserting the post content.
    # For instance, if the response_template uses the placeholder {post_content}, fill it with the actual text.
    if "{post_content}" in response_template:
        formatted_response = response_template.format(post_content=instagram_entry.get("text", ""))
    else:
        formatted_response = response_template
        
    # Save the formatted response into the entry under baseline_response
    instagram_entry["baseline_response"] = formatted_response

    # Optionally, if you want to include additional data such as guidelines or examples, you can add those too.
    instagram_entry["social_guidelines"] = instagram_template.get("guidelines", [])
    instagram_entry["social_examples"] = instagram_template.get("examples", [])
    
    return instagram_entry

def process_instagram_posts(instagram_jsonl_path: str, socials_yaml_path: str, output_path: str) -> None:
    socials_config = load_socials_config(socials_yaml_path)
    enriched_entries = []
    with open(instagram_jsonl_path, "r") as infile:
        # Each line is a JSON object
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Ensure the entry is for instagram by checking its type (or other key)
                if entry.get("type", "").lower() == "instagram":
                    enriched_entry = enrich_instagram_post(entry, socials_config)
                else:
                    enriched_entry = entry
                enriched_entries.append(enriched_entry)
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)
    
    # Write the enriched entries back into a JSONL file
    with open(output_path, "w") as outfile:
        for entry in enriched_entries:
            outfile.write(json.dumps(entry) + "\n")

def enrich_website_content(website_entry: dict, socials_config: dict) -> dict:
    # Retrieve the website template from the socials config
    website_template = socials_config.get("socials", {}).get("website", {})
    
    # The website template's response_template expects a placeholder {homepage_content}
    response_template = website_template.get("response_template", "")
    
    # Format the response by inserting the website entry's content into the template
    formatted_response = response_template.format(
        homepage_content=website_entry.get("content", "")
    )
    
    # Add the baseline response (you can also include subject if needed)
    website_entry["baseline_response"] = {
        "subject": website_template.get("label", "Website Content"),
        "body": formatted_response
    }
    
    # Optionally, include guidelines and examples from the configuration
    website_entry["guidelines"] = website_template.get("guidelines", [])
    website_entry["examples"] = website_template.get("examples", [])
    
    return website_entry

def process_website_posts(website_jsonl_path: str, socials_yaml_path: str, output_path: str) -> None:
    socials_config = load_socials_config(socials_yaml_path)
    enriched_entries = []
    with open(website_jsonl_path, "r") as infile:
        # Each line is a JSON object
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Check that this entry is marked as website content
                if entry.get("type", "").lower() == "website":
                    enriched_entry = enrich_website_content(entry, socials_config)
                else:
                    enriched_entry = entry
                enriched_entries.append(enriched_entry)
            except Exception as e:
                print("Error processing line:")
                print(line)
                print(e)
                
    # Write the enriched entries back into a JSONL file
    with open(output_path, "w") as outfile:
        for entry in enriched_entries:
            outfile.write(json.dumps(entry) + "\n")


import re
import json

def parse_brandbook(text):
    # Use regex to capture each numbered section.
    # This regex expects a number followed by a period and a space.
    pattern = r"\s*(\d+)\.\s+"
    # Find all split positions
    splits = re.split(pattern, text.strip())
    
    # The first element might be the header (if present), so we ignore if it doesn't look like a number.
    # The splits result will be like: [header?, num, content, num, content, ...]
    entries = {}
    if len(splits) >= 3:
        # If the first element is not a number, ignore it and process subsequent number-content pairs.
        i = 1 if not splits[0].strip().isdigit() else 0  
        # Adjust starting index: if splits[0] is header, then i = 1
        if i == 1:
            header = splits[0].strip()
            entries["header"] = header
        # Process the remaining parts in pairs: number then content.
        for j in range(i, len(splits) - 1, 2):
            number = splits[j].strip()
            content = splits[j+1].strip()
            entries[number] = content
    
    return entries

import json

def convert_conversation_to_example(conversation):
    # For each conversation, we will use the first user message as prompt
    # and the assistant’s first response as completion
    messages = conversation.get("messages", [])
    prompt = ""
    completion = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += msg["text"].strip() + "\n"
        elif msg["role"] == "assistant" and not completion:
            completion = msg["text"].strip()
    # Optionally, add a stop sequence (ensure newlines or a token is defined for your training)
    if not completion.endswith("\n"):
        completion += "\n"
    return {"prompt": prompt, "completion": completion}
   
    print("Training file created at", output_path)

import glob
import json
import os

def load_all_jsonl_files(directory: str) -> list:
    """
    Loads and combines data from all JSONL files in the specified directory.
    
    Parameters:
        directory (str): Path to the directory containing JSONL files.
    
    Returns:
        A list with the combined data entries from all processed JSONL files.
    """
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    all_data = []
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
    return all_data

import pandas as pd
def load_normalized_jsonl(file_path: str) -> pd.DataFrame:
    """
    Loads a JSONL file into a Pandas DataFrame and normalizes nested fields.
    
    Parameters:
        file_path (str): Path to the JSONL file.
    
    Returns:
        pd.DataFrame: A normalized DataFrame containing the data.
    """
    
    # Read the JSONL file using Pandas
    df = pd.read_json(file_path, lines=True)
    # Normalize the DataFrame entries
    normalized_df = pd.json_normalize(pd.DataFrame(df).to_dict(orient='records'))
    return normalized_df

def load_txt_files_to_df(directory: str) -> pd.DataFrame:
    """
    Loads all .txt files in a directory into a pandas DataFrame.
    
    Parameters:
        directory (str): Path to the directory containing .txt files.
        
    Returns:
        pd.DataFrame: A DataFrame with columns "filename" and "content".
    """
    data = []
    # List all files in the directory
    for fname in os.listdir(directory):
        if fname.lower().endswith(".txt"):
            file_path = os.path.join(directory, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                data.append({"filename": fname, "content": content})
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    return pd.DataFrame(data)

from sklearn.model_selection import train_test_split
import pandas as pd

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Splits a DataFrame into train and test DataFrames.

    Parameters:
        df (pd.DataFrame): DataFrame to split.
        test_size (float): Fraction of the data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

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

    input_file = "/path/to/your/input.json"
    output_file = "/path/to/your/output.json"
    remove_empty_body_entries(input_file, output_file)
    print(f"Filtered output written to {output_file}")

    # Example usage: add "waitlist" as the form type to all entries
    input_file = "/path/to/your/input.json"
    output_file = "/path/to/your/output.json"
    form_type = "waitlist"  # Change as needed, e.g. "contact", "phase3_sales", etc.
    add_form_field(input_file, output_file, form_type)
    print(f'Updated entries with form="{form_type}" written to {output_file}')

    input_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/website.jsonl"
    output_file = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/website_company_voice.jsonl"
    voice = "Friendly, professional, and warm - reflecting Springs RV Resort's brand identity."
    add_company_voice(input_file, output_file, voice)
    print(f"Updated file with company_voice metadata written to {output_file}")

    emails_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/pre-output/emails_contact.jsonl"
    forms_yaml = "/Users/blairjdaniel/AI-Assistant-Springs/config/forms.yaml"  # Your YAML file with baseline responses
    output_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/output/emails_contact_enriched.jsonl"
    process_emails(emails_jsonl, forms_yaml, output_jsonl)
    print("Enriched email data written to", output_jsonl)

    instagram_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/instagram.jsonl"
    socials_yaml = "/Users/blairjdaniel/AI-Assistant-Springs/config/socials_response.yaml"
    output_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/enriched/instagram_enriched.jsonl"
    process_instagram_posts(instagram_jsonl, socials_yaml, output_jsonl)
    print("Enriched Instagram data written to", output_jsonl)

    website_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/website.jsonl"
    socials_yaml = "/Users/blairjdaniel/AI-Assistant-Springs/config/socials_response.yaml"
    output_jsonl = "/Users/blairjdaniel/AI-Assistant-Springs/data/enriched/website_enriched.jsonl"
    process_website_posts(website_jsonl, socials_yaml, output_jsonl)
    print("Enriched website data written to", output_jsonl)

    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook_clean.txt"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook.json"
    with open(input_path, "r") as infile:
        text = infile.read()
    parsed_entries = parse_brandbook(text)
    with open(output_path, "w") as outfile:
        json.dump(parsed_entries, outfile, indent=4)
    print("Parsed brand book saved to", output_path)

    input_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/pre-output/gpt.json"
    output_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/training_data.jsonl"
    with open(input_path, "r") as infile:
        conversations = json.load(infile)
    with open(output_path, "w") as outfile:
        for conversation in conversations:
            example = convert_conversation_to_example(conversation)
            outfile.write(json.dumps(example) + "\n")

    data_dir = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs"
    combined_data = load_all_jsonl_files(data_dir)
    print("Combined dataset count:", len(combined_data))

    file_path = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs/brandbook.jsonl"
    normalized_brandbook = load_normalized_jsonl(file_path)
    print("Normalized DataFrame shape:", normalized_brandbook.shape)

    txt_directory = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs"
    df_txt = load_txt_files_to_df(txt_directory)
    print("Loaded", len(df_txt), "text files")
    print(df_txt.head())

    data_dir = "/Users/blairjdaniel/AI-Assistant-Springs/data/outputs"
    combined_data = load_all_jsonl_files(data_dir)  # or your combined DataFrame
    combined_df = pd.DataFrame(combined_data).fillna("")  # Ensure NaNs become empty strings
    train_df, test_df = split_train_test(combined_df,
                                         test_size=0.2,
                                         random_state=42)
    print("Train DataFrame shape:", train_df.shape)
    print("Test DataFrame shape:", test_df.shape)