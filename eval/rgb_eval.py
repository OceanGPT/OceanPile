import json
import argparse
import os
import base64
from openai import OpenAI
from tqdm import tqdm
import re
from pathlib import Path
import time


def encode_image(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_gpt4o_api(image_path, prompt, client, model_name):
    """Call GPT-4o API for image recognition"""
    try:
        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API call error: {e}")
        return "Error"


def normalize_species_name(name):
    """Normalize species name for comparison"""
    if not name:
        return ""

    # Convert to lowercase and strip whitespace
    name = name.lower().strip()

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)

    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)

    return name.strip()


def extract_species_from_output(output):
    """Extract species name from model output - directly return model output"""
    if not output:
        return ""

    # Directly return model output stripped of whitespace
    return output.strip()


def evaluate_prediction(predicted, actual):
    """Evaluate whether the prediction is correct"""
    if not predicted or not actual:
        return False

    # Normalize
    predicted_norm = normalize_species_name(predicted)
    actual_norm = normalize_species_name(actual)

    # If prediction is unknown, return False directly
    if "unknown" in predicted_norm:
        return False

    # Method 1: Exact match
    if predicted_norm == actual_norm:
        return True

    # Method 2: Contains match
    if actual_norm in predicted_norm:
        return True

    # Method 3: Check original format for containment (handling case differences)
    predicted_lower = predicted.lower()
    actual_lower = actual.lower()

    # Check underscore format
    if actual_lower in predicted_lower:
        return True

    # Check space format
    actual_with_space = actual.replace('_', ' ').lower()
    if actual_with_space in predicted_lower:
        return True

    return False


def get_absolute_image_path(relative_path, json_file_path):
    """Convert relative path to absolute path"""
    # Get the directory of the JSON file
    json_dir = os.path.dirname(json_file_path)

    # Build absolute image path
    absolute_path = os.path.join(json_dir, relative_path)

    return absolute_path


def save_results(results, output_path):
    """Save results to JSON file"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving file: {e}")


def main():
    parser = argparse.ArgumentParser(description='GPT-4o image recognition evaluation script')
    parser.add_argument('--input_file', type=str,
                        default='./OceanVision_test_data.json',
                        help='Input JSON file path')
    parser.add_argument('--output_file', type=str,
                        default='./rgb_results.json',
                        help='Output result JSON file path')
    parser.add_argument('--api_key', type=str, required=True,
                        help='OpenAI API key')
    parser.add_argument('--api_base', type=str, required=True,
                        help='API base URL')
    parser.add_argument('--model_name', type=str,
                        default='gpt-5',
                        help='Model name')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save interval (save after processing this many images)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index position for processing')

    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist {args.input_file}")
        return

    # Read input data
    print("Reading input data...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Total images to process: {len(data)}")

    # Initialize statistics variables
    results = []
    correct_count = 0
    unknown_count = 0
    error_count = 0
    processed_count = 0

    # If output file exists, try to load previous results
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                if len(existing_results) > args.start_index:
                    results = existing_results
                    # Recalculate statistics
                    for result in results:
                        if result.get('is_correct'):
                            correct_count += 1
                        if result.get('predicted_species', '').lower().strip() == 'unknown':
                            unknown_count += 1
                        if result.get('model_output') == 'Error':
                            error_count += 1
                    processed_count = len(results)
                    print(f"Found existing results file, continuing from index {len(results)}")
        except Exception as e:
            print(f"Error reading existing results file: {e}")

    # Create progress bar
    pbar = tqdm(range(len(results), len(data)), desc="Processing progress")

    try:
        for i in pbar:
            item = data[i]
            relative_image_path = item.get('image_path', '')
            category = item.get('category', '')
            prompt = item.get('prompt', '')

            # Convert to absolute path
            absolute_image_path = get_absolute_image_path(relative_image_path, args.input_file)

            # Check if image file exists
            if not os.path.exists(absolute_image_path):
                print(f"\nWarning: Image file does not exist {absolute_image_path}")
                model_output = "FileNotFound"
                predicted_species = ""
                is_correct = False
                error_count += 1
            else:
                # Call API
                print(f"\nProcessing image {i + 1}/{len(data)}: {os.path.basename(absolute_image_path)}")
                model_output = call_gpt4o_api(absolute_image_path, prompt, client, args.model_name)

                # Extract species name
                predicted_species = extract_species_from_output(model_output)

                # Evaluate result
                is_correct = evaluate_prediction(predicted_species, category)

                # Update statistics
                if predicted_species.lower().strip() == 'unknown':
                    unknown_count += 1
                elif model_output == "Error":
                    error_count += 1
                elif is_correct:
                    correct_count += 1

            processed_count += 1

            # Print processing results
            print(f"Actual category: {category}")
            print(f"Model output: {model_output}")
            print(f"Extracted species: {predicted_species}")
            print(f"Normalized actual: {normalize_species_name(category)}")
            print(f"Normalized predicted: {normalize_species_name(predicted_species)}")
            print(f"Result: {'Correct' if is_correct else 'Incorrect'}")

            # Calculate current accuracy
            valid_predictions = processed_count - error_count
            if valid_predictions > 0:
                accuracy = correct_count / valid_predictions * 100
                unknown_rate = unknown_count / valid_predictions * 100
                print(f"Current accuracy: {accuracy:.2f}% ({correct_count}/{valid_predictions})")
                print(f"Unknown rate: {unknown_rate:.2f}% ({unknown_count}/{valid_predictions})")

            # Save result
            result_item = {
                'image_path': relative_image_path,
                'absolute_image_path': absolute_image_path,
                'category': category,
                'prompt': prompt,
                'model_output': model_output,
                'predicted_species': predicted_species,
                'normalized_category': normalize_species_name(category),
                'normalized_prediction': normalize_species_name(predicted_species),
                'is_correct': is_correct
            }

            if i < len(results):
                results[i] = result_item
            else:
                results.append(result_item)

            # Save results periodically
            if (i + 1) % args.save_interval == 0:
                print(f"\nSaving intermediate results to {args.output_file}")
                save_results(results, args.output_file)

            # Update progress bar description
            if valid_predictions > 0:
                pbar.set_description(f"Accuracy: {accuracy:.1f}%, Unknown: {unknown_rate:.1f}%")

            # Avoid too frequent API calls
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupt detected, saving current results...")
        save_results(results, args.output_file)
        print(f"Results saved to {args.output_file}")
        return

    # Save final results
    print(f"\nSaving final results to {args.output_file}")
    save_results(results, args.output_file)

    # Output final statistics
    print("\n" + "=" * 50)
    print("Final Evaluation Results:")
    print(f"Total images processed: {processed_count}")
    print(f"Successful API calls: {processed_count - error_count}")
    print(f"API call errors: {error_count}")

    if processed_count - error_count > 0:
        valid_predictions = processed_count - error_count
        accuracy = correct_count / valid_predictions * 100
        unknown_rate = unknown_count / valid_predictions * 100

        print(f"Correct predictions: {correct_count}")
        print(f"Incorrect predictions: {valid_predictions - correct_count - unknown_count}")
        print(f"Unknown outputs: {unknown_count}")
        print(f"Overall accuracy: {accuracy:.2f}%")
        print(f"Unknown rate: {unknown_rate:.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    main()