import json
import argparse
import os
import base64
from openai import OpenAI
from tqdm import tqdm
import re
from pathlib import Path
import time
import ast


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


def normalize_object_name(name):
    """Normalize object name for comparison"""
    if not name:
        return ""

    # Convert to lowercase and strip whitespace
    name = name.lower().strip()

    # Handle ROV case
    if name == "rov":
        return "rov"

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)

    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)

    return name.strip()


def extract_json_from_output(output):
    """Extract JSON content from model output"""
    if not output:
        return None

    try:
        # First try to parse the entire output directly
        return json.loads(output)
    except:
        pass

    # Try to find JSON array
    json_patterns = [
        r'\[\s*\{.*?\}\s*\]',  # Array format
        r'\{.*?\}',  # Single object format
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, output, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

    # If all fail, try ast.literal_eval
    try:
        # Try to extract something that looks like a list
        if '[' in output and ']' in output:
            start = output.find('[')
            end = output.rfind(']') + 1
            json_str = output[start:end]
            return ast.literal_eval(json_str)
    except:
        pass

    return None


def calculate_bbox_iou(box1, box2):
    """Calculate IoU of two bounding boxes"""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0

    try:
        # Convert to float
        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]

        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union
    except:
        return 0.0


def evaluate_object_prediction(predicted, actual):
    """Evaluate if object prediction is correct"""
    if not predicted or not actual:
        return False

    # Normalize
    predicted_norm = normalize_object_name(predicted)
    actual_norm = normalize_object_name(actual)

    # Exact match
    if predicted_norm == actual_norm:
        return True

    # Contains match
    if actual_norm in predicted_norm or predicted_norm in actual_norm:
        return True

    return False


def evaluate_bbox_prediction(predicted_bbox, actual_bbox, iou_threshold=0.5):
    """Evaluate if bounding box prediction is correct"""
    if not predicted_bbox or not actual_bbox:
        return False

    iou = calculate_bbox_iou(predicted_bbox, actual_bbox)
    return iou >= iou_threshold


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
    parser = argparse.ArgumentParser(description='Sonar data GPT-4o evaluation script')
    parser.add_argument('--input_file', type=str,
                        default='./sona_test_data.json',
                        help='Input JSON file path')
    parser.add_argument('--output_file', type=str,
                        default='./sonar_results.json',
                        help='Output result JSON file path')
    parser.add_argument('--api_key', type=str, required=True,
                        help='OpenAI API key')
    parser.add_argument('--api_base', type=str, required=True,
                        help='API base URL')
    parser.add_argument('--model_name', type=str,
                        default='gpt-4o',
                        help='Model name')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save interval (save after processing this many images)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index position for processing')
    parser.add_argument('--iou_threshold', type=float, default=0.7,
                        help='Bounding box IoU threshold, higher value means more accurate')

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
    object_correct_count = 0
    bbox_correct_count = 0
    both_correct_count = 0
    error_count = 0
    valid_json_count = 0
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
                        if result.get('valid_json'):
                            valid_json_count += 1
                            if result.get('object_correct'):
                                object_correct_count += 1
                            if result.get('bbox_correct'):
                                bbox_correct_count += 1
                            if result.get('both_correct'):
                                both_correct_count += 1
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
            actual_object = item.get('object', '')
            actual_bbox = item.get('bnd_box', [])
            prompt = item.get('input', '')

            # Convert to absolute path
            absolute_image_path = get_absolute_image_path(relative_image_path, args.input_file)

            # Check if image file exists
            if not os.path.exists(absolute_image_path):
                print(f"\nWarning: Image file does not exist {absolute_image_path}")
                model_output = "FileNotFound"
                parsed_json = None
                predicted_object = ""
                predicted_bbox = []
                object_correct = False
                bbox_correct = False
                both_correct = False
                valid_json = False
                error_count += 1
            else:
                # Call API
                print(f"\nProcessing image {i + 1}/{len(data)}: {os.path.basename(absolute_image_path)}")
                model_output = call_gpt4o_api(absolute_image_path, prompt, client, args.model_name)

                # Parse JSON output
                parsed_json = extract_json_from_output(model_output)

                if parsed_json is None or model_output == "Error":
                    predicted_object = ""
                    predicted_bbox = []
                    object_correct = False
                    bbox_correct = False
                    both_correct = False
                    valid_json = False
                    if model_output == "Error":
                        error_count += 1
                else:
                    valid_json = True
                    valid_json_count += 1

                    # Extract first detection result
                    if isinstance(parsed_json, list) and len(parsed_json) > 0:
                        first_detection = parsed_json[0]
                    elif isinstance(parsed_json, dict):
                        first_detection = parsed_json
                    else:
                        first_detection = {}

                    predicted_object = first_detection.get('object', '')
                    predicted_bbox = first_detection.get('bnd_box', [])

                    # Evaluate results
                    object_correct = evaluate_object_prediction(predicted_object, actual_object)
                    bbox_correct = evaluate_bbox_prediction(predicted_bbox, actual_bbox, args.iou_threshold)
                    both_correct = object_correct and bbox_correct

                    # Update statistics
                    if object_correct:
                        object_correct_count += 1
                    if bbox_correct:
                        bbox_correct_count += 1
                    if both_correct:
                        both_correct_count += 1

            processed_count += 1

            # Print processing results
            print(f"Actual object: {actual_object}")
            print(f"Actual bbox: {actual_bbox}")
            print(f"Model output: {model_output[:200]}...")
            print(f"Parsed JSON: {parsed_json}")
            print(f"Predicted object: {predicted_object}")
            print(f"Predicted bbox: {predicted_bbox}")
            print(f"Object correct: {'Yes' if object_correct else 'No'}")
            print(f"Bbox correct: {'Yes' if bbox_correct else 'No'}")
            print(f"Both correct: {'Yes' if both_correct else 'No'}")

            # Calculate current accuracy
            if valid_json_count > 0:
                object_accuracy = object_correct_count / valid_json_count * 100
                bbox_accuracy = bbox_correct_count / valid_json_count * 100
                both_accuracy = both_correct_count / valid_json_count * 100
                valid_rate = valid_json_count / processed_count * 100

                print(f"Valid JSON rate: {valid_rate:.1f}% ({valid_json_count}/{processed_count})")
                print(f"Object accuracy: {object_accuracy:.1f}% ({object_correct_count}/{valid_json_count})")
                print(f"Bbox accuracy: {bbox_accuracy:.1f}% ({bbox_correct_count}/{valid_json_count})")
                print(f"Both correct rate: {both_accuracy:.1f}% ({both_correct_count}/{valid_json_count})")

            # Calculate IoU if both bounding boxes exist
            iou = 0.0
            if predicted_bbox and actual_bbox:
                iou = calculate_bbox_iou(predicted_bbox, actual_bbox)
                print(f"Bounding box IoU: {iou:.3f}")

            # Save result
            result_item = {
                'image_path': relative_image_path,
                'absolute_image_path': absolute_image_path,
                'actual_object': actual_object,
                'actual_bbox': actual_bbox,
                'prompt': prompt,
                'model_output': model_output,
                'parsed_json': parsed_json,
                'predicted_object': predicted_object,
                'predicted_bbox': predicted_bbox,
                'object_correct': object_correct,
                'bbox_correct': bbox_correct,
                'both_correct': both_correct,
                'valid_json': valid_json,
                'bbox_iou': iou,
                'normalized_actual_object': normalize_object_name(actual_object),
                'normalized_predicted_object': normalize_object_name(predicted_object)
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
            if valid_json_count > 0:
                pbar.set_description(f"Obj:{object_accuracy:.1f}% Bbox:{bbox_accuracy:.1f}% Both:{both_accuracy:.1f}%")

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
    print("\n" + "=" * 60)
    print("Final Evaluation Results:")
    print(f"Total images processed: {processed_count}")
    print(f"Successful API calls: {processed_count - error_count}")
    print(f"API call errors: {error_count}")
    print(f"Valid JSON outputs: {valid_json_count}")

    if valid_json_count > 0:
        object_accuracy = object_correct_count / valid_json_count * 100
        bbox_accuracy = bbox_correct_count / valid_json_count * 100
        both_accuracy = both_correct_count / valid_json_count * 100
        valid_rate = valid_json_count / processed_count * 100

        print(f"Valid JSON rate: {valid_rate:.2f}% ({valid_json_count}/{processed_count})")
        print("-" * 40)
        print(f"Object-only correct count: {object_correct_count}")
        print(f"Object-only accuracy: {object_accuracy:.2f}%")
        print("-" * 40)
        print(f"Bbox-only correct count: {bbox_correct_count}")
        print(f"Bbox-only accuracy: {bbox_accuracy:.2f}% (IoU >= {args.iou_threshold})")
        print("-" * 40)
        print(f"Object + Bbox both correct count: {both_correct_count}")
        print(f"Object + Bbox both accuracy: {both_accuracy:.2f}%")

        # Calculate average IoU
        total_iou = 0
        iou_count = 0
        for result in results:
            if result.get('bbox_iou', 0) > 0:
                total_iou += result['bbox_iou']
                iou_count += 1

        if iou_count > 0:
            avg_iou = total_iou / iou_count
            print(f"Average bounding box IoU: {avg_iou:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()