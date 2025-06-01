import json
import os
from datetime import datetime
import base64
import time

# --- Google Gemini API Setup ---
import google.generativeai as genai # Renamed from 'google.genai' for standard usage
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings
from google.api_core import exceptions as google_exceptions # For more specific error handling

# Ensure your GOOGLE_API_KEY is set as an environment variable
# or replace "YOUR-API-KEY" with your actual API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Recommended way
# GOOGLE_API_KEY = "YOUR-ACTUAL-GEMINI-API-KEY" # Alternative: hardcode (NOT RECOMMENDED for production)

if not GOOGLE_API_KEY:
    print("CRITICAL: GOOGLE_API_KEY environment variable not found. Please set it.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Configuration ---
MODEL_TO_USE = "gemini-2.5-pro-preview-05-06" # Changed to a standard public Gemini Vision model
EVALUATION_MODEL_NAME_FOR_OUTPUT = "gemini-2.5-pro-preview-05-06" # Reflects the actual model being used

try:
    # Initialize Gemini model with safety settings
    # Adjust safety settings as needed. These are for demonstration to be less strict.
    gemini_model = genai.GenerativeModel(
        model_name=MODEL_TO_USE,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    API_CLIENT_ENABLED = True
    print(f"Gemini client initialized for model: {MODEL_TO_USE}")
except Exception as e:
    print(f"Failed to initialize Gemini model: {e}.")
    API_CLIENT_ENABLED = False
    gemini_model = None
    exit(1)

# --- Paths and Constants ---
# Assuming these paths are relative to the script's execution directory
# You might need to adjust IMAGE_BASE_DIR to be an absolute path or relative to a common root
IMAGE_BASE_DIR = r"xx\data_0512" # Absolute path for clarity
JSON_SURVEY_DATA_PATH = r"xx\merged_light_images.json" # Absolute path
RESULTS_DIR = f'results_{EVALUATION_MODEL_NAME_FOR_OUTPUT}_eval'
EVALUATIONS_FILE = os.path.normpath(os.path.join(RESULTS_DIR, f'{EVALUATION_MODEL_NAME_FOR_OUTPUT}_merged_evaluations.json'))
RETRY_DELAY_SECONDS = 20
MAX_RETRIES = 3 # Increased retries to 3 for robustness

# --- Helper Functions (remain largely the same, but ensure paths are handled correctly) ---
def load_survey_data(json_path_str):
    json_path_norm = os.path.normpath(json_path_str)
    if not os.path.exists(json_path_norm):
        print(f"ERROR: JSON survey data file not found at {json_path_norm}")
        return []
    try:
        with open(json_path_norm, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i, item in enumerate(data): # Add original index from the input file
                item['original_index_in_survey_file'] = i
            return data
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {json_path_norm}")
        return []
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading {json_path_norm}: {e}")
        return []

def get_reality_questions(case):
    en_questions = []
    # We will only use "Questions_reality" if it exists.
    if "Questions_reality" in case and case["Questions_reality"]:
        en_questions = [q.strip() for q in case["Questions_reality"].split("\n") if q.strip()]

    cleaned_en_questions = []
    for q_en in en_questions:
        if q_en and q_en[0].isdigit() and '. ' in q_en:
            cleaned_en_questions.append(q_en.split('. ', 1)[1])
        else:
            cleaned_en_questions.append(q_en)

    # We create a dummy list for combined_questions if no real Chinese questions exist.
    combined_questions = []
    for i, q_en in enumerate(cleaned_en_questions):
        combined_questions.append({"en": q_en, "zh": "N/A", "original_question_index": i}) # zh will be N/A
    
    return combined_questions

def image_to_base64(image_path_str):
    image_path_norm = os.path.normpath(image_path_str)
    try:
        with open(image_path_norm, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path_norm}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path_norm} to base64: {e}")
        return None

# --- Model Evaluation Logic ---
def load_existing_evaluations():
    if os.path.exists(EVALUATIONS_FILE):
        try:
            with open(EVALUATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {EVALUATIONS_FILE}. Treating as empty.")
            return {}
        except Exception as e:
            print(f"Warning: Error loading evaluations from {EVALUATIONS_FILE}: {e}. Treating as empty.")
            return {}
    return {}

def save_evaluation(case_id_str, evaluation_data_dict, all_evaluations_dict):
    all_evaluations_dict[case_id_str] = evaluation_data_dict
    try:
        with open(EVALUATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_evaluations_dict, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving evaluations to {EVALUATIONS_FILE}: {e}")
        return False

def call_model_for_full_evaluation(image_path_str, case_data_obj, reality_qs_for_case, attempt=1):
    if not API_CLIENT_ENABLED or not gemini_model:
        print(f"Gemini client not available. Skipping {MODEL_TO_USE} evaluation.")
        return None

    base64_image = image_to_base64(image_path_str)
    if not base64_image:
        return None
    
    try:
        # Convert base64 string to a `generative_models.Part` for Gemini
        # Ensure the correct mime type, e.g., "image/jpeg"
        image_part = genai.GenerativeModel.Part.from_data(
            data=base64.b64decode(base64_image),
            mime_type="image/jpeg" # Assuming JPEG. Adjust if your images are PNG, etc.
        )
    except Exception as e:
        print(f"Error creating image part for Gemini: {e}")
        return None

    case_id = case_data_obj.get("index")
    prompt_en = case_data_obj.get("prompt", "N/A")

    reality_questions_prompt_text = "Reality Questions to Answer (provide only 'Yes', 'No', or 'Cannot Determine' for each):\n"
    if reality_qs_for_case:
        for i, q_obj in enumerate(reality_qs_for_case):
            # Only use English question if Chinese is N/A
            q_display = f"Q_EN: \"{q_obj['en']}\""
            if q_obj['zh'] != "N/A":
                q_display += f" / Q_ZH: \"{q_obj['zh']}\""
            reality_questions_prompt_text += f"  {i+1}. {q_display}\n"
    else:
        reality_questions_prompt_text += "  (No specific reality questions provided for this case.)\n"

    full_prompt_text = f"""You are an expert image critic and AI model evaluator.
You will be given an image and the English and Chinese prompts used for its generation.
Additionally, you will be provided with a list of specific "Reality Questions" about the image.

Your task is to evaluate the image based on ALL three criteria: Reality Questions, Aesthetics, and Instruction Consistency.

1.  **Reality Question Answers**: For each numbered reality question provided below, answer with only "Yes", "No", or "Cannot Determine".
    The answers should be provided in a list corresponding to the order of the questions. If no reality questions are provided, return an empty list for "realityAnswers_Eval".

2.  **Aesthetics Rating**: Rate the overall visual appeal and quality of the image.
    Scale: 1 (Poor), 2 (Fair), 3 (Average), 4 (Good), 5 (Excellent).

3.  **Instruction Consistency Rating**: Rate how well the image's content and style align with the provided English and Chinese generation prompts. Consider:
    - Accuracy of the main subject(s).
    - Representation of specific details (e.g., lighting, environment, actions).
    - Adherence to style/mood.
    - For "direct light" or "merged light" scenarios, assess if the specified lighting is clearly demonstrated.
    Scale:
    1: Main subject largely inaccurate OR key instructions (like lighting) completely misrepresented.
    2: Main subject partially accurate, significant deviations in instructions.
    3: Main subject accurate, but some secondary instructions/nuances (like lighting quality) weakly represented.
    4: Main subject accurate, most instructions (including lighting) well-represented.
    5: Main subject highly accurate, all instructions (including nuanced lighting) exceptionally well-represented.

VERY IMPORTANT: Provide your response ONLY as a single, valid JSON object. Do not include any text before or after the JSON object.
The JSON object must have the following structure:
{{
  "realityAnswers_Eval": ["AnswerQ1", "AnswerQ2", ...],
  "aestheticsRating_Eval": <integer 1-5>,
  "consistencyRating_Eval": <integer 1-5>,
  "reasoning_Eval": "Brief overall reasoning for your ratings (max 100 words).",
  "model_used_for_eval": "{EVALUATION_MODEL_NAME_FOR_OUTPUT}"
}}

Image Generation Prompts:
English Prompt: "{prompt_en}"

{reality_questions_prompt_text}
Return your complete evaluation STRICTLY as a JSON object according to the instructions.
Ensure the "realityAnswers_Eval" list has an answer for each question asked (or is empty if no questions were asked).
The entire response must be a single JSON object starting with {{ and ending with }}.
"""

    print(f"  Requesting {MODEL_TO_USE} evaluation for case {case_id} (Attempt {attempt})...")
    model_response_content = None

    # For Gemini, the content is a list of text and image parts
    contents = [
        full_prompt_text,
        image_part
    ]

    generation_config = {
        "max_output_tokens": 1500, # Corresponds to max_tokens
        "temperature": 0.1,
    }

    try:
        print(f"    Sending request with params: model='{MODEL_TO_USE}', max_output_tokens={generation_config['max_output_tokens']}, temp={generation_config['temperature']}")
        
        # Call Gemini API directly
        response = gemini_model.generate_content(
            contents,
            generation_config=generation_config
        )
        
        # Check if response has text content
        if response.parts and response.parts[0].text:
            model_response_content = response.parts[0].text
        elif hasattr(response, 'text') and response.text: # Fallback for simpler responses
            model_response_content = response.text
        else:
            print(f"  Warning for {case_id}: Gemini API response structure unexpected or parts/text missing.")
            print(f"  Full API Response (first 500 chars): {str(response)[:500]}")
            # Check for blocking reasons
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"  Prompt was blocked: {response.prompt_feedback.block_reason.name}")
            if response.candidates and response.candidates[0].finish_reason:
                print(f"  Candidate finish reason: {response.candidates[0].finish_reason.name}")
                if response.candidates[0].finish_reason.name == "SAFETY":
                    print(f"  Candidate blocked due to safety settings: {response.candidates[0].safety_ratings}")
            return None # No content to parse

        if not model_response_content:
            print(f"  Error for {case_id}: Model returned empty or null content. Raw content: '{model_response_content}'")
            return None

        # Clean JSON response (Gemini might also wrap in ```json)
        cleaned_content = model_response_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            model_response_content = cleaned_content.strip()
        elif not (cleaned_content.startswith("{") and cleaned_content.endswith("}")):
            print(f"  Warning for {case_id}: Model response not clean JSON. Raw response snippet: {model_response_content[:300]}")
            # Fall through to attempt parsing anyway, JSONDecodeError will catch it if it's not valid.

        parsed_eval = json.loads(model_response_content)

        model_reality_answers = parsed_eval.get("realityAnswers_Eval", [])
        if not isinstance(model_reality_answers, list):
            print(f"  Warning for {case_id}: Model 'realityAnswers_Eval' was not a list. Defaulting to empty list.")
            model_reality_answers = []

        num_expected_reality_answers = len(reality_qs_for_case)
        if len(model_reality_answers) != num_expected_reality_answers:
            print(f"  Warning for {case_id}: Model provided {len(model_reality_answers)} reality answers, but {num_expected_reality_answers} were asked.")

        if not isinstance(parsed_eval.get("aestheticsRating_Eval"), int) or \
           not isinstance(parsed_eval.get("consistencyRating_Eval"), int):
            print(f"  Error for {case_id}: Model response missing or has invalid type for aesthetics/consistency ratings. Received: {parsed_eval.get('aestheticsRating_Eval')}, {parsed_eval.get('consistencyRating_Eval')}")
            # Depending on strictness, you might return None here
            
        final_eval_data = {
            "case_info": {
                "original_index_in_survey_file": case_data_obj.get("original_index_in_survey_file", -1),
                "caseId": case_id,
                "title_EN": case_data_obj.get("title", "N/A"),
                "title_ZH": case_data_obj.get("中文标题", "N/A"), # Will likely be N/A after previous script
                "imageFile": case_data_obj.get("image", "N/A"),
                "prompt_EN": prompt_en,
                "prompt_ZH": prompt_zh, # Will likely be N/A after previous script
                "reality_questions_asked_to_model": reality_qs_for_case
            },
            "model_evaluation": {
                "realityAnswers_Eval": model_reality_answers,
                "aestheticsRating_Eval": parsed_eval.get("aestheticsRating_Eval"),
                "consistencyRating_Eval": parsed_eval.get("consistencyRating_Eval"),
                "reasoning_Eval": parsed_eval.get("reasoning_Eval", ""),
                "model_used_for_eval": parsed_eval.get("model_used_for_eval", EVALUATION_MODEL_NAME_FOR_OUTPUT),
                "evaluationTimestamp": datetime.now().isoformat()
            }
        }
        return final_eval_data

    except json.JSONDecodeError as e:
        print(f"  Error for {case_id}: Model response was not valid JSON: {e}")
        print(f"  Received content that failed to parse: '{model_response_content}'")
    except (google_exceptions.ResourceExhausted, google_exceptions.RateLimitExceeded) as e:
        print(f"  Rate limit or resource exhausted for {case_id}: {e}")
        error_message = str(e).lower()
        if "rate limit" in error_message or "resource exhausted" in error_message:
            print(f"  Rate limit possibly exceeded. Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)
            return call_model_for_full_evaluation(image_path_str, case_data_obj, reality_qs_for_case, attempt + 1)
    except google_exceptions.BlockedPromptException as e:
        print(f"  Prompt for {case_id} was blocked by safety settings: {e}. Adjust safety settings or prompt content.")
    except google_exceptions.ServiceUnavailable as e:
        print(f"  Gemini API service unavailable for {case_id}: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...")
        time.sleep(RETRY_DELAY_SECONDS)
        return call_model_for_full_evaluation(image_path_str, case_data_obj, reality_qs_for_case, attempt + 1)
    except google_exceptions.InternalServerError as e:
        print(f"  Gemini API internal server error for {case_id}: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...")
        time.sleep(RETRY_DELAY_SECONDS)
        return call_model_for_full_evaluation(image_path_str, case_data_obj, reality_qs_for_case, attempt + 1)
    except Exception as e: # Catch any other unexpected errors
        print(f"  An unexpected error occurred calling Gemini API for case {case_id}: {e}")

    if attempt < MAX_RETRIES:
        print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
        time.sleep(RETRY_DELAY_SECONDS)
        return call_model_for_full_evaluation(image_path_str, case_data_obj, reality_qs_for_case, attempt + 1)
    else:
        print(f"  Max retries reached for case {case_id}. Skipping.")
    return None

# --- Main Processing Logic (ensure paths are created before use) ---
def main():
    if not API_CLIENT_ENABLED or not gemini_model:
        print(f"Gemini client for {MODEL_TO_USE} is not initialized. Cannot proceed.")
        return

    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        try:
            os.makedirs(RESULTS_DIR)
            print(f"Created results directory: {RESULTS_DIR}")
        except OSError as e:
            print(f"Error creating results directory {RESULTS_DIR}: {e}")
            exit(1)

    print(f"Loading survey data from: {JSON_SURVEY_DATA_PATH}")
    all_survey_cases = load_survey_data(JSON_SURVEY_DATA_PATH)
    if not all_survey_cases:
        print("No survey cases loaded. Exiting.")
        return
    print(f"Loaded {len(all_survey_cases)} survey cases.")

    print(f"Loading existing {EVALUATION_MODEL_NAME_FOR_OUTPUT} evaluations from: {EVALUATIONS_FILE}")
    evaluations_on_disk = load_existing_evaluations()
    print(f"Found {len(evaluations_on_disk)} existing {EVALUATION_MODEL_NAME_FOR_OUTPUT} evaluations.")

    new_evals_this_run = 0
    failed_evals_this_run = 0

    for i, case_data in enumerate(all_survey_cases):
        case_id = str(case_data.get("index"))
        if not case_id:
            print(f"Warning: Case at original_index_in_survey_file {case_data.get('original_index_in_survey_file', 'N/A')} is missing 'index' field. Skipping.")
            continue

        print(f"\nProcessing case {i+1}/{len(all_survey_cases)}: ID '{case_id}'")

        if case_id in evaluations_on_disk:
            print(f"  Evaluation for case {case_id} already exists. Skipping.")
            continue

        # Adjust for `image_paths` or `image` depending on your JSON structure
        # The previous script outputs `image_paths` but this script expects `image` (single string).
        # Assuming the merged_light_images.json still has the 'image' field for single images.
        image_filename = case_data.get("image")
        if not image_filename:
            print(f"  Warning: Case {case_id} is missing 'image' filename. Skipping.")
            continue
        
        # Construct the full image path
        image_path = os.path.join(IMAGE_BASE_DIR, image_filename)
        if not os.path.exists(image_path):
            print(f"  Warning: Image file not found for case {case_id} at {image_path}. Skipping.")
            continue

        reality_questions_for_this_case = get_reality_questions(case_data)

        # Add some delay between actual API calls
        if new_evals_this_run > 0 or failed_evals_this_run > 0 : 
             if (new_evals_this_run + failed_evals_this_run) % 5 == 0: # Longer delay every 5 calls
                 print("    Pausing for 5s to avoid potential rate limits...")
                 time.sleep(5)
             else:
                 time.sleep(1)


        evaluation_result_data = call_model_for_full_evaluation(image_path, case_data, reality_questions_for_this_case)

        if evaluation_result_data:
            print(f"  SUCCESS: {EVALUATION_MODEL_NAME_FOR_OUTPUT} ({MODEL_TO_USE}) Evaluation for {case_id}")
            save_evaluation(case_id, evaluation_result_data, evaluations_on_disk)
            new_evals_this_run += 1
        else:
            print(f"  FAILURE: Failed to evaluate case {case_id} with {MODEL_TO_USE} after retries (if applicable).")
            failed_evals_this_run +=1

    print("\n--- Evaluation Run Summary ---")
    print(f"Evaluation Model Used: {MODEL_TO_USE} (Output label: {EVALUATION_MODEL_NAME_FOR_OUTPUT})")
    print(f"Total cases in survey file: {len(all_survey_cases)}")
    print(f"Total evaluations on disk before run: {len(evaluations_on_disk) - new_evals_this_run}")
    print(f"Newly evaluated cases in this run: {new_evals_this_run}")
    print(f"Failed evaluations in this run: {failed_evals_this_run}")
    print(f"Total evaluations now on disk: {len(evaluations_on_disk)}")
    print(f"All results saved to: {EVALUATIONS_FILE}")
    print("Processing complete.")


if __name__ == "__main__":
    print(f"Starting {EVALUATION_MODEL_NAME_FOR_OUTPUT} ({MODEL_TO_USE}) Automated Full Survey Evaluation Script...")
    print(f"!!! CRITICAL: ENSURE YOUR GOOGLE_API_KEY IS VALID AND THE MODEL NAME '{MODEL_TO_USE}' IS CORRECT FOR DIRECT GEMINI API. !!!")
    
    # Pre-check paths
    if not os.path.isdir(IMAGE_BASE_DIR):
        print(f"CRITICAL ERROR: IMAGE_BASE_DIR '{IMAGE_BASE_DIR}' does not exist or is not a directory!")
        exit(1)
    if not os.path.isfile(JSON_SURVEY_DATA_PATH):
        print(f"CRITICAL ERROR: JSON_SURVEY_DATA_PATH '{JSON_SURVEY_DATA_PATH}' does not exist or is not a file!")
        exit(1)

    main()