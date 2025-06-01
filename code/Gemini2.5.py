import json
import os
from datetime import datetime
import base64
import time
import argparse # 引入argparse模块

# --- Google Gemini API Setup ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("CRITICAL: GOOGLE_API_KEY environment variable not found. Please set it.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Configuration (These now represent the JUDGE model) ---
# IMPORTANT: 'gemini-2.5-pro-preview-05-06' is not a standard public Gemini model name via genai.
# Please ensure this specific model name is valid for your API key.
# For general use, 'gemini-1.5-pro-latest' is a robust choice for vision tasks.
MODEL_TO_USE = "gemini-2.5-pro-preview-05-06" # This is the judge model
EVALUATION_MODEL_NAME_FOR_OUTPUT = "gemini-2.5-pro-preview-05-06" # This is the judge model's name in output files

try:
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
    print(f"Gemini client initialized for judge model: {MODEL_TO_USE}")
except Exception as e:
    print(f"Failed to initialize Gemini judge model: {e}.")
    API_CLIENT_ENABLED = False
    gemini_model = None
    exit(1)

# --- Paths and Constants (These will be dynamically set by argparse) ---
# Set initial defaults, which will be overridden by command-line arguments
JSON_SURVEY_DATA_PATH = '' # Path to the JSON file containing original prompts and questions
ORIGINAL_IMAGE_BASE_DIR = '' # Base directory for the *original* images referenced in JSON_SURVEY_DATA_PATH
GENERATED_IMAGE_ROOT_DIR = '' # Root directory for the *generated* images (e.g., 'outputs/MODEL_NAME')
MODEL_TO_EVALUATE_NAME = '' # The name of the model being evaluated (e.g., 'StableDiffusionXL')

RESULTS_DIR = '' # Output directory for evaluation results
EVALUATIONS_FILE = '' # Full path to the evaluation results JSON file

RETRY_DELAY_SECONDS = 20
MAX_RETRIES = 3

# --- Helper Functions (remain largely the same) ---
def load_survey_data(json_path_str):
    json_path_norm = os.path.normpath(json_path_str)
    if not os.path.exists(json_path_norm):
        print(f"ERROR: JSON survey data file not found at {json_path_norm}")
        return []
    try:
        with open(json_path_norm, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i, item in enumerate(data):
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
    if "Questions_reality" in case and case["Questions_reality"]:
        en_questions = [q.strip() for q in case["Questions_reality"].split("\n") if q.strip()]

    cleaned_en_questions = []
    for q_en in en_questions:
        if q_en and q_en[0].isdigit() and '. ' in q_en:
            cleaned_en_questions.append(q_en.split('. ', 1)[1])
        else:
            cleaned_en_questions.append(q_en)

    combined_questions = []
    for i, q_en in enumerate(cleaned_en_questions):
        combined_questions.append({"en": q_en, "zh": "N/A", "original_question_index": i})
    
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

def load_existing_evaluations():
    # Use the global EVALUATIONS_FILE path
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
    # Ensure RESULTS_DIR exists before saving
    if not os.path.exists(RESULTS_DIR):
        try:
            os.makedirs(RESULTS_DIR)
            print(f"Created results directory: {RESULTS_DIR}")
        except OSError as e:
            print(f"Error creating results directory {RESULTS_DIR}: {e}")
            return False

    all_evaluations_dict[case_id_str] = evaluation_data_dict
    try:
        with open(EVALUATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_evaluations_dict, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving evaluations to {EVALUATIONS_FILE}: {e}")
        return False

def call_model_for_full_evaluation(image_path_to_evaluate_str, case_data_obj, reality_qs_for_case, attempt=1):
    if not API_CLIENT_ENABLED or not gemini_model:
        print(f"Gemini client not available. Skipping {MODEL_TO_USE} evaluation.")
        return None

    base64_image = image_to_base64(image_path_to_evaluate_str) # This is the generated image to evaluate
    if not base64_image:
        return None
    
    try:
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

    contents = [
        full_prompt_text,
        image_part
    ]

    generation_config = {
        "max_output_tokens": 1500,
        "temperature": 0.1,
    }

    try:
        print(f"    Sending request with params: model='{MODEL_TO_USE}', max_output_tokens={generation_config['max_output_tokens']}, temp={generation_config['temperature']}")
        
        response = gemini_model.generate_content(
            contents,
            generation_config=generation_config
        )
        
        if response.parts and response.parts[0].text:
            model_response_content = response.parts[0].text
        elif hasattr(response, 'text') and response.text:
            model_response_content = response.text
        else:
            print(f"  Warning for {case_id}: Gemini API response structure unexpected or parts/text missing.")
            print(f"  Full API Response (first 500 chars): {str(response)[:500]}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"  Prompt was blocked: {response.prompt_feedback.block_reason.name}")
            if response.candidates and response.candidates[0].finish_reason:
                print(f"  Candidate finish reason: {response.candidates[0].finish_reason.name}")
                if response.candidates[0].finish_reason.name == "SAFETY":
                    print(f"  Candidate blocked due to safety settings: {response.candidates[0].safety_ratings}")
            return None

        if not model_response_content:
            print(f"  Error for {case_id}: Model returned empty or null content. Raw content: '{model_response_content}'")
            return None

        cleaned_content = model_response_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            model_response_content = cleaned_content.strip()
        elif not (cleaned_content.startswith("{") and cleaned_content.endswith("}")):
            print(f"  Warning for {case_id}: Model response not clean JSON. Raw response snippet: {model_response_content[:300]}")

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
            
        final_eval_data = {
            "case_info": {
                "original_index_in_survey_file": case_data_obj.get("original_index_in_survey_file", -1),
                "caseId": case_id,
                "title_EN": case_data_obj.get("title", "N/A"),
                "originalImageFile": case_data_obj.get("image", "N/A"), # Renamed for clarity
                "evaluatedGeneratedImageFile": image_path_to_evaluate_str, # Explicitly store path to the generated image that was evaluated
                "prompt_EN": prompt_en,
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
        time.sleep(RETRY_DELAY_SECONDS)
        return call_model_for_full_evaluation(image_path_to_evaluate_str, case_data_obj, reality_qs_for_case, attempt + 1)
    except google_exceptions.BlockedPromptException as e:
        print(f"  Prompt for {case_id} was blocked by safety settings: {e}. Adjust safety settings or prompt content.")
    except google_exceptions.ServiceUnavailable as e:
        print(f"  Gemini API service unavailable for {case_id}: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...")
        time.sleep(RETRY_DELAY_SECONDS)
        return call_model_for_full_evaluation(image_path_to_evaluate_str, case_data_obj, reality_qs_for_case, attempt + 1)
    except google_exceptions.InternalServerError as e:
        print(f"  Gemini API internal server error for {case_id}: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...")
        time.sleep(RETRY_DELAY_SECONDS)
        return call_model_for_full_evaluation(image_path_to_evaluate_str, case_data_obj, reality_qs_for_case, attempt + 1)
    except Exception as e:
        print(f"  An unexpected error occurred calling Gemini API for case {case_id}: {e}")

    if attempt < MAX_RETRIES:
        print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
        time.sleep(RETRY_DELAY_SECONDS)
        return call_model_for_full_evaluation(image_path_to_evaluate_str, case_data_obj, reality_qs_for_case, attempt + 1)
    else:
        print(f"  Max retries reached for case {case_id}. Skipping.")
    return None

# --- Main Processing Logic ---
def main():
    global JSON_SURVEY_DATA_PATH, ORIGINAL_IMAGE_BASE_DIR, GENERATED_IMAGE_ROOT_DIR, MODEL_TO_EVALUATE_NAME, RESULTS_DIR, EVALUATIONS_FILE

    parser = argparse.ArgumentParser(description="Evaluate generated images for lighting realism using Gemini as a judge.")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the JSON survey data file (e.g., data/merged_light_images.json). "
                             "The directory of this file will also be used as the base for original images.")
    parser.add_argument('--output', type=str, required=True,
                        help="Base directory for generated model outputs (e.g., outputs/MODEL_NAME). "
                             "Evaluation results will be saved within this directory. "
                             "The last part of this path will be used as the evaluated MODEL_NAME.")
    # Optional: Allow changing the judge model name if needed, but not strictly asked for in the prompt.
    # parser.add_argument('--judge_model', type=str, default=MODEL_TO_USE,
    #                     help="Name of the Gemini model to use for judging (e.g., gemini-1.5-pro-latest).")

    args = parser.parse_args()

    # Set global paths based on parsed arguments
    JSON_SURVEY_DATA_PATH = os.path.normpath(args.input)
    # The original images (e.g., 'direct_light_images/1.1.png') are expected to be relative to the input JSON's directory.
    ORIGINAL_IMAGE_BASE_DIR = os.path.dirname(JSON_SURVEY_DATA_PATH) 

    GENERATED_IMAGE_ROOT_DIR = os.path.normpath(args.output) # This is `outputs/MODEL_NAME`
    MODEL_TO_EVALUATE_NAME = os.path.basename(GENERATED_IMAGE_ROOT_DIR) # Extracts 'MODEL_NAME'

    RESULTS_DIR = GENERATED_IMAGE_ROOT_DIR # Results will be saved inside `outputs/MODEL_NAME`
    # Evaluation file name now includes *both* the evaluated model and the judge model.
    EVALUATIONS_FILE = os.path.normpath(os.path.join(RESULTS_DIR, f'{MODEL_TO_EVALUATE_NAME}_judged_by_{EVALUATION_MODEL_NAME_FOR_OUTPUT}_evaluations.json'))

    if not API_CLIENT_ENABLED or not gemini_model:
        print(f"Gemini client for {MODEL_TO_USE} is not initialized. Cannot proceed.")
        return

    # Ensure results directory exists before starting processing
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

    print(f"Loading existing evaluations for {MODEL_TO_EVALUATE_NAME} from: {EVALUATIONS_FILE}")
    evaluations_on_disk = load_existing_evaluations()
    print(f"Found {len(evaluations_on_disk)} existing evaluations for this run.")

    new_evals_this_run = 0
    failed_evals_this_run = 0

    for i, case_data in enumerate(all_survey_cases):
        case_id = str(case_data.get("index"))
        if not case_id:
            print(f"Warning: Case at original_index_in_survey_file {case_data.get('original_index_in_survey_file', 'N/A')} is missing 'index' field. Skipping.")
            continue

        print(f"\nProcessing case {i+1}/{len(all_survey_cases)}: ID '{case_id}' for model '{MODEL_TO_EVALUATE_NAME}'")

        if case_id in evaluations_on_disk:
            print(f"  Evaluation for case {case_id} already exists. Skipping.")
            continue

        original_image_filename_relative = case_data.get("image") # e.g., "direct_light_images/1.1.png"
        if not original_image_filename_relative:
            print(f"  Warning: Case {case_id} is missing 'image' filename. Skipping.")
            continue
        
        # Construct the full path to the *generated* image for this case
        # Expected structure: outputs/{MODEL_NAME}/images/{category}/{image_file_name_from_json}
        # Example: outputs/StableDiffusionXL/images/direct_light_images/1.1.png
        generated_image_full_path = os.path.join(GENERATED_IMAGE_ROOT_DIR, 'images', original_image_filename_relative)

        if not os.path.exists(generated_image_full_path):
            print(f"  Warning: Generated image file not found for case {case_id} at {generated_image_full_path}. Skipping.")
            continue

        reality_questions_for_this_case = get_reality_questions(case_data)

        if new_evals_this_run > 0 or failed_evals_this_run > 0 : 
             if (new_evals_this_run + failed_evals_this_run) % 5 == 0:
                 print("    Pausing for 5s to avoid potential rate limits...")
                 time.sleep(5)
             else:
                 time.sleep(1)

        # Pass the path to the *generated image* for evaluation
        evaluation_result_data = call_model_for_full_evaluation(generated_image_full_path, case_data, reality_questions_for_this_case)

        if evaluation_result_data:
            print(f"  SUCCESS: {EVALUATION_MODEL_NAME_FOR_OUTPUT} ({MODEL_TO_USE}) Evaluation for {case_id} on '{MODEL_TO_EVALUATE_NAME}'")
            save_evaluation(case_id, evaluation_result_data, evaluations_on_disk)
            new_evals_this_run += 1
        else:
            print(f"  FAILURE: Failed to evaluate case {case_id} for '{MODEL_TO_EVALUATE_NAME}' with {MODEL_TO_USE} after retries (if applicable).")
            failed_evals_this_run +=1

    print("\n--- Evaluation Run Summary ---")
    print(f"Evaluated Model: {MODEL_TO_EVALUATE_NAME}")
    print(f"Judge Model Used: {MODEL_TO_USE} (Output label: {EVALUATION_MODEL_NAME_FOR_OUTPUT})")
    print(f"Total cases in survey file: {len(all_survey_cases)}")
    print(f"Total evaluations on disk before run: {len(evaluations_on_disk) - new_evals_this_run}")
    print(f"Newly evaluated cases in this run: {new_evals_this_run}")
    print(f"Failed evaluations in this run: {failed_evals_this_run}")
    print(f"Total evaluations now on disk: {len(evaluations_on_disk)}")
    print(f"All results saved to: {EVALUATIONS_FILE}")
    print("Processing complete.")


if __name__ == "__main__":
    print(f"Starting Automated Evaluation Script...")
    print(f"!!! CRITICAL: ENSURE YOUR GOOGLE_API_KEY IS VALID AND THE JUDGE MODEL NAME '{MODEL_TO_USE}' IS CORRECT FOR DIRECT GEMINI API. !!!")
    
    # Pre-checks for arguments are handled by argparse's `required=True`
    # Path validity will be checked within main() after arguments are parsed.

    main()
