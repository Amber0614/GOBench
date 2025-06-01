import gradio as gr
import json
import os
from datetime import datetime

# Set this to the absolute path where your images are stored
IMAGE_BASE_DIR = r"D:\work\phd_work\vqa\4.22lqe\github\data\data"  # REPLACE THIS
# Set this to the path of your JSON survey data
JSON_SURVEY_DATA_PATH = r'D:\work\phd_work\vqa\4.22lqe\github\data\data\data_total.json' # REPLACE THIS
# JSON_SURVEY_DATA_PATH = r'D:\work\phd_work\vqa\4.22lqe\data_0512\merged_merged_original_images_no_chinese.json' # REPLACE THIS



# Load the survey data from JSON file
def load_survey_data(json_path):
    json_path = os.path.normpath(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON survey data file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {json_path}")
        return []

# Global variables
survey_data = []
current_case_index = 0
participant_name = ""
responses = []

def save_responses():
    if not participant_name:
        print("Participant name is empty, not saving responses.")
        return "No responses saved (participant name missing)."
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    filename = os.path.join(results_dir, f"{participant_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "participantName": participant_name,
            "responses": responses,
            "surveyDataPath": JSON_SURVEY_DATA_PATH,
            "imageBaseDir": IMAGE_BASE_DIR
        }, f, ensure_ascii=False, indent=2)
    return filename

def display_current_case_text():
    global current_case_index, survey_data
    if not survey_data:
         return "## Error: No survey data loaded.\nCannot proceed."
    if current_case_index >= len(survey_data):
        saved_file = save_responses()
        return f"## Survey Completed!\nThank you for your participation, {participant_name}.\nYour responses have been saved to {saved_file}"
    current_case = survey_data[current_case_index]
    case_md_text = f"""## Case {current_case_index + 1} of {len(survey_data)}
### {current_case.get('title', 'N/A')}
**Prompt:** {current_case.get('prompt', 'N/A')}
"""
    return case_md_text

def get_image_path_for_case(case_data):
    image_filename = case_data.get('image', '')
    if image_filename:
        image_path = os.path.join(IMAGE_BASE_DIR, image_filename)
        image_path = os.path.normpath(image_path)
        if os.path.exists(image_path):
            return image_path
        else:
            print(f"Image not found: {image_path}")
            return None
    return None

def get_reality_questions_en(case):
    en_questions = []
    if "Questions_reality" in case and case["Questions_reality"]:
        questions_text = case["Questions_reality"]
        if isinstance(questions_text, str):
            en_questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        else:
            print(f"Warning: 'Questions_reality' for case {case.get('index', 'N/A')} is not a string: {questions_text}")
    return en_questions

with gr.Blocks(title="Image Evaluation Study: Direct Light Scenario", theme=gr.themes.Default()) as app:
    gr.Markdown("# Image Evaluation Study: Direct Light Scenario")
    
    with gr.Group(visible=True) as registration_group:
        gr.Markdown("## Please enter your name to begin the evaluation process")
        name_input = gr.Textbox(label="Your Name", placeholder="Enter your full name")
        start_button = gr.Button("Start Evaluation", variant="primary")
    
    with gr.Group(visible=False) as question_group:
        with gr.Row():
            with gr.Column(scale=3): 
                case_md = gr.Markdown(elem_id="case-description")
                
                gr.Markdown("### Reality Questions")
                reality_radios = []
                for i in range(5): 
                    r = gr.Radio(
                        choices=["Yes", "No", "Cannot Determine"],
                        label=f"Question {i+1}", 
                        interactive=True,
                        visible=False
                    )
                    reality_radios.append(r)
                
                gr.Markdown("### Aesthetics Rating")
                aesthetics_rating = gr.Radio(
                    choices=[("1 (Poor)", "1"), ("2", "2"), ("3 (Average)", "3"), ("4", "4"), ("5 (Excellent)", "5")],
                    label="Rate the aesthetics (1=Poor, 5=Excellent)",
                    interactive=True
                )
                
                #==================================================================
                # 修改指令一致性评分的选项和标签 (英文版)
                #==================================================================
                gr.Markdown("### Instruction Consistency: Object & Lighting") # 英文标题
                consistency_rating = gr.Radio(
                    choices=[
                        ("1: Inaccurate subject and weak lighting", "1"),
                        ("2: Partially accurate subject and fair lighting", "2"),
                        ("3: Accurate subject but fair lighting", "3"),
                        ("4: Accurate subject and good lighting", "4"),
                        ("5: Subject and lighting are both very accurate", "5")
                    ],
                    label="Please rate instruction consistency based on the descriptions:", # 英文标签
                    interactive=True
                )
                #==================================================================
                # END 修改
                #==================================================================
            
            with gr.Column(scale=2): 
                image_display = gr.Image(label="Case Image", type="filepath", interactive=False, height=750)

                # --- START: Jump to Case Functionality UI (ADDED) ---
                with gr.Row(elem_id="jump_row"):
                    # Initial value will be set by on_start
                    jump_input = gr.Number(label="Jump to case number", precision=0, minimum=1, step=1, interactive=True, value=1)
                    jump_button = gr.Button("Jump")
                # --- END: Jump to Case Functionality UI (ADDED) ---
        
        submit_button = gr.Button("Submit and Continue", variant="primary")
    
    with gr.Group(visible=False) as completion_group:
        completion_display_md = gr.Markdown() 
    
    def on_start(name):
        global participant_name, survey_data, current_case_index, responses
        current_image_path = None
        if not name.strip():
            gr.Warning("Please enter your name to begin!")
            initial_radio_updates = [gr.update(visible=False) for _ in reality_radios]
            # --- MODIFIED OUTPUTS for jump_input initialization ---
            return (gr.update(), gr.update(visible=False), gr.update(visible=True),
                    *initial_radio_updates, None, gr.update(value=1, maximum=1))
        
        participant_name = name
        current_case_index = 0
        responses = [] # Ensure responses is cleared for new participant
        survey_data = load_survey_data(JSON_SURVEY_DATA_PATH)
        if not survey_data:
            gr.Error("Failed to load survey data. Cannot start.")
            initial_radio_updates = [gr.update(visible=False) for _ in reality_radios]
            # --- MODIFIED OUTPUTS for jump_input initialization ---
            return ("Error: Survey data not loaded.", gr.update(visible=False), gr.update(visible=True),
                    *initial_radio_updates, None, gr.update(value=1, maximum=1))

        # Initialize responses list based on total cases if not already done
        if not responses and len(survey_data) > 0:
            responses.extend([None] * len(survey_data))

        case_md_text = display_current_case_text()
        current_case_data = survey_data[current_case_index]
        current_image_path = get_image_path_for_case(current_case_data)
        questions_en = get_reality_questions_en(current_case_data)
        updated_radio_comps = []
        for i in range(len(reality_radios)):
            if i < len(questions_en):
                en_q = questions_en[i]
                if en_q and en_q[0].isdigit() and '. ' in en_q: en_q = en_q.split('. ', 1)[1]
                updated_radio_comps.append(gr.update(label=f"{en_q}", value=None, visible=True))
            else:
                updated_radio_comps.append(gr.update(visible=False, value=None))
        
        # --- MODIFIED OUTPUTS: Added jump_input to the return values ---
        return (case_md_text, gr.update(visible=True), gr.update(visible=False),
                *updated_radio_comps, current_image_path, gr.update(value=current_case_index + 1, maximum=len(survey_data)))
    
    start_button.click(
        on_start,
        inputs=[name_input],
        # --- MODIFIED OUTPUTS: Added jump_input to the list ---
        outputs=[case_md, question_group, registration_group] + reality_radios + [image_display, jump_input]
    )
    
    def on_submit(*reality_responses_values, aesthetics, consistency):
        global current_case_index, responses, survey_data
        next_image_path = None
        if not survey_data or current_case_index >= len(survey_data):
             gr.Error("Survey data missing or index out of bounds.")
             no_change_radio_updates = [gr.update() for _ in reality_radios]
             # --- MODIFIED OUTPUTS: Added jump_input to the return for error state ---
             return (gr.update(value="Error state, please restart."), gr.update(), gr.update(), 
                     *no_change_radio_updates, gr.update(), gr.update(), gr.update(), gr.update(value=current_case_index + 1))
        
        current_case_data_for_submit = survey_data[current_case_index]
        questions_en_current = get_reality_questions_en(current_case_data_for_submit)
        num_current_questions = len(questions_en_current)
        visible_reality_responses = reality_responses_values[:num_current_questions]
        
        if not all(r is not None for r in visible_reality_responses) or not aesthetics or not consistency:
            gr.Warning("Please answer all displayed questions and ratings.")
            no_change_radio_updates = [gr.update() for _ in reality_radios]
            # --- MODIFIED OUTPUTS: Added jump_input to the return for warning state ---
            return (gr.update(), gr.update(), gr.update(), 
                    *no_change_radio_updates, gr.update(), gr.update(), gr.update(), gr.update(value=current_case_index + 1))
        
        # Ensure responses list is long enough
        while len(responses) <= current_case_index:
            responses.append(None)

        responses[current_case_index] = {
            "caseIndex": current_case_index,
            "caseTitle": survey_data[current_case_index].get("title", "N/A"),
            "caseId": survey_data[current_case_index].get("index", ""),
            "imageFile": survey_data[current_case_index].get("image", "N/A"),
            "realityResponses": list(visible_reality_responses),
            "aestheticsRating": int(aesthetics),
            "consistencyRating": int(consistency),
            "timestamp": datetime.now().isoformat()
        }
        current_case_index += 1
        
        if current_case_index < len(survey_data):
            case_md_text_next = display_current_case_text()
            next_case_data = survey_data[current_case_index]
            next_image_path = get_image_path_for_case(next_case_data)
            questions_en_next = get_reality_questions_en(next_case_data)
            updated_radio_comps_next = []
            for i in range(len(reality_radios)):
                if i < len(questions_en_next):
                    en_q = questions_en_next[i]
                    if en_q and en_q[0].isdigit() and '. ' in en_q: en_q = en_q.split('. ', 1)[1]
                    updated_radio_comps_next.append(gr.update(label=f"{en_q}", value=None, visible=True))
                else:
                    updated_radio_comps_next.append(gr.update(visible=False, value=None))
            # --- MODIFIED OUTPUTS: Added jump_input to the return for next case ---
            return (case_md_text_next, gr.update(visible=True), gr.update(visible=False),
                    *updated_radio_comps_next, None, None, next_image_path, gr.update(value=current_case_index + 1))
        else: 
            completion_text = display_current_case_text()
            hidden_reality_updates = [gr.update(visible=False, value=None) for _ in reality_radios]
            # --- MODIFIED OUTPUTS: Added jump_input to the return for completion ---
            # Set jump_input to invisible or update its value, assuming survey completion.
            return (completion_text, gr.update(visible=False), gr.update(visible=True),
                    *hidden_reality_updates, None, None, None, gr.update(visible=False)) # Hide jump_input on completion

    submit_button.click(
        on_submit,
        inputs=reality_radios + [aesthetics_rating, consistency_rating],
        # --- MODIFIED OUTPUTS: Added jump_input to the list ---
        outputs=[case_md, question_group, completion_group] + reality_radios + [aesthetics_rating, consistency_rating, image_display, jump_input]
    )

    # --- START: Jump to Case Functionality (ADDED) ---
    def jump_to_case_callback(target_serial_input, 
                              # Pass all current UI values to save them before jumping
                              *current_reality_radio_values, 
                              current_aesthetics_rating, 
                              current_consistency_rating):
        
        global current_case_index, responses, survey_data, participant_name
        
        # Initial checks
        if not survey_data:
            gr.Error("Survey data not loaded. Cannot jump.")
            # Return current UI state (no changes)
            return (display_current_case_text(), get_image_path_for_case(survey_data[current_case_index]) if survey_data else None,
                    *[gr.update()] * len(reality_radios), gr.update(), gr.update(), gr.update(value=current_case_index + 1))

        if target_serial_input is None:
            gr.Warning("Please enter a case number to jump to.")
            # Return current UI state (no changes)
            return (display_current_case_text(), get_image_path_for_case(survey_data[current_case_index]),
                    *[gr.update()] * len(reality_radios), gr.update(), gr.update(), gr.update(value=current_case_index + 1))

        target_idx_raw = int(target_serial_input) - 1 # Convert serial number to 0-based index

        # 1. Save current case's responses before jumping
        # Only save if current ratings are filled for the current case
        if 0 <= current_case_index < len(survey_data):
            current_case_data_for_save = survey_data[current_case_index]
            questions_en_current = get_reality_questions_en(current_case_data_for_save)
            num_current_questions = len(questions_en_current)
            visible_reality_responses_to_save = current_reality_radio_values[:num_current_questions]

            # Check if all relevant fields for the current case are non-None
            # Ensure responses list is large enough before attempting to set
            while len(responses) <= current_case_index:
                responses.append(None)
                
            # If all ratings are complete, save them
            if all(r is not None for r in visible_reality_responses_to_save) and \
               current_aesthetics_rating is not None and current_consistency_rating is not None:
                
                responses[current_case_index] = {
                    "caseIndex": current_case_index,
                    "caseTitle": current_case_data_for_save.get("title", "N/A"),
                    "caseId": current_case_data_for_save.get("index", ""),
                    "imageFile": current_case_data_for_save.get("image", "N/A"),
                    "realityResponses": list(visible_reality_responses_to_save),
                    "aestheticsRating": int(current_aesthetics_rating),
                    "consistencyRating": int(current_consistency_rating),
                    "timestamp": datetime.now().isoformat()
                }
                gr.Info(f"Saved progress for case {current_case_index + 1}.")
            else:
                gr.Warning(f"Case {current_case_index + 1} has incomplete ratings. Not saving this case before jumping.")
        
        # 2. Validate the jump target
        if not (0 <= target_idx_raw < len(survey_data)):
            gr.Warning(f"Invalid case number: {target_serial_input}. Please enter a number between 1 and {len(survey_data)}.")
            # If target invalid, just re-display the current case (no jump)
            # This requires re-loading the state for current_case_index
            case_md_curr = display_current_case_text()
            image_path_curr = get_image_path_for_case(survey_data[current_case_index])
            
            # Re-populate current reality questions and ratings
            current_case_data = survey_data[current_case_index]
            questions_en = get_reality_questions_en(current_case_data)
            updated_radio_comps = []
            for i in range(len(reality_radios)):
                if i < len(questions_en):
                    en_q = questions_en[i]
                    if en_q and en_q[0].isdigit() and '. ' in en_q: en_q = en_q.split('. ', 1)[1]
                    updated_radio_comps.append(gr.update(label=f"{en_q}", visible=True))
                else:
                    updated_radio_comps.append(gr.update(visible=False, value=None)) # Hide if no question
            
            # Restore existing answers if any for the current case
            current_aes_val, current_con_val = None, None
            if len(responses) > current_case_index and responses[current_case_index] is not None:
                case_response = responses[current_case_index]
                current_aes_val = case_response.get("aestheticsRating")
                current_con_val = case_response.get("consistencyRating")
                if current_aes_val is not None: current_aes_val = str(current_aes_val)
                if current_con_val is not None: current_con_val = str(current_con_val)

                reality_ans = case_response.get("realityResponses", [])
                for i in range(len(updated_radio_comps)):
                    if i < len(reality_ans) and reality_ans[i] is not None:
                        updated_radio_comps[i] = gr.update(value=reality_ans[i], visible=updated_radio_comps[i]['visible'])
                    else:
                        updated_radio_comps[i] = gr.update(value=None, visible=updated_radio_comps[i]['visible'])
            else: # If no existing response for current case, ensure all are cleared
                for i in range(len(updated_radio_comps)):
                    updated_radio_comps[i] = gr.update(value=None, visible=updated_radio_comps[i]['visible'])

            return (case_md_curr, image_path_curr, *updated_radio_comps, gr.update(value=current_aes_val), gr.update(value=current_con_val), gr.update(value=current_case_index + 1))
        
        # 3. Perform the jump and update UI for the target case
        current_case_index = target_idx_raw # Update global index

        case_md_jump = display_current_case_text() # Uses updated global index
        image_path_jump = get_image_path_for_case(survey_data[current_case_index])
        
        # Populate reality questions and ratings for the target case
        target_case_data = survey_data[current_case_index]
        questions_en_target = get_reality_questions_en(target_case_data)
        updated_radio_comps_target = []
        for i in range(len(reality_radios)):
            if i < len(questions_en_target):
                en_q = questions_en_target[i]
                if en_q and en_q[0].isdigit() and '. ' in en_q: en_q = en_q.split('. ', 1)[1]
                updated_radio_comps_target.append(gr.update(label=f"{en_q}", visible=True))
            else:
                updated_radio_comps_target.append(gr.update(visible=False, value=None)) # Hide if no question

        jump_aes_val, jump_con_val = None, None
        if len(responses) > current_case_index and responses[current_case_index] is not None:
            case_response_target = responses[current_case_index]
            jump_aes_val = case_response_target.get("aestheticsRating")
            jump_con_val = case_response_target.get("consistencyRating")
            if jump_aes_val is not None: jump_aes_val = str(jump_aes_val)
            if jump_con_val is not None: jump_con_val = str(jump_con_val)

            reality_ans_target = case_response_target.get("realityResponses", [])
            for i in range(len(updated_radio_comps_target)):
                if i < len(reality_ans_target) and reality_ans_target[i] is not None:
                    updated_radio_comps_target[i] = gr.update(value=reality_ans_target[i], visible=updated_radio_comps_target[i]['visible'])
                else:
                    updated_radio_comps_target[i] = gr.update(value=None, visible=updated_radio_comps_target[i]['visible'])
        else: # If no existing response for target case, ensure all are cleared
            for i in range(len(updated_radio_comps_target)):
                updated_radio_comps_target[i] = gr.update(value=None, visible=updated_radio_comps_target[i]['visible'])
            
        return (case_md_jump, image_path_jump, *updated_radio_comps_target, gr.update(value=jump_aes_val), gr.update(value=jump_con_val), gr.update(value=current_case_index + 1))

    jump_button.click(
        jump_to_case_callback,
        # Inputs needed for jump: the jump input, and all current UI element values to save them
        inputs=[jump_input] + reality_radios + [aesthetics_rating, consistency_rating],
        # Outputs to update: case_md, image_display, all reality_radios, aesthetics, consistency, and jump_input itself
        outputs=[case_md, image_display] + reality_radios + [aesthetics_rating, consistency_rating, jump_input]
    )
    # --- END: Jump to Case Functionality (ADDED) ---


if __name__ == "__main__":
    print("Starting Image Evaluation Survey...")
    print(f"Ensure IMAGE_BASE_DIR ('{IMAGE_BASE_DIR}') and JSON_SURVEY_DATA_PATH ('{JSON_SURVEY_DATA_PATH}') are correct.")
    if not os.path.isdir(IMAGE_BASE_DIR):
        print(f"WARNING: IMAGE_BASE_DIR '{IMAGE_BASE_DIR}' does not exist or is not a directory!")
    if not os.path.isfile(JSON_SURVEY_DATA_PATH):
        print(f"WARNING: JSON_SURVEY_DATA_PATH '{JSON_SURVEY_DATA_PATH}' does not exist or is not a file!")
    app.launch()