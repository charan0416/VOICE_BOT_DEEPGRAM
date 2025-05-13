import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    SpeakOptions,
    PrerecordedOptions,
    FileSource,
    SpeakResponse,
    PrerecordedResponse
)
import io
import logging
import asyncio

# For Gemini
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Deepgram Client Initialization ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables.")
try:
    dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    logger.info("Deepgram client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Deepgram client: {e}")
    raise

# --- Google Gemini Client Initialization ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"  # Or "gemini-1.0-pro-latest", "gemini-1.5-pro-latest" etc.
gemini_configured = False
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. LLM features will use mock responses.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        logger.info(f"Google Gemini client configured successfully for model {GEMINI_MODEL_NAME}.")
    except Exception as e:
        logger.error(f"Failed to configure Google Gemini client: {e}. LLM features may be disabled or mocked.")
        gemini_configured = False

# --- CSV Data Handling ---
DATA_FILE = 'cleaned_data.csv'
customer_df = None


def load_data():
    global customer_df
    try:
        customer_df = pd.read_csv(DATA_FILE)
        for col in ['Credit Score', 'Monthly Debt', 'Current Loan Amount', 'Random_Name']:
            if col not in customer_df.columns:
                customer_df[col] = None if col == 'Random_Name' else 0

        customer_df['Credit Score'] = customer_df['Credit Score'].fillna(0)
        customer_df['Monthly Debt'] = pd.to_numeric(customer_df['Monthly Debt'], errors='coerce').fillna(0)
        customer_df['Current Loan Amount'] = pd.to_numeric(customer_df['Current Loan Amount'], errors='coerce').fillna(
            0)
        customer_df['Random_Name'] = customer_df['Random_Name'].astype(str).fillna("Unknown Customer")
        logger.info("CSV data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: The file {DATA_FILE} was not found.")
        customer_df = pd.DataFrame(columns=['Random_Name', 'Credit Score', 'Monthly Debt', 'Current Loan Amount'])
    except Exception as e:
        logger.error(f"Error loading or processing CSV: {e}")
        customer_df = pd.DataFrame(columns=['Random_Name', 'Credit Score', 'Monthly Debt', 'Current Loan Amount'])


load_data()

# --- LLM System Prompt Template ---
LLM_SYSTEM_PROMPT_TEMPLATE = """
You are LoanMate, an advanced AI collections agent for Global Finance Solutions. Your primary objective is to discuss outstanding loan payments with customers in a way that is **exceptionally human-like, deeply empathetic, and highly understanding.** Your goal is not just to collect payments, but to do so while preserving and even enhancing the customer's relationship with Global Finance Solutions.

**Your Core Persona & Emotional Intelligence:**
*   **Empathetic Listener:** Actively listen to the customer. Your first priority is to make them feel heard and understood, especially if they are distressed.
*   **Warm & Approachable:** Your tone should be consistently warm, patient, and reassuring. Avoid sounding robotic, scripted, or judgmental.
*   **Emotionally Astute:** Detect and mirror the customer's emotional state appropriately. If they are sad, be compassionate. If they are frustrated, be patient and understanding. If they are cooperative, be appreciative.
*   **Natural Conversationalist:** Use natural language, vary your sentence structure, use conversational fillers if appropriate (e.g., "I see," "Hmm, I understand," "Well,"), and avoid repetitive phrases. Your responses should flow like a real human conversation.

**Understanding and Responding to Situations:**
*   **Beyond Keywords:** Do not rely solely on keywords. Understand the *intent* and *context* behind the customer's words. If a customer says, "Things have been really tough since the factory closed," understand this implies job loss and financial hardship without them needing to say "I lost my job."
*   **Handling Sensitive Information:**
    *   If a customer mentions health issues (their own or a family member's: "hospital," "sick," "doctor," "surgery," "medical bills"), accidents ("car crash," "injured"), death or bereavement ("passed away," "funeral," "condolences"), job loss ("laid off," "unemployed," "no income"), or general severe hardship ("crisis," "struggling badly," "difficult times"):
        1.  **Prioritize Empathy:** Immediately offer genuine, heartfelt sympathy and acknowledge the difficulty of their situation. For example: "Oh, I am so incredibly sorry to hear you're going through that. That sounds exceptionally challenging, and my thoughts are with you." or "Please accept my deepest condolences. That's a terrible loss, and I can only imagine how difficult this time must be."
        2.  **Gentle Transition:** After expressing empathy, if appropriate and after a slight pause or acknowledgment from them, gently and respectfully inquire if they are in a position to discuss the loan, or if there's anything related to the loan account that might ease their burden slightly (like exploring options if company policy allows).
*   **Payment Discussions:**
    *   **Customer Context (Provided to you):**
        *   Customer Name: {customer_name}
        *   Current Loan Amount: {loan_amount}
        *   Monthly EMI: {monthly_debt}
        *   Credit Score: {credit_score_text}
    *   **Initial Interaction Strategy (VERY IMPORTANT - Follow this based on 'Customer Name' above):**
        *   If "Customer Name" is "{unknown_customer_placeholder}": Your *first and only goal* for your initial response is to politely ask for their full name to look up their account. Example: "Hello, this is LoanMate from Global Finance Solutions. To start, could you please tell me your full name so I can bring up your account details?" Do NOT proceed with any loan details until you have a name. Await their response.
        *   If a specific "Customer Name" (e.g., "John Doe") is provided: Your initial response should be to greet them by name, confirm if it's a good time to talk, and then state the purpose: "Hello {customer_name}, this is LoanMate from Global Finance Solutions. Is this a good time to talk? I'm calling regarding your loan account. The current outstanding amount is {loan_amount}, with a monthly payment of {monthly_debt} which is now due. I was hoping we could discuss that today."
    *   **Inability to Pay Full Amount:** If they cannot pay the full amount, explore the reasons with understanding. "I understand that making the full payment might be difficult right now."
    *   **Offering Partial Payment (50%):** If full payment isn't possible, and after understanding their situation (especially if sensitive), gently suggest a partial payment: "Given what you've shared, I completely understand that the full amount might be a stretch. To help keep the account in good standing and avoid further issues, would it be at all possible to manage at least 50% of that, which would be {half_monthly_debt}, for now?"
    *   **Explaining Consequences:** If they refuse any payment or ask for extensions beyond policy, you must explain the consequences (late fees, credit score impact, further collection). Do this *gently but clearly*, framing it as information to help them avoid negative outcomes, not as a threat. "I want to be transparent about how this works. If the payment isn't made, it can unfortunately lead to things like late fees and could impact your credit score (currently {credit_score_text}), which we certainly want to help you avoid."
    *   **Negotiation (if allowed by policy):** Be open to what the customer *can* do. "What amount do you feel you could manage today, even if it's not the full 50%? Any payment helps."
*   **Call Closure:**
    * If payment agreed (full or partial): Thank them, provide generic payment instructions (e.g., "You can make the payment through our online portal or by calling our payment line."), and end politely.
    * If no consent to talk or final refusal: Politely acknowledge and end the call. For refusal, reiterate awareness of consequences if appropriate.
*   **Maintaining Control & Objective:** While being empathetic, remember the call's purpose. Gently guide the conversation back to the payment if it strays too far for too long, but only after adequately addressing the customer's immediate emotional needs or concerns.
*   **Flexibility:** Be prepared for unexpected questions or statements. Don't get flustered. Use your understanding to respond appropriately. If you don't have an answer, say you'll find out (if that's an option) or politely state the limits of your knowledge.

**Output:**
*   Your response should be plain text, suitable for a Text-to-Speech (TTS) engine.
*   Do NOT include non-verbal cues like "[gentle tone]" in your text response. The TTS engine should handle expressiveness based on its capabilities and the inherent emotion in the text.
*   Keep responses concise and conversational.
"""
UNKNOWN_CUSTOMER_PLACEHOLDER = "the customer (name not yet identified)"

active_call_data = {
    "customer_info": None,
    "conversation_history": [],
    "asked_for_name_in_last_turn": False
}


# --- Helper functions ---
def get_customer_details(name):
    if customer_df is None or customer_df.empty:
        logger.warning("customer_df is not loaded or is empty in get_customer_details.")
        return None
    customer_data = customer_df[customer_df['Random_Name'].str.strip().str.lower() == name.strip().lower()]
    if not customer_data.empty:
        return customer_data.iloc[0].to_dict()
    logger.info(f"Customer '{name}' not found in CSV.")
    return None


def format_currency(amount):
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return str(amount) if amount else "N/A"


def generate_system_prompt(customer_info_dict):
    if customer_info_dict:
        name = customer_info_dict.get('Random_Name', UNKNOWN_CUSTOMER_PLACEHOLDER)
        loan_val = customer_info_dict.get('Current Loan Amount', 0)
        emi_val = customer_info_dict.get('Monthly Debt', 0)
        loan = format_currency(loan_val)
        emi = format_currency(emi_val)
        half_emi = format_currency(float(emi_val) * 0.5 if emi_val else 0)
        score_val = customer_info_dict.get('Credit Score', 0)
        score = 'N/A' if not score_val or score_val == 0 else str(int(score_val))
    else:
        name = UNKNOWN_CUSTOMER_PLACEHOLDER
        loan = 'N/A (details not yet available)'
        emi = 'N/A (details not yet available)'
        half_emi = 'N/A (details not yet available)'
        score = 'N/A (details not yet available)'

    return LLM_SYSTEM_PROMPT_TEMPLATE.format(
        customer_name=name,
        loan_amount=loan,
        monthly_debt=emi,
        half_monthly_debt=half_emi,
        credit_score_text=score,
        unknown_customer_placeholder=UNKNOWN_CUSTOMER_PLACEHOLDER
    )


async def call_gemini_model(system_prompt_text, conversation_history_for_llm):
    if not gemini_configured:
        logger.warning("Gemini client not configured. Using mock LLM response.")
        await asyncio.sleep(0.2)
        if active_call_data["asked_for_name_in_last_turn"] and not active_call_data["customer_info"]:
            # This mock is if LLM asked for name, and user provides it in next turn.
            # For initial call, asked_for_name_in_last_turn is false or customer_info is known.
            return "Thank you. My name is Mock User. What is this call regarding?"
        # If customer_info is known or it's the initial prompt:
        if active_call_data.get("customer_info"):
            return f"Hello {active_call_data['customer_info']['Random_Name']}, this is LoanMate from Global Finance Solutions. Your mock loan of $1000 is due. Can you pay?"
        else:  # No customer info, initial call, mock LLM should ask for name
            return "Hello, this is LoanMate from Global Finance Solutions. Could you please tell me your full name so I can bring up your account details?"

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            system_instruction=system_prompt_text,
            generation_config={"temperature": 0.7}
        )

        logger.info(
            f"Sending to Gemini ({GEMINI_MODEL_NAME}). System prompt applied. History length: {len(conversation_history_for_llm)}")

        # *** FIX: Ensure 'contents' is not empty for the initial call ***
        effective_contents_for_gemini = conversation_history_for_llm
        if not conversation_history_for_llm:  # First turn from model, history is empty
            # Provide a generic prompt to make the model start based on its system_instruction.
            # This prompt acts as the "current turn's input" that generate_content needs.
            effective_contents_for_gemini = "Please begin the conversation according to your system instructions."
            logger.info(
                f"Conversation history is empty. Using initial trigger prompt for Gemini: '{effective_contents_for_gemini}'")

        response = await asyncio.to_thread(
            model.generate_content,
            contents=effective_contents_for_gemini
        )

        bot_response_text = ""
        if response.parts:
            bot_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text') and response.text:
            bot_response_text = response.text
        else:
            logger.error(
                f"Gemini response empty or blocked. Prompt_feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
            block_reason_msg = ""
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback,
                           'block_reason_message') and response.prompt_feedback.block_reason_message:
                    block_reason_msg = f" Reason: {response.prompt_feedback.block_reason_message}."
                elif hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    block_reason_msg = f" Reason code: {response.prompt_feedback.block_reason}."

            return f"My apologies, my response was prevented due to a content filter.{block_reason_msg} Could we try rephrasing or discussing a different aspect?"

        bot_response_text = bot_response_text.strip()
        if not bot_response_text:
            logger.warning("Gemini returned an empty string after stripping.")
            return "I seem to be at a loss for words. Could you try that again?"

        logger.info(f"Gemini response: {bot_response_text}")
        return bot_response_text

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return "I'm sorry, I'm having trouble connecting to my main system right now. Please try again in a moment."


async def text_to_speech(text):
    if not text:
        logger.error("TTS received empty text.")
        text = "I encountered an internal error and cannot respond right now."

    speak_options = SpeakOptions(model="aura-asteria-en", encoding="mp3")
    source = {"text": text}

    try:
        response_obj: SpeakResponse = await asyncio.to_thread(
            dg_client.speak.rest.v("1").stream_memory,
            source,
            speak_options
        )
        if response_obj and hasattr(response_obj, 'stream') and response_obj.stream:
            audio_bytes = response_obj.stream.read()
            if not audio_bytes:
                logger.error("TTS generated empty audio bytes from stream_memory.")
                raise ValueError("TTS generated empty audio.")
            logger.info(f"TTS generated for: '{text[:50]}...', byte length: {len(audio_bytes)}")
            return audio_bytes
        else:
            logger.error(
                f"Deepgram TTS stream_memory did not return a valid response object or stream. Type: {type(response_obj)}")
            raise ValueError("Invalid response from TTS stream_memory.")
    except Exception as e:
        logger.error(f"Error in Deepgram TTS: {e}", exc_info=True)
        fallback_text = "I'm having trouble speaking at the moment. Please try again later."
        source_fallback = {"text": fallback_text}
        try:
            response_fallback_obj: SpeakResponse = await asyncio.to_thread(
                dg_client.speak.rest.v("1").stream_memory,
                source_fallback,
                speak_options
            )
            if response_fallback_obj and hasattr(response_fallback_obj, 'stream') and response_fallback_obj.stream:
                audio_bytes_fallback = response_fallback_obj.stream.read()
                if audio_bytes_fallback:
                    return audio_bytes_fallback
            raise ValueError("Fallback TTS also failed.")
        except Exception as fallback_e:
            logger.error(f"Fallback TTS also failed: {fallback_e}", exc_info=True)
            raise


# --- Flask Routes ---
@app.route('/')
def index():
    active_call_data["customer_info"] = None
    active_call_data["conversation_history"] = []
    active_call_data["asked_for_name_in_last_turn"] = False
    logger.info("Index page loaded, call state reset.")
    return render_template('index.html')


@app.route('/initiate_call', methods=['POST'])
async def initiate_call_route():
    data = request.get_json()
    customer_name_input = data.get('customerName', '').strip()

    active_call_data["conversation_history"] = []
    active_call_data["customer_info"] = None
    active_call_data["asked_for_name_in_last_turn"] = False

    if customer_name_input:
        customer_details = get_customer_details(customer_name_input)
        if customer_details:
            active_call_data["customer_info"] = customer_details
            logger.info(f"Customer '{customer_name_input}' found. Details: {customer_details}")
        else:
            logger.info(f"Customer '{customer_name_input}' provided but not found in CSV.")
            # customer_info remains None, system prompt will guide LLM to ask.
            pass

    system_prompt = generate_system_prompt(active_call_data["customer_info"])

    # Initial bot message (LLM generates greeting or asks for name)
    # Conversation history is empty here, call_gemini_model will handle it.
    bot_response_text = await call_gemini_model(system_prompt, active_call_data["conversation_history"])

    # Heuristic to set if LLM was prompted to ask for a name and did so.
    # Check if customer_info is still None AND the system prompt indicated it was unknown
    if not active_call_data["customer_info"] and UNKNOWN_CUSTOMER_PLACEHOLDER in system_prompt:
        # A more robust check would be to analyze bot_response_text for name-asking phrases.
        # For now, assume if customer unknown and prompt was for unknown, LLM likely asked.
        active_call_data["asked_for_name_in_last_turn"] = True
        logger.info("Initial call, customer unknown, setting asked_for_name_in_last_turn=True")

    active_call_data["conversation_history"].append({"role": "model", "parts": [{"text": bot_response_text}]})

    try:
        audio_bytes = await text_to_speech(bot_response_text)
        return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False,
                         download_name="response.mp3")
    except Exception as e:
        logger.error(f"TTS error during /initiate_call: {e}", exc_info=True)
        return jsonify({"error": "TTS Error", "details": str(e)}), 500


@app.route('/process_audio', methods=['POST'])
async def process_audio_route():
    if 'audio_data' not in request.files:
        logger.warning("No audio data in request.")
        return jsonify({"error": "No audio data"}), 400

    audio_file = request.files['audio_data']

    try:
        buffer_data = audio_file.read()
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-2", smart_format=True, language="en-US")

        logger.info("Sending audio to Deepgram STT...")
        response_stt_obj: PrerecordedResponse = await asyncio.to_thread(
            dg_client.listen.rest.v("1").transcribe_file,
            payload,
            options
        )

        transcript = ""
        if response_stt_obj.results and response_stt_obj.results.channels and response_stt_obj.results.channels[
            0].alternatives:
            transcript = response_stt_obj.results.channels[0].alternatives[0].transcript

        if not transcript:
            logger.warning("STT returned empty transcript.")
            audio_bytes = await text_to_speech("I'm sorry, I didn't catch that. Could you please repeat?")
            return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False,
                             download_name="response.mp3")

        logger.info(f"Deepgram STT transcript: {transcript}")
        active_call_data["conversation_history"].append({"role": "user", "parts": [{"text": transcript}]})

        if active_call_data["asked_for_name_in_last_turn"] and not active_call_data["customer_info"]:
            potential_name = transcript.strip()
            customer_details = get_customer_details(potential_name)
            if customer_details:
                active_call_data["customer_info"] = customer_details
                logger.info(
                    f"Customer identified from transcript: {potential_name} -> {customer_details['Random_Name']}")
            else:
                logger.info(f"Name '{potential_name}' from transcript not found. LLM will continue conversation.")
            active_call_data["asked_for_name_in_last_turn"] = False

        system_prompt = generate_system_prompt(active_call_data["customer_info"])
        bot_response_text = await call_gemini_model(system_prompt, active_call_data["conversation_history"])

        # After LLM response, re-check if it might be asking for name again if still unknown
        if not active_call_data["customer_info"] and UNKNOWN_CUSTOMER_PLACEHOLDER in system_prompt:
            # This is a simple heuristic. A better way is to check bot_response_text for name-asking patterns.
            if "name" in bot_response_text.lower() and (
                    "what is your" in bot_response_text.lower() or "could you tell me" in bot_response_text.lower()):
                active_call_data["asked_for_name_in_last_turn"] = True
                logger.info("LLM response seems to be asking for name again, setting asked_for_name_in_last_turn=True")

        active_call_data["conversation_history"].append({"role": "model", "parts": [{"text": bot_response_text}]})

        MAX_HISTORY_TURNS = 10
        if len(active_call_data["conversation_history"]) > MAX_HISTORY_TURNS * 2:
            active_call_data["conversation_history"] = active_call_data["conversation_history"][
                                                       -(MAX_HISTORY_TURNS * 2):]
            logger.info(f"Conversation history trimmed to last {MAX_HISTORY_TURNS} turns.")

        audio_bytes = await text_to_speech(bot_response_text)
        return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False,
                         download_name="response.mp3")

    except Exception as e:
        logger.error(f"Error in /process_audio: {e}", exc_info=True)
        try:
            error_audio_bytes = await text_to_speech(
                "I've encountered an unexpected problem. Please try your request again later.")
            return send_file(io.BytesIO(error_audio_bytes), mimetype="audio/mpeg", status=500)
        except Exception as tts_e:
            logger.error(f"TTS failed for error message as well: {tts_e}", exc_info=True)
            return jsonify({"error": "Critical System Error", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)