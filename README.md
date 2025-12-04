# AI Personality System (GuppShupp)

This project is a mini **companion AI** that can analyze chat messages, extract structured user memory (preferences, emotional patterns, and facts), and generate responses with different personalities. It is designed to showcase:

- Reasoning and prompt design
- Structured output parsing
- Working with user memory
- A modular, production-style architecture

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GuppShupp
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Open in browser**
   Open [http://localhost:8000](http://localhost:8000) in your web browser.

## How to Use

1. **Enter Chat Messages (up to 30)**
   - In section **"1. Enter Chat Messages"**, paste past chat messages (one message per line).
   - Click **"Analyze Messages"**.
   - The backend will:
     - Call Groq LLM with a structured memory-extraction prompt.
     - Parse the JSON into a `UserMemory` model.
     - Store memory in a SQLite database.
   - The **Extracted Memory** panel shows:
     - **Preferences** (e.g. lifestyle, behavior) with confidence and context
     - **Emotional patterns** (pattern, frequency, intensity, context)
     - **Facts** (personal/background/events with temporal relevance)

2. **Select Personality**
   - In section **"2. Select Personality"**, choose a personality:
     - 🎓 Calm Mentor (`mentor`)
     - 😂 Witty Friend (`friend`)
     - 🧠 Empathetic Therapist (`therapist`)
     - 🤖 Professional Assistant (`professional`)
   - These map to `PersonalityProfile`s in the `PersonalityEngine` with different tone characteristics and prompts.

3. **Test a Single Personality**
   - In section **"3. Test the Personality"**:
     - Type a message in the small input box (e.g. *"What should I cook this Sunday based on what you know about me?"*).
     - Click **"Send"**.
   - The backend calls `/api/generate` and returns a response conditioned on:
     - The selected personality
     - The stored `UserMemory`
   - The response appears in the chat area with a personality tag.

4. **Compare Personalities (Before/After View)**
   - After you have run **Analyze Messages**, the **"Compare Personalities"** button becomes enabled in section **"4. Compare Personalities"**.
   - Type (or re-type) a message in the **"Test the Personality"** input.
   - Click **"Compare Personalities"** (without clearing the input).
   - The frontend calls `/api/compare`, which:
     - Generates a **base response** (neutral/professional).
     - Generates responses for multiple personalities (mentor, friend, therapist, professional).
     - Returns a `PersonalityComparison` object with:
       - `user_message`
       - `base_response`
       - `personality_responses`
       - `comparison_analysis`
       - `recommendations`
   - The comparison panel shows:
     - User message
     - Base response
     - Personality responses side-by-side
     - JSON-style comparison analysis
     - Recommendations about which personality to use when

## Project Structure

- `main.py` – FastAPI application wiring together services and exposing:
  - `/api/analyze` (memory extraction)
  - `/api/generate` (personality-based response)
  - `/api/compare` (before/after personality comparison)
  - Health and user inspection endpoints
- `models/`
  - `memory_models.py` – `UserMemory`, `UserPreference`, `EmotionalPattern`, `UserFact`, request/response models.
  - `personality_models.py` – `PersonalityType`, `ToneCharacteristics`, `PersonalityProfile`, and response models.
- `services/`
  - `llm_client.py` – Groq API client with retries, error handling, and configurable `GROQ_MODEL`.
  - `memory_extractor.py` – Memory extraction logic, prompt design, JSON parsing, and statistics.
  - `personality_engine.py` – Personality profiles and response generation/comparison.
  - `database.py` – SQLite persistence for user memory and conversation data.
- `templates/index.html` – Tailwind-based UI for messages, personality selection, memory display, and comparison.
- `static/script.js` – Frontend logic (analyze, generate, compare personalities).
- `static/style.css` – Additional styling.
- `requirements.txt` – Python dependencies.

## Mapping to Original Requirements

> Given 30 chat messages from a user, build:
> - A memory extraction module that identifies user preferences, user emotional patterns, facts worth remembering
> - A “Personality Engine” that transforms the agent’s reply tone (e.g., calm mentor, witty friend, therapist-style)
> - Show before/after personality response differences

- **Memory extraction module**
  - `services/memory_extractor.py` + `models/memory_models.py` implement structured extraction of:
    - Preferences (by category, with confidence and context)
    - Emotional patterns (pattern, frequency, intensity)
    - Facts (typed, with temporal relevance)
  - Up to 30 messages are accepted and analyzed.

- **Personality Engine**
  - `services/personality_engine.py` defines multiple personalities (mentor, friend, therapist, professional, etc.) with:
    - Rich system prompts and tone guidelines
    - `ToneCharacteristics` for formality, empathy, directness, creativity, humor, supportiveness
  - `/api/generate` uses this engine plus `UserMemory` to produce personality-specific responses.

- **Before/after personality differences**
  - `/api/compare` + `PersonalityComparison` generate:
    - A neutral base response
    - Multiple personality-specific variants
    - Comparison analysis and recommendations
  - The UI section **"4. Compare Personalities"** visualizes these differences side-by-side.

- **Reasoning & prompt design**
  - Memory extractor and each personality use detailed, documented prompts aimed at high-quality reasoning.

- **Structured output parsing**
  - LLM responses are parsed as JSON into Pydantic models, with validation and error handling.

- **Working with user memory**
  - Extracted memory is stored in a database and reused for subsequent responses and comparisons.

- **Modular system design**
  - Clear separation between models, services, API layer, and UI, following a production-style layout.

## Dependencies

- FastAPI: Web framework for building APIs.
- Uvicorn: ASGI server for running FastAPI.
- Pydantic: Data validation and settings management.
- Python-dotenv: Load environment variables from .env file.

## License

This project is open-source and available under the MIT License.
