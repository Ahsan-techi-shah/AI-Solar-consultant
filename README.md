# SolarAI Multi-Agent Consultant MVP

## Flow
Customer uploads electricity bills (PDF / JPG / PNG) → bill reader extracts monthly kWh → deterministic Python sizing → CrewAI agents (bill auditor, PV sizing reviewer, solar consultant) → approved company pricing → quotation.

## How bills are read
- Text PDFs: text extraction with pdfplumber.
- Photos (JPG/PNG) and scanned PDFs: Groq vision model reads the units and billing month.
- Vision readings are AI estimates. Verify them against the bills before quoting.

## Models (Groq)
- Agents / summary: `openai/gpt-oss-120b` (text only), run through CrewAI.
- Bill images: `meta-llama/llama-4-scout-17b-16e-instruct`.
- Override with the secrets/env vars `GROQ_TEXT_MODEL` and `GROQ_VISION_MODEL` if Groq renames or retires a model.
- Keep deterministic electrical calculations in Python rather than asking the LLM to calculate numbers.

## Deploy on Streamlit Cloud
1. Create a GitHub repository and upload the project files/folders exactly as they are.
2. In Streamlit Community Cloud, create an app and select `app.py`.
3. In Advanced settings, choose Python 3.11 or 3.12.
4. Open App Settings → Secrets and add:

GROQ_API_KEY = "your_key"

Never commit the real API key to GitHub.

## Planned expansion
- Consumption validation by billing month (verified 12-month table)
- Product selection agent
- WhatsApp/email pricing communication agent
- Quotation PDF agent
- Human approval/audit step

Company prices should come from an authorized company contact/database. The app should not invent today's prices.
