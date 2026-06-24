# Japanese Study Backend v0.2

Adds real audio transcription endpoint.

Endpoints:
- `POST /transcribe-audio` receives one short audio file and returns English text.
- `POST /process-sentences` receives text sentences and creates Japanese study cards.

Environment:
- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `CORS_ORIGIN`

Do not put the OpenAI API key or Supabase service role key in the phone app.
