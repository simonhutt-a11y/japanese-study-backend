import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import { toFile } from "openai/uploads";

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });

app.use(cors({ origin: process.env.CORS_ORIGIN || "*", credentials: false }));
app.use(express.json({ limit: "2mb" }));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

async function getUser(req) {
  return { id: "test-user" };
}
  const { data, error } = await supabase.auth.getUser(token);
  if (error || !data?.user) {
    const err = new Error("Invalid login token");
    err.status = 401;
    throw err;
  }
  return data.user;
}

function cleanSentenceList(sentences) {
  if (!Array.isArray(sentences)) return [];
  return sentences.map(s => String(s || "").trim()).filter(Boolean).slice(0, 50);
}

function safeJsonParse(text) {
  const cleaned = String(text || "").trim()
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/```$/i, "")
    .trim();
  return JSON.parse(cleaned);
}

async function generateCards(sentences) {
  const response = await openai.responses.create({
    model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
    instructions:
      "You create Japanese study cards for an English-speaking learner. Return only valid JSON. " +
      "Keep translations natural and useful for travel/conversation. Kana must be Japanese script. " +
      "Romaji must be readable Hepburn-style romaji. Difficulty must be 1, 2, or 3.",
    input:
      "Create study cards for these English sentences:\n\n" +
      JSON.stringify(sentences, null, 2) +
      '\n\nReturn JSON shaped as: {"cards":[{"english":"","japanese":"","kana":"","romaji":"","difficulty":2,"words":[{"jp":"","romaji":"","meaning":""}]}]}'
  });
  const parsed = safeJsonParse(response.output_text);
  if (!Array.isArray(parsed.cards)) throw new Error("AI returned invalid cards JSON");
  return parsed.cards;
}

app.get("/health", (req, res) => res.json({ ok: true, version: "0.2.0" }));

app.post("/transcribe-audio", upload.single("audio"), async (req, res, next) => {
  try {
    await getUser(req);
    if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });

    const file = await toFile(req.file.buffer, req.file.originalname || "sentence.webm", {
      type: req.file.mimetype || "audio/webm"
    });

    const transcription = await openai.audio.transcriptions.create({
      file,
      model: process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe",
      language: "en"
    });

    res.json({ ok: true, text: transcription.text || "" });
  } catch (err) {
    next(err);
  }
});

app.post("/process-sentences", async (req, res, next) => {
  try {
    const user = await getUser(req);
    const { deckId, deckName, sentences } = req.body || {};
    const cleanSentences = cleanSentenceList(sentences);
    if (!cleanSentences.length) return res.status(400).json({ error: "No sentences supplied" });

    let finalDeckId = deckId || null;
    if (!finalDeckId) {
      const { data: deck, error } = await supabase
        .from("decks")
        .insert({ user_id: user.id, name: String(deckName || "Recorded sentences").trim() || "Recorded sentences" })
        .select()
        .single();
      if (error) throw error;
      finalDeckId = deck.id;
    }

    const cards = await generateCards(cleanSentences);
    const rows = cards.map((card, idx) => ({
      user_id: user.id,
      deck_id: finalDeckId,
      status: "complete",
      english: String(card.english || cleanSentences[idx] || "").trim(),
      japanese: String(card.japanese || "").trim(),
      kana: String(card.kana || "").trim(),
      romaji: String(card.romaji || "").trim(),
      difficulty: [1,2,3].includes(Number(card.difficulty)) ? Number(card.difficulty) : 2,
      words: Array.isArray(card.words) ? card.words : []
    }));

    const { data: savedCards, error } = await supabase.from("cards").insert(rows).select();
    if (error) throw error;
    res.json({ ok: true, deckId: finalDeckId, count: savedCards.length, cards: savedCards });
  } catch (err) {
    next(err);
  }
});

app.use((err, req, res, next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server error" });
});

const port = Number(process.env.PORT || 8787);
app.listen(port, () => console.log(`Japanese Study backend v0.2 running on port ${port}`));
