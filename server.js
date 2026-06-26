import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import { toFile } from "openai/uploads";
import { createClient } from "@supabase/supabase-js";

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });

app.use(cors({ origin: process.env.CORS_ORIGIN || "*", credentials: false }));
app.use(express.json({ limit: "2mb" }));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function getUser(req) {
  const auth = req.headers.authorization || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";

  if (!token) {
    const err = new Error("Missing login token. Please log in again.");
    err.status = 401;
    throw err;
  }

  const { data, error } = await supabase.auth.getUser(token);

  if (error || !data?.user?.id) {
    const err = new Error("Invalid or expired login token. Please log in again.");
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

async function saveDeckAndCards({ userId, deckName, cards }) {
  const cleanDeckName = String(deckName || "Untitled folder").trim() || "Untitled folder";

  const { data: deck, error: deckError } = await supabase
    .from("decks")
    .insert({
      user_id: userId,
      name: cleanDeckName
    })
    .select()
    .single();

  if (deckError) throw new Error(`Supabase deck save failed: ${deckError.message}`);

  const cardRows = cards.map((card, index) => ({
    user_id: userId,
    deck_id: deck.id,
    english: card.english || "",
    japanese: card.japanese || "",
    kana: card.kana || "",
    romaji: card.romaji || "",
    difficulty: Number(card.difficulty || 2),
    words: card.words || [],
    position: index
  }));

  const { data: savedCards, error: cardsError } = await supabase
    .from("cards")
    .insert(cardRows)
    .select();

  if (cardsError) throw new Error(`Supabase cards save failed: ${cardsError.message}`);

  return { deck, cards: savedCards || [] };
}

app.get("/health", (req, res) => {
  res.json({
    ok: true,
    version: "0.2.5-auth-user",
    supabase: Boolean(process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY)
  });
});

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
    const { deckName, sentences } = req.body || {};
    const cleanSentences = cleanSentenceList(sentences);

    if (!cleanSentences.length) {
      return res.status(400).json({ error: "No sentences supplied" });
    }

    const generatedCards = await generateCards(cleanSentences);
    const saved = await saveDeckAndCards({
      userId: user.id,
      deckName,
      cards: generatedCards
    });

    res.json({
      ok: true,
      deckId: saved.deck.id,
      deckName: saved.deck.name,
      count: saved.cards.length,
      cards: saved.cards
    });
  } catch (err) {
    next(err);
  }
});

app.get("/decks", async (req, res, next) => {
  try {
    const user = await getUser(req);

    const { data, error } = await supabase
      .from("decks")
      .select("*, cards(*)")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false });

    if (error) throw new Error(`Supabase deck load failed: ${error.message}`);

    res.json({ ok: true, decks: data || [] });
  } catch (err) {
    next(err);
  }
});

app.use((err, req, res, next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Server error" });
});

const port = Number(process.env.PORT || 8787);
app.listen(port, () => {
  console.log(`Japanese Study backend v0.2.5 auth-user running on port ${port}`);
});
