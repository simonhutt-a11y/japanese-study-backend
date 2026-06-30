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

async function transcribeAudio(fileBuffer, originalname, mimetype) {
  const file = await toFile(fileBuffer, originalname || "sentence.webm", {
    type: mimetype || "audio/webm"
  });

  const transcription = await openai.audio.transcriptions.create({
    file,
    model: process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe",
    language: "en"
  });

  return String(transcription.text || "").trim();
}

async function generateFastCards(sentences) {
  const response = await openai.responses.create({
    model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-4.1-mini",
    instructions:
      "Return only valid JSON. Create fast Japanese travel/conversation study cards. " +
      "Prioritise speed. No explanations. No markdown. " +
      "Return natural Japanese, readable Hepburn romaji, and difficulty 1-3. " +
      "Leave words as an empty array.",
    input:
      JSON.stringify({
        sentences,
        output_shape: {
          cards: [
            {
              english: "",
              japanese: "",
              kana: "",
              romaji: "",
              difficulty: 2,
              words: []
            }
          ]
        }
      })
  });

  const parsed = safeJsonParse(response.output_text);
  if (!Array.isArray(parsed.cards)) throw new Error("AI returned invalid fast cards JSON");

  return parsed.cards.map((card, index) => ({
    english: card.english || sentences[index] || "",
    japanese: card.japanese || "",
    kana: card.kana || card.japanese || "",
    romaji: card.romaji || "",
    difficulty: Number(card.difficulty || 2),
    words: []
  }));
}
async function generateInstantTranslation(english) {
  const response = await openai.responses.create({
    model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-4.1-mini",
    instructions:
      "Return only valid JSON. Translate the English sentence into natural Japanese for immediate spoken use. " +
      "Return only japanese and romaji. No explanation. No markdown.",
    input: JSON.stringify({
      english,
      output_shape: {
        japanese: "",
        romaji: ""
      }
    })
  });

  const parsed = safeJsonParse(response.output_text);

  return {
    japanese: parsed.japanese || "",
    romaji: parsed.romaji || ""
  };
}
async function generateFullCards(sentences) {
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
    words: Array.isArray(card.words) ? card.words : [],
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
    version: "0.2.9-instant-translate",
    supabase: Boolean(process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY),
    fastModel: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-4.1-mini",
    transcribeModel: process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe"
  });
});

app.post("/transcribe-audio", upload.single("audio"), async (req, res, next) => {
  const started = Date.now();

  try {
    await getUser(req);
    if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });

    const text = await transcribeAudio(req.file.buffer, req.file.originalname, req.file.mimetype);

    res.json({
      ok: true,
      text,
      timings: {
        totalMs: Date.now() - started
      }
    });
  } catch (err) {
    next(err);
  }
});
app.post("/translate-instant", async (req, res, next) => {
  const started = Date.now();

  try {
    await getUser(req);

    const english = String(req.body?.english || "").trim();

    if (!english) {
      return res.status(400).json({ error: "No English supplied" });
    }

    const translated = await generateInstantTranslation(english);

    res.json({
      ok: true,
      english,
      romaji: translated.romaji,
      japanese: translated.japanese,
      timings: {
        totalMs: Date.now() - started
      }
    });
  } catch (err) {
    next(err);
  }
});
app.post("/process-sentences", async (req, res, next) => {
  const started = Date.now();

  try {
    const user = await getUser(req);
    const { deckName, sentences, fast, mode, skipBreakdown } = req.body || {};
    const cleanSentences = cleanSentenceList(sentences);

    if (!cleanSentences.length) {
      return res.status(400).json({ error: "No sentences supplied" });
    }

    const useFast = fast === true || mode === "fast" || skipBreakdown === true;
    const generatedCards = useFast
      ? await generateFastCards(cleanSentences)
      : await generateFullCards(cleanSentences);

    const aiMs = Date.now() - started;

    const saved = await saveDeckAndCards({
      userId: user.id,
      deckName,
      cards: generatedCards
    });

    res.json({
      ok: true,
      fast: useFast,
      deckId: saved.deck.id,
      deckName: saved.deck.name,
      count: saved.cards.length,
      cards: saved.cards,
      timings: {
        aiMs,
        totalMs: Date.now() - started
      }
    });
  } catch (err) {
    next(err);
  }
});

app.post("/process-sentences-fast", async (req, res, next) => {
  req.body = { ...(req.body || {}), fast: true, skipBreakdown: true };
  app._router.handle(req, res, next);
});

app.post("/capture-audio-fast", upload.single("audio"), async (req, res, next) => {
  const started = Date.now();

  try {
    const user = await getUser(req);
    if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });

    const { deckName } = req.body || {};

    const english = await transcribeAudio(req.file.buffer, req.file.originalname, req.file.mimetype);
    const transcribeMs = Date.now() - started;

    if (!english) {
      return res.status(400).json({
        ok: false,
        error: "No speech detected",
        timings: {
          transcribeMs,
          totalMs: Date.now() - started
        }
      });
    }

    const generatedCards = await generateFastCards([english]);
    const aiMs = Date.now() - started - transcribeMs;

    const saved = await saveDeckAndCards({
      userId: user.id,
      deckName,
      cards: generatedCards
    });

    res.json({
      ok: true,
      fast: true,
      text: english,
      deckId: saved.deck.id,
      deckName: saved.deck.name,
      count: saved.cards.length,
      cards: saved.cards,
      timings: {
        transcribeMs,
        aiMs,
        totalMs: Date.now() - started
      }
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
  console.log(`Japanese Study backend v0.2.9 instant-translate running on port ${port}`);
});
