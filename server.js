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

const BACKEND_VERSION = "0.3.0-conversation-turn";

const SUPPORTED_LANGUAGES = {
  en: { code: "en", name: "English", transcriptionHint: "en" },
  ja: { code: "ja", name: "Japanese", transcriptionHint: "ja" },
  ko: { code: "ko", name: "Korean", transcriptionHint: "ko" },
  zh: { code: "zh", name: "Chinese", transcriptionHint: "zh" },
  th: { code: "th", name: "Thai", transcriptionHint: "th" },
  vi: { code: "vi", name: "Vietnamese", transcriptionHint: "vi" },
  es: { code: "es", name: "Spanish", transcriptionHint: "es" },
  fr: { code: "fr", name: "French", transcriptionHint: "fr" },
  de: { code: "de", name: "German", transcriptionHint: "de" },
  it: { code: "it", name: "Italian", transcriptionHint: "it" }
};

const SUPPORTED_LANGUAGE_LIST = Object.values(SUPPORTED_LANGUAGES);

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

function normalizeLanguageCode(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return "";

  if (raw.startsWith("en")) return "en";
  if (raw.startsWith("ja") || raw === "jp") return "ja";
  if (raw.startsWith("ko") || raw === "kr") return "ko";
  if (raw.startsWith("zh") || raw.startsWith("cmn") || raw === "cn") return "zh";
  if (raw.startsWith("th")) return "th";
  if (raw.startsWith("vi")) return "vi";
  if (raw.startsWith("es")) return "es";
  if (raw.startsWith("fr")) return "fr";
  if (raw.startsWith("de")) return "de";
  if (raw.startsWith("it")) return "it";

  return raw.slice(0, 2);
}

function languageName(code) {
  const clean = normalizeLanguageCode(code);
  return SUPPORTED_LANGUAGES[clean]?.name || clean || "Unknown";
}

function resolveConversationTargetLanguage({ sourceLanguage, primaryLanguage, partnerLanguage, targetLanguage }) {
  const source = normalizeLanguageCode(sourceLanguage);
  const primary = normalizeLanguageCode(primaryLanguage) || "en";
  const partner = normalizeLanguageCode(partnerLanguage) || "ja";
  const requestedTarget = normalizeLanguageCode(targetLanguage);

  if (requestedTarget && requestedTarget !== source) return requestedTarget;

  if (source === primary) return partner;
  if (source === partner) return primary;

  if (source === "en") return partner || "ja";
  return primary || "en";
}

async function transcribeAudio(fileBuffer, originalname, mimetype, options = {}) {
  const file = await toFile(fileBuffer, originalname || "sentence.webm", {
    type: mimetype || "audio/webm"
  });

  const request = {
    file,
    model: process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe"
  };

  const language = normalizeLanguageCode(options.language);
  if (language) request.language = language;

  const transcription = await openai.audio.transcriptions.create(request);

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
}async function generateFullCards(sentences) {
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

async function generateConversationTurn({
  transcript,
  sourceLanguage,
  targetLanguage,
  primaryLanguage,
  partnerLanguage
}) {
  const cleanTranscript = String(transcript || "").trim();
  const cleanSourceLanguage = normalizeLanguageCode(sourceLanguage);
  const cleanTargetLanguage = resolveConversationTargetLanguage({
    sourceLanguage: cleanSourceLanguage,
    primaryLanguage,
    partnerLanguage,
    targetLanguage
  });

  const response = await openai.responses.create({
    model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-4.1-mini",
    instructions:
      "Return only valid JSON. You are the WordHole multilingual conversation translator. " +
      "Translate natural spoken conversation without softening, censoring, moralising, or over-explaining. " +
      "Preserve the speaker's meaning, tone, directness, slang, profanity, and casual style when possible. " +
      "Detect the source language if the supplied source language is uncertain. " +
      "For Japanese output, return natural Japanese plus readable Hepburn romaji. " +
      "For Japanese input, return readable Hepburn romaji for the source transcript. " +
      "For non-Japanese languages, romaji may be an empty string. " +
      "Return no markdown and no explanations.",
    input: JSON.stringify({
      transcript: cleanTranscript,
      suppliedSourceLanguage: cleanSourceLanguage || "",
      sourceLanguageName: languageName(cleanSourceLanguage),
      targetLanguage: cleanTargetLanguage,
      targetLanguageName: languageName(cleanTargetLanguage),
      primaryLanguage: normalizeLanguageCode(primaryLanguage) || "en",
      partnerLanguage: normalizeLanguageCode(partnerLanguage) || "ja",
      supportedLanguages: SUPPORTED_LANGUAGE_LIST.map(lang => ({
        code: lang.code,
        name: lang.name
      })),
      output_shape: {
        transcript: "",
        translation: "",
        sourceLanguage: "",
        targetLanguage: "",
        romaji: "",
        confidence: 0.8
      }
    })
  });

  const parsed = safeJsonParse(response.output_text);

  const finalSourceLanguage = normalizeLanguageCode(parsed.sourceLanguage) || cleanSourceLanguage || "";
  const finalTargetLanguage =
    normalizeLanguageCode(parsed.targetLanguage) ||
    resolveConversationTargetLanguage({
      sourceLanguage: finalSourceLanguage,
      primaryLanguage,
      partnerLanguage,
      targetLanguage: cleanTargetLanguage
    });

  return {
    transcript: parsed.transcript || cleanTranscript,
    translation: parsed.translation || "",
    sourceLanguage: finalSourceLanguage,
    sourceLanguageName: languageName(finalSourceLanguage),
    targetLanguage: finalTargetLanguage,
    targetLanguageName: languageName(finalTargetLanguage),
    romaji: parsed.romaji || "",
    confidence: Number(parsed.confidence || 0.8)
  };
}

async function detectLanguageFromText({
  text,
  primaryLanguage,
  partnerLanguage
}) {
  const cleanText = String(text || "").trim();

  const response = await openai.responses.create({
    model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-4.1-mini",
    instructions:
      "Return only valid JSON. Detect the language of the supplied text. " +
      "Prefer one of the supported WordHole language codes. No markdown. No explanations.",
    input: JSON.stringify({
      text: cleanText,
      primaryLanguage: normalizeLanguageCode(primaryLanguage) || "en",
      partnerLanguage: normalizeLanguageCode(partnerLanguage) || "ja",
      supportedLanguages: SUPPORTED_LANGUAGE_LIST.map(lang => ({
        code: lang.code,
        name: lang.name
      })),
      output_shape: {
        sourceLanguage: "",
        confidence: 0.8
      }
    })
  });

  const parsed = safeJsonParse(response.output_text);

  return {
    sourceLanguage: normalizeLanguageCode(parsed.sourceLanguage) || "",
    confidence: Number(parsed.confidence || 0.8)
  };
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
    version: BACKEND_VERSION,
    supabase: Boolean(process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY),
    fastModel: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-4.1-mini",
    transcribeModel: process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe",
    supportedLanguages: SUPPORTED_LANGUAGE_LIST.map(lang => ({
      code: lang.code,
      name: lang.name
    }))
  });
});

app.post("/transcribe-audio", upload.single("audio"), async (req, res, next) => {
  const started = Date.now();

  try {
    await getUser(req);
    if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });

    const language = normalizeLanguageCode(req.body?.language);
    const text = await transcribeAudio(
      req.file.buffer,
      req.file.originalname,
      req.file.mimetype,
      { language }
    );

    res.json({
      ok: true,
      text,
      language: language || "",
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
});app.post("/conversation-turn", upload.single("audio"), async (req, res, next) => {
  const started = Date.now();

  try {
    await getUser(req);

    const primaryLanguage = normalizeLanguageCode(req.body?.primaryLanguage) || "en";
    const partnerLanguage = normalizeLanguageCode(req.body?.partnerLanguage) || "ja";
    const suppliedSourceLanguage = normalizeLanguageCode(req.body?.sourceLanguage);
    const suppliedTargetLanguage = normalizeLanguageCode(req.body?.targetLanguage);

    let transcript = String(req.body?.text || req.body?.transcript || "").trim();
    let transcribeMs = 0;

    if (!transcript && req.file) {
      const transcribeStarted = Date.now();
      transcript = await transcribeAudio(
        req.file.buffer,
        req.file.originalname,
        req.file.mimetype,
        {
          language: suppliedSourceLanguage
        }
      );
      transcribeMs = Date.now() - transcribeStarted;
    }

    if (!transcript) {
      return res.status(400).json({
        ok: false,
        error: "No speech or text supplied",
        timings: {
          totalMs: Date.now() - started
        }
      });
    }

    let sourceLanguage = suppliedSourceLanguage;

    if (!sourceLanguage) {
      const detected = await detectLanguageFromText({
        text: transcript,
        primaryLanguage,
        partnerLanguage
      });
      sourceLanguage = detected.sourceLanguage;
    }

    const translated = await generateConversationTurn({
      transcript,
      sourceLanguage,
      targetLanguage: suppliedTargetLanguage,
      primaryLanguage,
      partnerLanguage
    });

    res.json({
      ok: true,
      transcript: translated.transcript,
      translation: translated.translation,
      sourceLanguage: translated.sourceLanguage,
      sourceLanguageName: translated.sourceLanguageName,
      targetLanguage: translated.targetLanguage,
      targetLanguageName: translated.targetLanguageName,
      romaji: translated.romaji,
      confidence: translated.confidence,
      primaryLanguage,
      partnerLanguage,
      timings: {
        transcribeMs,
        totalMs: Date.now() - started
      }
    });
  } catch (err) {
    next(err);
  }
});

app.post("/conversation-translate-text", async (req, res, next) => {
  const started = Date.now();

  try {
    await getUser(req);

    const text = String(req.body?.text || req.body?.transcript || "").trim();

    if (!text) {
      return res.status(400).json({ error: "No text supplied" });
    }

    const primaryLanguage = normalizeLanguageCode(req.body?.primaryLanguage) || "en";
    const partnerLanguage = normalizeLanguageCode(req.body?.partnerLanguage) || "ja";
    let sourceLanguage = normalizeLanguageCode(req.body?.sourceLanguage);
    const targetLanguage = normalizeLanguageCode(req.body?.targetLanguage);

    if (!sourceLanguage) {
      const detected = await detectLanguageFromText({
        text,
        primaryLanguage,
        partnerLanguage
      });
      sourceLanguage = detected.sourceLanguage;
    }

    const translated = await generateConversationTurn({
      transcript: text,
      sourceLanguage,
      targetLanguage,
      primaryLanguage,
      partnerLanguage
    });

    res.json({
      ok: true,
      transcript: translated.transcript,
      translation: translated.translation,
      sourceLanguage: translated.sourceLanguage,
      sourceLanguageName: translated.sourceLanguageName,
      targetLanguage: translated.targetLanguage,
      targetLanguageName: translated.targetLanguageName,
      romaji: translated.romaji,
      confidence: translated.confidence,
      primaryLanguage,
      partnerLanguage,
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

    const english = await transcribeAudio(
      req.file.buffer,
      req.file.originalname,
      req.file.mimetype,
      { language: "en" }
    );

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
  console.log(`Japanese Study backend ${BACKEND_VERSION} running on port ${port}`);
});
