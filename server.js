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
  it: { code: "it", name: "Italian", transcriptionHint: "it" },
  da: { code: "da", name: "Danish", transcriptionHint: "da" },
  pt: { code: "pt", name: "Portuguese", transcriptionHint: "pt" }
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
  if (raw.startsWith("da")) return "da";
  if (raw.startsWith("pt")) return "pt";

  return raw.slice(0, 2);
}

function languageName(code) {
  const clean = normalizeLanguageCode(code);
  return SUPPORTED_LANGUAGES[clean]?.name || clean || "Unknown";
}

// 12.0 (Simon: "English to german has DBs in japanese" / "same with italian"): the
// process-sentences card generators below were hardcoded to Japanese with no language
// parameter at all, so every non-Japanese decks def-boxes were silently generated in
// Japanese regardless of the language actually selected in the app. Unknown/missing codes
// fall back to "ja" (the original behavior) for zero regression risk to the already-
// correct, most-used language.
function resolveCardsLanguage(targetLanguage) {
  const normalized = normalizeLanguageCode(targetLanguage);
  return SUPPORTED_LANGUAGES[normalized] ? normalized : "ja";
}
function buildNonJapaneseFastInstructions(langLabel) {
  return "Return only valid JSON. Create fast " + langLabel + " travel/conversation study cards for an English-speaking learner. " +
    "Prioritise speed. No explanations. No markdown. " +
    "The \"japanese\" field must hold the natural " + langLabel + " translation (despite its name, this field always holds whichever language is being studied). The \"romaji\" field must repeat the \"japanese\" field text EXACTLY, character-for-character - no pronunciation stress marks, syllable breaks, phonetic respelling, or capitalization changes, since " + langLabel + " already uses the Latin alphabet. Difficulty 1-3. " +
    "Leave words as an empty array.";
}
function buildNonJapaneseFullInstructions(langLabel) {
  return "You create " + langLabel + " study cards for an English-speaking learner. Return only valid JSON. " +
    "Keep translations natural and useful for travel/conversation. The \"japanese\" field must contain the natural " + langLabel + " translation (despite its name, this field always holds whichever language is being studied, not literally Japanese). The \"kana\" field should repeat that same " + langLabel + " text exactly - no script conversion is needed for this language. The \"romaji\" field must repeat the \"japanese\" field text EXACTLY, character-for-character - no pronunciation stress marks, syllable breaks, phonetic respelling, or capitalization changes, since " + langLabel + " already uses the Latin alphabet. Difficulty must be 1, 2, or 3. Use the vocabulary a native speaker would naturally use in the situation the sentence implies. The words array must be a word-by-word (or short natural phrase) breakdown of the EXACT " + langLabel + " sentence returned: cover it completely and in order, with no missing or extra words, and never words from a different translation of the same english sentence. Every words entry must include jp (the " + langLabel + " word or phrase exactly as written in the sentence - despite the field name, not literally Japanese), kana (repeat jp - no separate reading needed for this language), romaji (repeat jp EXACTLY, character-for-character - no stress marks, syllable breaks, or capitalization changes), and a short english meaning. The card kana field must exactly match the japanese field for this language.";
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

const WHISPER_HALLUCINATION_PHRASES = [
  "thanks for watching", "thank you for watching", "thanks for listening",
  "thank you for listening", "please subscribe", "don't forget to subscribe",
  "like and subscribe", "hit the subscribe button", "see you next time",
  "see you in the next video", "check out my channel", "bye bye", "bye-bye",
  "goodbye everyone", "thank you for watching this video",
  "thanks for watching this video"
  ];
function looksLikeWhisperHallucination(text, durationSeconds) {
  const normalized = text.toLowerCase().replace(/[.!?,]/g, "").trim();
  if (!normalized) return false;
  const isShortClip = durationSeconds === 0 || durationSeconds < 4;
  if (!isShortClip) return false;
  return WHISPER_HALLUCINATION_PHRASES.some(phrase => normalized === phrase || normalized.includes(phrase));
}

async function transcribeAudio(fileBuffer, originalname, mimetype, options = {}) {
  const file = await toFile(fileBuffer, originalname || "sentence.webm", {
    type: mimetype || "audio/webm"
  });

  const transcribeModel = process.env.OPENAI_TRANSCRIBE_MODEL || "whisper-1";
  const canVerify = transcribeModel === "whisper-1";
  const request = {
    file,
    model: transcribeModel
  };
  if (canVerify) request.response_format = "verbose_json";

  const language = normalizeLanguageCode(options.language);
  if (language) request.language = language;
  if (options.prompt) request.prompt = options.prompt;

  const transcription = await openai.audio.transcriptions.create(request);
  const text = String(transcription?.text || "").trim();
  const duration = canVerify ? (Number(transcription?.duration) || 0) : 0;

  if (canVerify) {
    console.log("Whisper transcription", { duration, textLength: text.length, text: text.slice(0, 80) });
  }

  if (canVerify && transcription && typeof transcription === "object") {
    const segments = Array.isArray(transcription.segments) ? transcription.segments : [];
    const impliedTooLong = duration > 0 && duration < 1.5 && text.length > 25;
    const lowConfidenceSpeech = segments.length > 0 && segments.every(seg =>
      typeof seg?.no_speech_prob === "number" && seg.no_speech_prob > 0.6 &&
      typeof seg?.avg_logprob === "number" && seg.avg_logprob < -0.5
      );
    const stockPhraseHallucination = looksLikeWhisperHallucination(text, duration);
    if (impliedTooLong || lowConfidenceSpeech || stockPhraseHallucination) {
      console.warn("Whisper hallucination guard rejected a transcript", {
        duration, textLength: text.length, impliedTooLong, lowConfidenceSpeech, stockPhraseHallucination
      });
      return "";
    }
  }

  return text;
}

const CJK_RE = /[぀-ヿ㐀-鿿가-힯]/;

function scriptLooksWrong(text, langCode) {
    const clean = String(text || "").trim();
    if (!clean) return false; // emptiness is its own, separate validation failure
    const code = normalizeLanguageCode(langCode);
    const isCjkLanguage = code === "ja" || code === "zh" || code === "ko";
    const hasCjk = CJK_RE.test(clean);
    if (isCjkLanguage && !hasCjk) return true; // claims a CJK language but has no CJK characters
    if (!isCjkLanguage && hasCjk) return true; // claims a Latin-alphabet language but has CJK characters
    return false;
}

async function generateWithValidation(label, generateFn, validateFn, { maxAttempts = 3 } = {}) {
    let lastReason = "";
    let lastResult = null;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
          let result;
          try {
                  result = await generateFn();
          } catch (e) {
                  lastReason = `attempt threw: ${(e && e.message) || e}`;
                  console.warn(`${label}: attempt ${attempt}/${maxAttempts} threw`, lastReason);
                  continue;
          }

          const verdict = validateFn(result);
          if (verdict === true) {
                  if (attempt > 1) {
                            console.log(`${label}: succeeded on attempt ${attempt}/${maxAttempts} after earlier failure: ${lastReason}`);
                  }
                  return result;
          }

          lastReason = String(verdict || "validation failed");
          lastResult = result;
          console.warn(`${label}: attempt ${attempt}/${maxAttempts} failed validation: ${lastReason}`);
    }

    console.error(`${label}: all ${maxAttempts} attempts failed validation, last reason: ${lastReason}`);
    const err = new Error("Translation could not be completed accurately - please try again.");
    err.status = 502;
    err.lastReason = lastReason;
    err.lastResult = lastResult;
    throw err;
}

async function generateFastCards(sentences, targetLanguage) {
    const fastLangCode = resolveCardsLanguage(targetLanguage);
    const fastLangLabel = languageName(fastLangCode);

    async function attempt() {
          const response = await openai.responses.create({
                      model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-mini",
                  instructions: fastLangCode === "ja" ? (
                            "Return only valid JSON. Create fast Japanese travel/conversation study cards. " +
                            "Prioritise speed. No explanations. No markdown. " +
                            "Return natural Japanese, readable Hepburn romaji, and difficulty 1-3. Prefer the vocabulary a native speaker would use in the implied situation; avoid katakana loanwords when a natural native word exists. " +
                            "Leave words as an empty array."
                          ) : buildNonJapaneseFastInstructions(fastLangLabel),
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

    function validate(cards) {
          if (!Array.isArray(cards) || cards.length !== sentences.length) return "card count did not match sentence count";
          for (const card of cards) {
                  if (!String(card.english || "").trim()) return "a card had an empty english field";
                  if (!String(card.japanese || "").trim()) return "a card had an empty japanese field";
                  if (!String(card.romaji || "").trim()) return "a card had an empty romaji field";
                  if (scriptLooksWrong(card.japanese, fastLangCode)) return "a card's japanese field script did not match the target language";
          }
          return true;
    }

    return generateWithValidation("generateFastCards", attempt, validate);
}

async function generateInstantTranslation(english) {
    async function attempt() {
          const response = await openai.responses.create({
                      model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-mini",
                  instructions:
                            "Return only valid JSON. Translate the English sentence into natural Japanese for immediate spoken use. " +
                            "Return only japanese and romaji. No explanation. No markdown. Prefer the vocabulary a native speaker would use in the implied situation; avoid katakana loanwords when a natural native word exists (a shopping bag is 袋, not バッグ).",
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

    function validate(result) {
          if (!String(result.japanese || "").trim()) return "empty japanese field";
          if (!String(result.romaji || "").trim()) return "empty romaji field";
          if (scriptLooksWrong(result.japanese, "ja")) return "japanese field did not contain Japanese script";
          return true;
    }

    return generateWithValidation("generateInstantTranslation", attempt, validate);
}

async function generateFullCards(sentences, targetLanguage) {
    const fullLangCode = resolveCardsLanguage(targetLanguage);
    const fullLangLabel = languageName(fullLangCode);

    async function attempt() {
          const response = await openai.responses.create({
                  model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
                  instructions: fullLangCode === "ja" ? (
                            "You create Japanese study cards for an English-speaking learner. Return only valid JSON. " +
                            "Keep translations natural and useful for travel/conversation. Kana must be Japanese script. " +
                            "Romaji must be readable Hepburn-style romaji. Difficulty must be 1, 2, or 3. Use the vocabulary a native Japanese speaker would naturally use in the situation the sentence implies, and prefer native words over katakana loanwords when both exist (for a shopping bag say 袋 fukuro, not バッグ). The words array must be a word-by-word breakdown of the EXACT japanese sentence returned: cover it completely and in order, with no missing or extra words, and never words from a different translation of the same english. Every words entry must include jp exactly as written in the sentence, kana giving that word's reading in hiragana exactly as pronounced in this sentence, readable Hepburn romaji, and a short english meaning. The card kana field must be the exact hiragana reading of the japanese field."
                          ) : buildNonJapaneseFullInstructions(fullLangLabel),
                  input:
                            "Create study cards for these English sentences:\n\n" +
                            JSON.stringify(sentences, null, 2) +
                            '\n\nReturn JSON shaped as: {"cards":[{"english":"","japanese":"","kana":"","romaji":"","difficulty":2,"words":[{"jp":"","kana":"","romaji":"","meaning":""}]}]}'
          });

          const parsed = safeJsonParse(response.output_text);
          if (!Array.isArray(parsed.cards)) throw new Error("AI returned invalid cards JSON");
          return parsed.cards;
    }

    function validate(cards) {
          if (!Array.isArray(cards) || cards.length !== sentences.length) return "card count did not match sentence count";
          for (const card of cards) {
                  if (!String(card.english || "").trim()) return "a card had an empty english field";
                  if (!String(card.japanese || "").trim()) return "a card had an empty japanese field";
                  if (!String(card.kana || "").trim()) return "a card had an empty kana field";
                  if (!String(card.romaji || "").trim()) return "a card had an empty romaji field";
                  if (scriptLooksWrong(card.japanese, fullLangCode)) return "a card's japanese field script did not match the target language";
                  if (!Array.isArray(card.words) || card.words.length === 0) return "a card had an empty words breakdown";
                  for (const w of card.words) {
                            if (!String(w?.jp || "").trim() || !String(w?.meaning || "").trim()) return "a card's words array had an incomplete entry";
                  }
          }
          return true;
    }

    return generateWithValidation("generateFullCards", attempt, validate);
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

    async function attempt() {
          const response = await openai.responses.create({
                      model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-mini",
                  // Simon (2026-07-08): "make it faster" - conversation-translate-text was measured at
                  // ~3.6s end-to-end in production, almost entirely model "thinking" time rather than
                  // network latency. This is a straight translation task with no multi-step reasoning
                  // to do, so minimal reasoning effort should cut latency without changing the actual
                  // translation quality. Reversible: delete this line to go back to the model's default
                  // effort if quality/latency doesn't improve as hoped.
              reasoning: { effort: "none" },
                  instructions:
                            "Return only valid JSON. You are the WordHole multilingual conversation translator. " +
                            "Translate natural spoken conversation without softening, censoring, moralising, or over-explaining. " +
                            "Preserve the speaker's meaning, tone, directness, slang, profanity, and casual style when possible. " +
                            "A supplied source language and target language, when given, are authoritative - trust them exactly as given and do not silently re-detect or substitute a different language pair. Only detect the source language yourself if none was supplied. Always produce a genuine translation: the translation field must be a real translation of the transcript into the target language, in the target language’s own script/words, and must not simply repeat the transcript text back unmodified, unless source and target are genuinely the same language. " +
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

    function validate(result) {
          const translation = String(result.translation || "").trim();
          if (!translation) return "empty translation field";
          if (result.targetLanguage && scriptLooksWrong(translation, result.targetLanguage)) {
                  return "translation field script did not match the target language";
          }
          if (
                  result.sourceLanguage && result.targetLanguage &&
                  result.sourceLanguage !== result.targetLanguage && cleanTranscript &&
                  translation.toLowerCase() === cleanTranscript.trim().toLowerCase()
                ) {
                  return "translation just echoed the transcript unchanged";
          }
          if ((result.targetLanguage === "ja" || result.sourceLanguage === "ja") && !String(result.romaji || "").trim()) {
                  return "missing romaji when Japanese is involved";
          }
          return true;
    }

    return generateWithValidation("generateConversationTurn", attempt, validate);
}

async function detectLanguageFromText({
  text,
  primaryLanguage,
  partnerLanguage
}) {
  const cleanText = String(text || "").trim();

  const response = await openai.responses.create({
        model: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-mini",
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
        fastModel: process.env.OPENAI_FAST_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-mini",
        transcribeModel: process.env.OPENAI_TRANSCRIBE_MODEL || "whisper-1",
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
          language: suppliedSourceLanguage, prompt: suppliedSourceLanguage === "ja" ? "袋はありますか。コーヒーをお願いします。駅はどこですか。すみません、これをください。トイレはどこですか。ありがとうございます。" : undefined
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

console.log("Conversation turn result", {
  requestedSourceLanguage: suppliedSourceLanguage || "(auto-detect)",
  requestedTargetLanguage: suppliedTargetLanguage || "(auto)",
  detectedSourceLanguage: sourceLanguage,
  transcript,
  translatedTranscript: translated.transcript,
  translation: translated.translation,
  translatedSourceLanguage: translated.sourceLanguage,
  translatedTargetLanguage: translated.targetLanguage,
  romaji: translated.romaji
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
    const { deckName, sentences, fast, mode, skipBreakdown, targetLanguage } = req.body || {};
    const cleanSentences = cleanSentenceList(sentences);

    if (!cleanSentences.length) {
      return res.status(400).json({ error: "No sentences supplied" });
    }

    const useFast = fast === true || mode === "fast" || skipBreakdown === true;
    const generatedCards = useFast
      ? await generateFastCards(cleanSentences, targetLanguage)
      : await generateFullCards(cleanSentences, targetLanguage);

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
