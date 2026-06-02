/**
 * ============================================================
 * MEDANIXX (MDX) — Preclinical Study App
 * server.js — Main Backend Server (Gemini Version)
 * Built for: UNN 100 Level Students
 * ============================================================
 */

"use strict";

const express        = require("express");
const multer         = require("multer");
const pdfParse       = require("pdf-parse");
const cors           = require("cors");
const dotenv         = require("dotenv");
const morgan         = require("morgan");
const helmet         = require("helmet");
const rateLimit      = require("express-rate-limit");
const path           = require("path");
const fs             = require("fs");
const { v4: uuidv4 } = require("uuid");

dotenv.config();

// ── Google Gemini AI Setup ─────────────────────────────────────
let geminiModel = null;
try {
  const { GoogleGenerativeAI } = require("@google/generative-ai");
  if (process.env.GEMINI_API_KEY) {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    geminiModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    console.log("✅ Gemini AI initialized — AI quiz generation enabled.");
  } else {
    console.log("⚠️  No GEMINI_API_KEY found in .env — using rule-based quiz generation.");
  }
} catch (err) {
  console.log("ℹ️  @google/generative-ai setup skipped or missing.");
}

const app  = express();
const PORT = process.env.PORT || 3000;

const store = {
  documents: {},  
  quizzes:   {},  
  sessions:  {},  
};

const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// ── Middleware ────────────────────────────────────────────────
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan("dev"));
app.use(express.static(path.join(__dirname, "public")));

const aiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 50,
  message: { error: "Too many requests. Please wait a moment." },
});

// ── Multer Config ─────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename:    (_req, file, cb) => {
    const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_");
    cb(null, `${uuidv4()}_${safe}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype !== "application/pdf") {
      return cb(new Error("Only PDF files are accepted."));
    }
    cb(null, true);
  },
});

// ── Helpers & Rule-Based Fallback Engine ───────────────────────
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function cleanText(text) {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[^\x20-\x7E\n]/g, " ")
    .trim();
}

function extractFactSentences(text) {
  return text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 60 && s.length < 400)
    .filter(s => /\b(is|are|was|were|causes|produces|defined|classified|function|process|reaction|equation|law|theory|principle|cell|molecule)\b/i.test(s));
}

function makeFillBlank(sentence) {
  const words = sentence.split(/\s+/).filter(w =>
    w.length > 5 && /^[A-Za-z]/.test(w) &&
    !/^(which|these|their|those|there|about|would|could|should|after|before|during|while|because|however|therefore)$/i.test(w)
  );
  if (!words.length) return null;
  const answer = words[Math.floor(words.length / 2)].replace(/[.,;:()]/g, "");
  return {
    id: uuidv4(), type: "fill_blank",
    question: `Complete: "${sentence.replace(answer, "___________")}"`,
    answer, explanation: `The correct word is "${answer}".`,
  };
}

function makeTrueFalse(sentence) {
  const swaps = [
    ["increases","decreases"],["produces","destroys"],["high","low"],
    ["positive","negative"],["directly","inversely"],["acid","base"]
  ];
  let statement = sentence;
  let isTrue = true;
  if (Math.random() > 0.5) {
    for (const [from, to] of swaps) {
      if (new RegExp(`\\b${from}\\b`, "i").test(statement)) {
        statement = statement.replace(new RegExp(`\\b${from}\\b`, "i"), to);
        isTrue = false;
        break;
      }
    }
  }
  return {
    id: uuidv4(), type: "true_false",
    question: `True or False: "${statement}"`,
    options: ["True", "False"],
    answer: isTrue ? "True" : "False",
    explanation: isTrue ? "This statement is correct." : `Altered statement.`,
  };
}

function generateRuleBasedQuiz(text, num = 10) {
  const facts = extractFactSentences(text);
  if (facts.length < 3) return [];
  const questions = [];
  for (const sentence of shuffle(facts)) {
    if (questions.length >= num) break;
    const q = Math.random() > 0.5 ? makeFillBlank(sentence) : makeTrueFalse(sentence);
    if (q) questions.push(q);
  }
  return questions;
}

// ── Gemini AI Logic ───────────────────────────────────────────
async function generateGeminiQuiz(text, numQuestions = 10, difficulty = "medium", course = "General") {
  if (!geminiModel) {
    return { questions: generateRuleBasedQuiz(text, numQuestions), method: "rule-based" };
  }
  const excerpt = cleanText(text).slice(0, 6000);
  const prompt = `You are an expert science educator creating a quiz for 100-level university students studying ${course}.
Generate exactly ${numQuestions} multiple-choice questions (MCQs) based on the content below.
Difficulty: ${difficulty}

Return ONLY a valid JSON object. No extra text, no markdown code fences.
Format:
{
  "questions": [
    {
      "id": "unique_string",
      "type": "mcq",
      "question": "Question text?",
      "options": ["A. Option one", "B. Option two", "C. Option three", "D. Option four"],
      "answer": "A. Option one",
      "explanation": "Explanation here."
    }
  ]
}
Content:
${excerpt}`;

  try {
    const result = await geminiModel.generateContent(prompt);
    const raw    = result.response.text();
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const parsed  = JSON.parse(cleaned);
    parsed.questions = parsed.questions.map(q => ({ ...q, id: q.id || uuidv4() }));
    return { questions: parsed.questions, method: "gemini-ai" };
  } catch (err) {
    return { questions: generateRuleBasedQuiz(text, numQuestions), method: "rule-based-fallback" };
  }
}

async function geminiChat(message, subject = "General") {
  if (!geminiModel) return "AI Tutor is not configured. Add GEMINI_API_KEY to .env.";
  const prompt = `You are MDX AI, an intelligent study assistant for 100-level university students in Nigeria studying ${subject}. Explain concepts clearly and concisely. Use bullet points for lists.\nStudent: ${message}`;
  try {
    const result = await geminiModel.generateContent(prompt);
    return result.response.text();
  } catch (err) {
    return "Sorry, I couldn't process that response. Please try again.";
  }
}

// ── API Routes ────────────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({ status: "ok", app: "Medanixx (MDX)", aiEnabled: !!geminiModel, documents: Object.keys(store.documents).length });
});

app.post("/api/pdf/upload", upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No PDF file received." });
    const buffer = fs.readFileSync(req.file.path);
    const parsed = await pdfParse(buffer);

    const doc = {
      id: uuidv4(),
      name: req.body.name || req.file.originalname,
      filename: req.file.filename,
      path: req.file.path,
      text: parsed.text,
      pages: parsed.numpages,
      sizeKB: (req.file.size / 1024).toFixed(1),
      wordCount: parsed.text.split(/\s+/).length,
      course: req.body.course || "General",
      uploadedAt: new Date().toISOString(),
    };
    store.documents[doc.id] = doc;
    res.status(201).json({ message: "PDF processed.", document: { id: doc.id, name: doc.name, pages: doc.pages, course: doc.course } });
  } catch (err) {
    res.status(500).json({ error: "Failed to process PDF.", detail: err.message });
  }
});

app.get("/api/pdf/list", (_req, res) => {
  res.json({ documents: Object.values(store.documents) });
});

app.post("/api/quiz/generate", aiLimiter, async (req, res) => {
  try {
    const { docId, difficulty = "medium", title } = req.body;
    const numQuestions = Math.min(parseInt(req.body.numQuestions) || 10, 30);
    const doc = store.documents[docId];
    if (!doc) return res.status(404).json({ error: "Document not found." });

    const { questions, method } = await generateGeminiQuiz(doc.text, numQuestions, difficulty, doc.course);
    const quiz = { id: uuidv4(), docId, title: title || `${doc.name} — Quiz`, course: doc.course, difficulty, method, questions, numQuestions: questions.length, createdAt: new Date().toISOString() };
    store.quizzes[quiz.id] = quiz;
    res.status(201).json({ quiz });
  } catch (err) {
    res.status(500).json({ error: "Failed to generate quiz." });
  }
});

app.post("/api/quiz/:id/submit", (req, res) => {
  const quiz = store.quizzes[req.params.id];
  if (!quiz) return res.status(404).json({ error: "Quiz not found." });
  const { answers = {} } = req.body;
  let correct = 0;

  const feedback = quiz.questions.map(q => {
    const userAnswer = (answers[q.id] || "").toString().trim().toLowerCase();
    const correctAnswer = q.answer.toString().trim().toLowerCase();
    const isCorrect = userAnswer === correctAnswer;
    if (isCorrect) correct++;
    return { questionId: q.id, question: q.question, userAnswer, correctAnswer, isCorrect, explanation: q.explanation };
  });

  const total = quiz.questions.length;
  const score = Math.round((correct / total) * 100);
  res.json({ score, percentage: `${score}%`, correct, total, feedback });
});

app.post("/api/tutor/chat", aiLimiter, async (req, res) => {
  const { message, subject } = req.body;
  if (!message) return res.status(400).json({ error: "Message is required." });
  const reply = await geminiChat(message, subject);
  res.json({ reply, timestamp: new Date().toISOString() });
});

// ── Complete Error Handlers & Server Boot ─────────────────────
app.use((_req, res) => {
  res.status(404).json({ error: "Route resource not found." });
});

app.use((err, _req, res, _next) => {
  console.error("🔥 Server Error Stack:", err.stack);
  res.status(500).json({ error: "Internal Server Error", message: err.message });
});

app.listen(PORT, () => {
  console.log(`🚀 MDX Backend running on http://localhost:${PORT}`);
});