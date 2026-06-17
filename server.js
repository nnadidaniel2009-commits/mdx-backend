/** * ============================================================
 * MEDANIXX (MDX) — Preclinical Study App
 * server.js — Main Backend Server (Full Document & Quiz Version)
 * Built for: UNN 100 Level Students
 * ============================================================
 */

"use strict";

const express        = require("express");
const cors           = require("cors");
const dotenv         = require("dotenv");
const morgan         = require("morgan");
const helmet         = require("helmet");
const rateLimit      = require("express-rate-limit");
const path           = require("path");
const fs             = require("fs");
const os             = require("os");
const multer         = require("multer");
const pdfParse       = require("pdf-parse").default || require("pdf-parse");
const Tesseract      = require("tesseract.js");
const Groq           = require("groq-sdk");

// ── Environment Variable Resolution ─────────────────────────
const localEnvPath  = path.resolve(__dirname, ".env");
const parentEnvPath = path.resolve(__dirname, "../.env");

if (fs.existsSync(localEnvPath)) {
  dotenv.config({ path: localEnvPath });
} else if (fs.existsSync(parentEnvPath)) {
  dotenv.config({ path: parentEnvPath });
} else {
  dotenv.config();
}

// ── Groq AI Setup ─────────────────────────────────────────────
const GROQ_API_KEY = process.env.GROQ_API_KEY;

let groqClient = null;
if (GROQ_API_KEY) {
  groqClient = new Groq({ apiKey: GROQ_API_KEY });
  console.log("✅ Groq AI initialized — Speed mode active.");
} else {
  console.log("⚠️  No GROQ_API_KEY found. Set it in your .env file.");
}

const app  = express();
const PORT = process.env.PORT || 5000;

// ── Multer File Upload Setup ──────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB file size limit
});

// ── Middleware ────────────────────────────────────────────────
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({
  origin: "*",
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization", "Accept"],
  credentials: true,
}));
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(morgan("dev"));

const aiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  message: { error: "Too many requests. Please wait a moment." },
});

// ── JSON Cleaner — strips markdown fences Groq sometimes adds ─
function cleanJsonResponse(raw) {
  let cleaned = raw.trim();
  cleaned = cleaned.replace(/^```json\s*/i, "");
  cleaned = cleaned.replace(/^```\s*/i, "");
  cleaned = cleaned.replace(/```\s*$/i, "");
  cleaned = cleaned.trim();
  const objectMatch = cleaned.match(/\{[\s\S]*\}/);
  if (objectMatch) cleaned = objectMatch[0];
  return cleaned;
}

// ── OCR Helper — extracts text from scanned/image-based PDFs ──
// Uses pdfjs-dist (pure JS) + canvas to render pages, then Tesseract OCR.
// No system binaries required — works on Windows, Mac, and Linux.
async function extractTextFromScannedPDF(fileBuffer) {
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");
  const { createCanvas } = require("canvas");

  console.log("🔍 [OCR] Loading PDF with pdfjs...");

  const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(fileBuffer) });
  const pdfDoc = await loadingTask.promise;
  const numPages = pdfDoc.numPages;

  console.log(`🔍 [OCR] PDF has ${numPages} page(s). Running OCR...`);

  let fullText = "";

  for (let pageNum = 1; pageNum <= numPages; pageNum++) {
    const page     = await pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: 2.0 }); // 2x scale for better OCR accuracy

    // Render the PDF page onto a Node canvas
    const canvas  = createCanvas(viewport.width, viewport.height);
    const context = canvas.getContext("2d");

    await page.render({ canvasContext: context, viewport }).promise;

    // Export canvas to PNG buffer (no temp files needed)
    const imgBuffer = canvas.toBuffer("image/png");

    console.log(`🔍 [OCR] Tesseract on page ${pageNum}/${numPages}...`);

    // Run Tesseract OCR directly on the PNG buffer
    const { data: { text } } = await Tesseract.recognize(imgBuffer, "eng", {
      logger: m => {
        if (m.status === "recognizing text") {
          process.stdout.write(`\r🔍 [OCR] Page ${pageNum}: ${Math.round(m.progress * 100)}%`);
        }
      },
    });

    fullText += text + "\n";
  }

  console.log("\n✅ [OCR] Text extraction complete.");
  return fullText.trim();
}

// ── Core AI Helper ────────────────────────────────────────────
async function groqChat(userMessage, systemPrompt, isJsonExpected = false) {
  if (!groqClient) return "AI Tutor configuration missing. Please set your GROQ_API_KEY.";
  try {
    const options = {
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user",   content: userMessage  },
      ],
      model: "llama-3.3-70b-versatile",
      temperature: 0.1,
      max_tokens: 4000,
    };

    if (isJsonExpected) {
      options.response_format = { type: "json_object" };
    }

    const completion = await groqClient.chat.completions.create(options);
    return completion.choices[0]?.message?.content || "";
  } catch (err) {
    console.error("🔥 Groq Handshake API Error:", err.message);
    return "";
  }
}

// ── API Routes ────────────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({ status: "ok", app: "Medanixx (MDX)", aiEnabled: !!groqClient });
});

// 1. Chat Route
app.post("/api/chat", aiLimiter, async (req, res) => {
  const { message, subject } = req.body;
  if (!message) return res.status(400).json({ error: "Message field is required." });

  const systemPrompt = `You are Medanixx AI, an intelligent study assistant for 100-level university students in Nigeria studying ${subject || "General"}.

CRITICAL IDENTITY RULE: If the user asks who created you, who made you, or anything about your developer, you must reply EXACTLY with:
"I was created by Nnadi Daniel popularly known as Danixx, a 100 level Medical student at the University of Nigeria."

CRITICAL IDENTITY RULE: If the user asks you who or what is Medanixx Ai, you must reply EXACTLY with:
"I am Medanixx Ai specifically created by Nnadi Daniel for university students in Nigeria. My purpose is to assist students with their studies and provide helpful explanations."

For all other general study questions, explain concepts clearly and concisely, listing them in a structured format.`;

  const reply = await groqChat(message, systemPrompt, false);
  res.json({ reply, timestamp: new Date().toISOString() });
});

// 2. Document Upload & Quiz Generation Route
app.post("/api/upload", aiLimiter, upload.single("file"), async (req, res) => {
  if (!groqClient) return res.status(500).json({ error: "AI Backend not configured. Check your GROQ_API_KEY." });
  if (!req.file)   return res.status(400).json({ error: "No file uploaded." });

  try {
    let extractedText = "";
    let usedOCR = false;

    if (req.file.mimetype === "application/pdf") {
      const pdfData = await pdfParse(req.file.buffer);
      extractedText = pdfData.text;
    } else {
      extractedText = req.file.buffer.toString("utf-8");
    }

    // Clean whitespace
    extractedText = extractedText.replace(/\s+/g, " ").trim();

    console.log(`\n📝 [File Upload] File: "${req.file.originalname}"`);
    console.log(`📝 [File Upload] Extracted text length: ${extractedText.length} characters`);

    // ── If text too short, attempt OCR on scanned PDF ──────────
    if (!extractedText || extractedText.length < 200) {
      console.log("⚠️  [OCR Trigger] Low text detected — attempting OCR on scanned PDF...");

      if (req.file.mimetype !== "application/pdf") {
        return res.status(400).json({
          error: "Could not extract readable text. Please upload a PDF or plain text file.",
        });
      }

      try {
        extractedText = await extractTextFromScannedPDF(req.file.buffer);
        extractedText = extractedText.replace(/\s+/g, " ").trim();
        usedOCR = true;
        console.log(`📝 [OCR Result] Extracted ${extractedText.length} characters via Tesseract OCR.`);
      } catch (ocrErr) {
        console.error("❌ [OCR Failed]:", ocrErr.message);
        return res.status(400).json({
          error: "This appears to be a scanned document but OCR extraction failed. Please try a clearer or higher quality scan.",
        });
      }

      // Final check after OCR
      if (!extractedText || extractedText.length < 200) {
        return res.status(400).json({
          error: "Could not extract enough readable text even after OCR. Please use a higher quality scan or a typed digital PDF.",
        });
      }
    }

    // Sanitize OCR text — removes non-ASCII garbage that confuses Groq
    extractedText = extractedText.replace(/[^\x20-\x7E\n]/g, " ");

    // Truncate at a word boundary to stay under Groq free tier TPM limit
    let trimmedText = extractedText.substring(0, 8000);
    trimmedText = trimmedText.replace(/\s\S*$/, "");

    const systemPrompt = `You are an expert exam generator for university medical and engineering students. You must always return valid JSON only.

Your ONLY job is to return a valid JSON object containing an array named "questions" with exactly 20 high-yield multiple-choice questions based on the document text provided.

You must wrap everything inside an outer object containing a "questions" key array like this:
{
  "questions": [
    {
      "question": "The question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "The exact string matching the correct option",
      "explanation": "A clear scientific explanation of why this answer is correct."
    }
  ]
}

CRITICAL: Return raw JSON only. No introductory text, no conversational text, no markdown code blocks, no backticks, no fences. The response must start with { and end with }.`;

    const userMessage = `Here is the course material text. Generate exactly 20 exam questions and return them strictly inside a valid JSON object matching the requested schema layout:\n\n${trimmedText}`;

    console.log("⏳ [Groq Pipeline] Sending to Llama-3.3-70b...");

    // Retry logic — attempt up to 2 times before returning 422
    let questionsArray = null;

    for (let attempt = 1; attempt <= 2; attempt++) {
      console.log(`🔄 [Groq Pipeline] Attempt ${attempt} of 2...`);
      const aiResponse = await groqChat(userMessage, systemPrompt, true);

      if (!aiResponse) {
        console.error(`❌ [Attempt ${attempt}] Empty response from Groq.`);
        continue;
      }

      try {
        const cleanJsonString = cleanJsonResponse(aiResponse);
        const parsedData      = JSON.parse(cleanJsonString);
        const extracted       = parsedData.questions || parsedData;

        if (!Array.isArray(extracted) || extracted.length === 0) {
          throw new Error("Parsed JSON has no valid questions array.");
        }

        questionsArray = extracted;
        console.log(`✅ [Groq Pipeline] Attempt ${attempt} succeeded — ${questionsArray.length} questions generated.`);
        break;

      } catch (parseErr) {
        console.error(`❌ [Attempt ${attempt}] JSON parse failed: ${parseErr.message}`);
      }
    }

    if (!questionsArray) {
      return res.status(422).json({
        error: "Could not parse AI questions after multiple attempts. Please ensure your document contains enough typed course content and try again.",
      });
    }

    return res.json({
      success: true,
      questions: questionsArray,
      usedOCR,
    });

  } catch (err) {
    console.error("🔥 Main Upload Endpoint Failure:", err.message);
    res.status(500).json({ error: "An error occurred while processing the document." });
  }
});

// ── 404 Handler ───────────────────────────────────────────────
app.use((_req, res) => res.status(404).json({ error: "Route not found on this localized port engine server." }));

// ── Start Server ──────────────────────────────────────────────
app.listen(PORT, () => console.log(`🚀 MDX Backend running on http://localhost:${PORT}`));

module.exports = app;