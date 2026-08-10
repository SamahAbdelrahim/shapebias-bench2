/**
 * POST /api/log - store one trial row.
 *
 * Mirrors the Express route in server.js so the frontend is identical whether
 * it runs locally or on Vercel.
 */

const { connect, ShapeBiasHumanTrial } = require("./_mongo");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ ok: false, error: "Method not allowed" });
  }
  try {
    await connect();
    const payload = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
    await ShapeBiasHumanTrial.create(payload);
    return res.status(200).json({ ok: true });
  } catch (err) {
    console.error("Failed to save trial:", err);
    return res.status(500).json({ ok: false, error: String((err && err.message) || err) });
  }
};
