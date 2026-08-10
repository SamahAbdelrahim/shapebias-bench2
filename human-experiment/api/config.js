/**
 * GET /api/config - completion code and which trial pool is deployed.
 */

const fs = require("fs");
const path = require("path");

module.exports = function handler(_req, res) {
  let poolVersion = null;
  let counts = null;
  try {
    const poolPath = path.join(process.cwd(), "public", "trial_pool.json");
    const pool = JSON.parse(fs.readFileSync(poolPath, "utf8"));
    poolVersion = pool.pool_version || null;
    counts = pool.counts || null;
  } catch (_err) {
    // The frontend reads the pool as a static file; config is informational.
  }
  res.status(200).json({
    completion_code: process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE",
    design: "matched_v2",
    pool_version: poolVersion,
    pool_counts: counts
  });
};
