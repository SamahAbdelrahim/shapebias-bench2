/**
 * Shared Mongo connection for the serverless handlers.
 *
 * Serverless invocations reuse a warm container, so the connection is cached on
 * the module scope: opening a new pool per request exhausts an Atlas cluster's
 * connection limit under Prolific-rate traffic.
 */

const mongoose = require("mongoose");

const ShapeBiasHumanTrial = require("../models/shapebias-human-logger");

let cached = global.__sbMongo;
if (!cached) {
  cached = global.__sbMongo = { conn: null, promise: null };
}

async function connect() {
  if (cached.conn) return cached.conn;
  const uri = process.env.MONGO_URI;
  if (!uri) {
    throw new Error("MONGO_URI is not set. Add it to the deployment environment.");
  }
  if (!cached.promise) {
    cached.promise = mongoose
      .connect(uri, {
        // Fail fast rather than holding a request open for the default 30 s.
        serverSelectionTimeoutMS: 5000,
        maxPoolSize: 5
      })
      .then((m) => m);
  }
  cached.conn = await cached.promise;
  return cached.conn;
}

module.exports = { connect, ShapeBiasHumanTrial };
