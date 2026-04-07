//! Typed error types for the hayai crate.

/// Errors that can occur when building or using hayai components.
#[derive(Debug, thiserror::Error)]
pub enum HayaiError {
    /// One or more regex patterns in the set are invalid.
    #[error("invalid regex in pattern set: {source}")]
    InvalidPattern {
        /// The underlying regex error.
        #[from]
        source: regex::Error,
    },

    /// An I/O operation (e.g. cache read/write) failed.
    #[error("I/O error: {source}")]
    Io {
        /// The underlying I/O error.
        #[from]
        source: std::io::Error,
    },

    /// JSON serialization or deserialization failed.
    #[error("JSON error: {source}")]
    Json {
        /// The underlying `serde_json` error.
        #[from]
        source: serde_json::Error,
    },

    /// A mutex was poisoned (concurrent access failure).
    #[error("mutex poisoned: {context}")]
    MutexPoisoned {
        /// Description of which mutex was poisoned.
        context: String,
    },
}
