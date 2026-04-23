//! Hayai (速い) — generic fast-match engine with pluggable normalizers and prefilters.
//!
//! This crate provides a domain-agnostic pattern matching pipeline:
//! Normalize → Prefilter → `RegexSet` DFA
//!
//! # Example
//! ```
//! use hayai::*;
//!
//! let patterns = vec!["rm\\s+-rf".to_string(), "DROP\\s+TABLE".to_string()];
//! let matcher = RegexMatcher::new(&patterns).unwrap();
//! let matches = matcher.check("rm -rf /");
//! assert!(!matches.is_empty());
//! ```

pub mod cache;
pub mod engine;
pub mod error;

pub use cache::{CacheStore, FixedFingerprinter, Fingerprinter, MemCache, resolve_cached};
pub use engine::{
    ChainedNormalizer, CompositePrefilter, FnNormalizer, FnPrefilter, IdentityNormalizer,
    KeywordPrefilter, MatchResult, MockMatchEngine, MockNormalizer, MockPrefilter, Normalizer,
    NullPrefilter, PathNormalizer, Prefilter, RegexMatcher, contains_ascii_ci,
};
pub use error::HayaiError;