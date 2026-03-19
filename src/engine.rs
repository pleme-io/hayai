use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::sync::LazyLock;

use regex::{RegexSet, RegexSetBuilder};

// ═══════════════════════════════════════════════════════════════════
// Trait: Normalizer — pluggable input preprocessing
// ═══════════════════════════════════════════════════════════════════

/// Abstracts input normalization (e.g. stripping path prefixes).
///
/// Uses `Cow` to avoid allocation when no transformation is needed.
/// Implementations must be `Send + Sync` for thread-safe use.
pub trait Normalizer: Send + Sync {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str>;
}

// ═══════════════════════════════════════════════════════════════════
// Trait: Prefilter — pluggable fast-reject
// ═══════════════════════════════════════════════════════════════════

/// Abstracts the fast-reject prefilter that skips DFA matching for
/// inputs that are definitely safe.
///
/// Returns `true` if the input is safe (skip DFA). Returns `false`
/// if the input might match (must run DFA).
pub trait Prefilter: Send + Sync {
    fn is_safe(&self, input: &str) -> bool;
}

// ═══════════════════════════════════════════════════════════════════
// Trait: MatchEngine — pluggable matching
// ═══════════════════════════════════════════════════════════════════

/// Trait for pattern matching engines.
pub trait MatchEngine {
    /// Check input against patterns. Returns indices of matched patterns.
    fn check(&self, input: &str) -> Vec<usize>;
    /// Number of patterns in the engine.
    fn pattern_count(&self) -> usize;
}

// ═══════════════════════════════════════════════════════════════════
// Normalizer implementations
// ═══════════════════════════════════════════════════════════════════

/// No-op normalizer — returns input unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityNormalizer;

impl Normalizer for IdentityNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(input)
    }
}

static PATH_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"(?:/nix/store/[a-z0-9]+-[^/]+/bin/|/usr/local/bin/|/usr/bin/|/bin/|/sbin/)",
    )
    .unwrap()
});

/// Strips absolute path prefixes from inputs:
/// - `/nix/store/{hash}-{pkg}/bin/`
/// - `/usr/bin/`, `/usr/local/bin/`, `/bin/`, `/sbin/`
///
/// Returns `Cow::Borrowed` when no path prefix is present (zero alloc).
#[derive(Debug, Clone, Copy, Default)]
pub struct PathNormalizer;

impl Normalizer for PathNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        if PATH_RE.is_match(input) {
            Cow::Owned(PATH_RE.replace_all(input, "").into_owned())
        } else {
            Cow::Borrowed(input)
        }
    }
}

/// Generic normalizer combinator: applies `A` then `B`.
///
/// Preserves `Cow::Borrowed` when neither normalizer transforms the input.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChainedNormalizer<A, B> {
    pub first: A,
    pub second: B,
}

impl<A: Normalizer, B: Normalizer> Normalizer for ChainedNormalizer<A, B> {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let after_first = self.first.normalize(input);
        match after_first {
            Cow::Borrowed(s) => self.second.normalize(s),
            Cow::Owned(s) => {
                let after_second = self.second.normalize(&s);
                match after_second {
                    Cow::Borrowed(_) => Cow::Owned(s),
                    Cow::Owned(s2) => Cow::Owned(s2),
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Prefilter implementations
// ═══════════════════════════════════════════════════════════════════

/// No-op prefilter — nothing is safe, all inputs reach the DFA.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullPrefilter;

impl Prefilter for NullPrefilter {
    fn is_safe(&self, _input: &str) -> bool {
        false
    }
}

/// Prefix-based prefilter: skips DFA for inputs whose first N words
/// don't match any known prefix in the set.
#[derive(Debug, Clone)]
pub struct PrefixPrefilter {
    prefixes: HashSet<String>,
    max_words: usize,
}

impl PrefixPrefilter {
    /// Create a new prefix prefilter.
    ///
    /// `prefixes` — set of dangerous first-word prefixes.
    /// `max_words` — how many leading words to check (typically 3).
    #[must_use]
    pub fn new(prefixes: impl IntoIterator<Item = impl Into<String>>, max_words: usize) -> Self {
        Self {
            prefixes: prefixes.into_iter().map(Into::into).collect(),
            max_words,
        }
    }

    /// Access the prefix set.
    #[must_use]
    pub fn prefix_set(&self) -> &HashSet<String> {
        &self.prefixes
    }
}

impl Prefilter for PrefixPrefilter {
    fn is_safe(&self, input: &str) -> bool {
        let mut count = 0;
        for word in input.split_whitespace() {
            if count >= self.max_words {
                break;
            }
            if self.prefixes.contains(word)
                || self
                    .prefixes
                    .iter()
                    .any(|p| word.starts_with(p.as_str()))
            {
                return false;
            }
            count += 1;
        }
        true
    }
}

/// Keyword-based prefilter: skips DFA for inputs that don't contain
/// any of the specified keywords (case-insensitive ASCII search).
#[derive(Debug, Clone)]
pub struct KeywordPrefilter {
    keywords: Vec<Vec<u8>>,
}

impl KeywordPrefilter {
    /// Create a new keyword prefilter. Keywords should be UPPERCASE ASCII.
    #[must_use]
    pub fn new(keywords: impl IntoIterator<Item = impl Into<Vec<u8>>>) -> Self {
        Self {
            keywords: keywords.into_iter().map(Into::into).collect(),
        }
    }
}

impl Prefilter for KeywordPrefilter {
    fn is_safe(&self, input: &str) -> bool {
        !self
            .keywords
            .iter()
            .any(|kw| contains_ascii_ci(input.as_bytes(), kw))
    }
}

/// Case-insensitive ASCII substring search without allocation.
/// `needle` must be uppercase ASCII bytes.
#[inline]
pub fn contains_ascii_ci(haystack: &[u8], needle: &[u8]) -> bool {
    let n = needle.len();
    if n == 0 {
        return true;
    }
    if haystack.len() < n {
        return false;
    }
    haystack
        .windows(n)
        .any(|w| w.iter().zip(needle).all(|(a, b)| a.to_ascii_uppercase() == *b))
}

// ═══════════════════════════════════════════════════════════════════
// RegexMatcher — generic over Normalizer + Prefilter
// ═══════════════════════════════════════════════════════════════════

/// Generic regex pattern matcher with pluggable normalizer and prefilter.
///
/// Default type parameters provide minimal overhead (identity normalizer,
/// null prefilter). Consumers compose their own production pipeline.
pub struct RegexMatcher<N: Normalizer = IdentityNormalizer, P: Prefilter = NullPrefilter> {
    set: RegexSet,
    pattern_count: usize,
    normalizer: N,
    prefilter: P,
}

impl<N: Normalizer + fmt::Debug, P: Prefilter + fmt::Debug> fmt::Debug for RegexMatcher<N, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegexMatcher")
            .field("pattern_count", &self.pattern_count)
            .field("normalizer", &self.normalizer)
            .field("prefilter", &self.prefilter)
            .finish()
    }
}

impl RegexMatcher {
    /// Create a matcher with default normalizer and prefilter.
    ///
    /// # Errors
    /// Returns an error if any regex pattern is invalid.
    pub fn new(patterns: Vec<String>) -> anyhow::Result<Self> {
        Self::with_plugins(patterns, IdentityNormalizer, NullPrefilter)
    }
}

impl<N: Normalizer, P: Prefilter> RegexMatcher<N, P> {
    /// Create a matcher with custom normalizer and prefilter.
    ///
    /// # Errors
    /// Returns an error if any regex pattern is invalid.
    pub fn with_plugins(patterns: Vec<String>, normalizer: N, prefilter: P) -> anyhow::Result<Self> {
        let count = patterns.len();
        let set = RegexSetBuilder::new(&patterns)
            .size_limit(100 * 1024 * 1024)
            .build()
            .map_err(|e| anyhow::anyhow!("invalid regex in pattern set: {e}"))?;
        Ok(Self {
            set,
            pattern_count: count,
            normalizer,
            prefilter,
        })
    }
}

impl<N: Normalizer, P: Prefilter> MatchEngine for RegexMatcher<N, P> {
    fn check(&self, input: &str) -> Vec<usize> {
        let normalized = self.normalizer.normalize(input);

        if self.prefilter.is_safe(&normalized) {
            return Vec::new();
        }

        self.set.matches(&normalized).iter().collect()
    }

    fn pattern_count(&self) -> usize {
        self.pattern_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Normalizer tests ─────────────────────────────────────────

    #[test]
    fn identity_normalizer_is_noop() {
        let n = IdentityNormalizer;
        let result = n.normalize("anything");
        assert!(matches!(result, Cow::Borrowed("anything")));
    }

    #[test]
    fn path_normalizer_strips_nix_store_path() {
        let n = PathNormalizer;
        let result = n.normalize("/nix/store/abc123-pkg-1.0/bin/cmd check");
        assert_eq!(&*result, "cmd check");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn path_normalizer_borrows_when_no_path() {
        let n = PathNormalizer;
        let result = n.normalize("cargo test");
        assert_eq!(&*result, "cargo test");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn path_normalizer_strips_multiple_paths() {
        let n = PathNormalizer;
        let result =
            n.normalize("/nix/store/abc-foo-1.0/bin/cmd1 && /nix/store/def-bar-2.0/bin/cmd2");
        assert_eq!(&*result, "cmd1 && cmd2");
    }

    #[test]
    fn path_normalizer_strips_usr_bin() {
        let n = PathNormalizer;
        assert_eq!(&*n.normalize("/usr/bin/rm -rf /"), "rm -rf /");
    }

    #[test]
    fn path_normalizer_strips_usr_local_bin() {
        let n = PathNormalizer;
        assert_eq!(
            &*n.normalize("/usr/local/bin/terraform destroy"),
            "terraform destroy"
        );
    }

    #[test]
    fn path_normalizer_strips_sbin() {
        let n = PathNormalizer;
        assert_eq!(
            &*n.normalize("/sbin/mkfs.ext4 /dev/sda1"),
            "mkfs.ext4 /dev/sda1"
        );
    }

    // ── ChainedNormalizer tests ──────────────────────────────────

    #[test]
    fn chained_normalizer_borrows_when_clean() {
        let n = ChainedNormalizer {
            first: IdentityNormalizer,
            second: IdentityNormalizer,
        };
        let result = n.normalize("cargo test");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn chained_normalizer_applies_first_only() {
        let n = ChainedNormalizer {
            first: PathNormalizer,
            second: IdentityNormalizer,
        };
        let result = n.normalize("/usr/bin/ls -la");
        assert_eq!(&*result, "ls -la");
    }

    // ── Prefilter tests ──────────────────────────────────────────

    #[test]
    fn null_prefilter_never_safe() {
        let p = NullPrefilter;
        assert!(!p.is_safe("ls -la"));
        assert!(!p.is_safe("cargo test"));
    }

    #[test]
    fn prefix_prefilter_safe_commands() {
        let p = PrefixPrefilter::new(["rm", "git", "kubectl"], 3);
        assert!(p.is_safe("ls -la"));
        assert!(p.is_safe("cat file.txt"));
    }

    #[test]
    fn prefix_prefilter_dangerous_commands() {
        let p = PrefixPrefilter::new(["rm", "git", "kubectl"], 3);
        assert!(!p.is_safe("rm -rf /"));
        assert!(!p.is_safe("git push --force"));
        assert!(!p.is_safe("kubectl delete namespace prod"));
    }

    #[test]
    fn keyword_prefilter_safe() {
        let p = KeywordPrefilter::new([b"DROP ".to_vec(), b"TRUNCATE ".to_vec()]);
        assert!(p.is_safe("SELECT * FROM users"));
    }

    #[test]
    fn keyword_prefilter_dangerous() {
        let p = KeywordPrefilter::new([b"DROP ".to_vec(), b"TRUNCATE ".to_vec()]);
        assert!(!p.is_safe("psql -c 'DROP TABLE users'"));
        assert!(!p.is_safe("psql -c 'drop table users'"));
    }

    // ── contains_ascii_ci tests ──────────────────────────────────

    #[test]
    fn contains_ascii_ci_matches() {
        assert!(contains_ascii_ci(b"hello DROP TABLE world", b"DROP "));
        assert!(contains_ascii_ci(b"hello drop table world", b"DROP "));
        assert!(contains_ascii_ci(b"hello Drop Table world", b"DROP "));
    }

    #[test]
    fn contains_ascii_ci_no_match() {
        assert!(!contains_ascii_ci(b"hello world", b"DROP "));
        assert!(!contains_ascii_ci(b"DROPX", b"DROP "));
    }

    #[test]
    fn contains_ascii_ci_empty() {
        assert!(contains_ascii_ci(b"anything", b""));
        assert!(!contains_ascii_ci(b"", b"DROP "));
    }

    // ── RegexMatcher tests ───────────────────────────────────────

    #[test]
    fn matcher_basic_match() {
        let patterns = vec![r"rm\s+-rf".to_string(), r"DROP\s+TABLE".to_string()];
        let matcher = RegexMatcher::new(patterns).unwrap();
        let matches = matcher.check("rm -rf /");
        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn matcher_no_match() {
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::new(patterns).unwrap();
        let matches = matcher.check("ls -la");
        assert!(matches.is_empty());
    }

    #[test]
    fn matcher_multiple_matches() {
        let patterns = vec![r"rm".to_string(), r"-rf".to_string()];
        let matcher = RegexMatcher::new(patterns).unwrap();
        let matches = matcher.check("rm -rf /");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn matcher_with_path_normalizer() {
        let patterns = vec![r"^rm\s+-rf".to_string()];
        let matcher =
            RegexMatcher::with_plugins(patterns, PathNormalizer, NullPrefilter).unwrap();
        let matches = matcher.check("/usr/bin/rm -rf /");
        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn matcher_with_prefilter_skips() {
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::with_plugins(
            patterns,
            IdentityNormalizer,
            PrefixPrefilter::new(["rm"], 3),
        )
        .unwrap();
        // "ls" is safe, skips DFA
        let matches = matcher.check("ls -la");
        assert!(matches.is_empty());
        // "rm" is dangerous, runs DFA
        let matches = matcher.check("rm -rf /");
        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn matcher_invalid_regex_rejected() {
        let patterns = vec!["[invalid".to_string()];
        assert!(RegexMatcher::new(patterns).is_err());
    }

    #[test]
    fn matcher_pattern_count() {
        let patterns = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let matcher = RegexMatcher::new(patterns).unwrap();
        assert_eq!(matcher.pattern_count(), 3);
    }

    #[test]
    fn matcher_empty_input() {
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::new(patterns).unwrap();
        assert!(matcher.check("").is_empty());
    }

    #[test]
    fn matcher_debug_impl() {
        let matcher = RegexMatcher::new(vec!["test".to_string()]).unwrap();
        let debug = format!("{matcher:?}");
        assert!(debug.contains("RegexMatcher"));
        assert!(debug.contains("pattern_count"));
    }
}
