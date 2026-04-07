use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::sync::LazyLock;

use regex::{RegexSet, RegexSetBuilder};

use crate::error::HayaiError;

// ═══════════════════════════════════════════════════════════════════
// Trait: Normalizer — pluggable input preprocessing
// ═══════════════════════════════════════════════════════════════════

/// Abstracts input normalization (e.g. stripping path prefixes).
///
/// Uses `Cow` to avoid allocation when no transformation is needed.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` because matchers are typically
/// constructed once and shared across threads (e.g. behind an `Arc`).
/// All built-in normalizers satisfy this automatically.
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
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` because prefilters live inside
/// matchers that may be shared across threads. All built-in prefilters
/// satisfy this automatically.
pub trait Prefilter: Send + Sync {
    fn is_safe(&self, input: &str) -> bool;
}

// ═══════════════════════════════════════════════════════════════════
// Trait: MatchEngine — pluggable matching
// ═══════════════════════════════════════════════════════════════════

/// Trait for pattern matching engines.
///
/// # Thread Safety
///
/// Implementors are not required to be `Send + Sync` by default, but
/// concrete types like `RegexMatcher` are when their normalizer and
/// prefilter are.
pub trait MatchEngine {
    /// Check input against patterns. Returns indices of matched patterns.
    fn check(&self, input: &str) -> Vec<usize>;
    /// Number of patterns in the engine.
    fn pattern_count(&self) -> usize;
}

// ═══════════════════════════════════════════════════════════════════
// MatchResult — richer match info
// ═══════════════════════════════════════════════════════════════════

/// Result of a pattern match with additional context.
///
/// Wraps the raw `Vec<usize>` of matched pattern indices with
/// convenience methods for common checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// Indices of matched patterns.
    pub indices: Vec<usize>,
}

impl MatchResult {
    /// Returns `true` if no patterns matched.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns `true` if at least one pattern matched.
    #[must_use]
    pub fn matched_any(&self) -> bool {
        !self.indices.is_empty()
    }

    /// Returns the index of the first matched pattern, if any.
    #[must_use]
    pub fn first(&self) -> Option<usize> {
        self.indices.first().copied()
    }

    /// Returns the number of matched patterns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

impl From<Vec<usize>> for MatchResult {
    fn from(indices: Vec<usize>) -> Self {
        Self { indices }
    }
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
        for (count, word) in input.split_whitespace().enumerate() {
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

/// Combines multiple prefilters: input is safe only if ALL prefilters agree it's safe.
///
/// Enables composing `PrefixPrefilter` + `KeywordPrefilter` without custom code.
/// Both prefilters must report the input as safe for the composite to skip DFA.
#[derive(Debug, Clone)]
pub struct CompositePrefilter<A: Prefilter, B: Prefilter> {
    pub first: A,
    pub second: B,
}

impl<A: Prefilter, B: Prefilter> Prefilter for CompositePrefilter<A, B> {
    fn is_safe(&self, input: &str) -> bool {
        self.first.is_safe(input) && self.second.is_safe(input)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Closure wrappers — FnNormalizer, FnPrefilter
// ═══════════════════════════════════════════════════════════════════

/// Wraps a function as a [`Normalizer`].
///
/// Because `Normalizer::normalize` uses a higher-ranked lifetime (`for<'a>`),
/// closures that capture environment cannot directly satisfy the bound.
/// Use [`FnNormalizer::new`] with a function pointer or non-capturing closure:
///
/// ```
/// use hayai::*;
/// use std::borrow::Cow;
///
/// let n = FnNormalizer::new(|s| Cow::Owned(s.to_uppercase()));
/// assert_eq!(&*n.normalize("hello"), "HELLO");
/// ```
pub struct FnNormalizer {
    f: fn(&str) -> Cow<'_, str>,
}

impl FnNormalizer {
    /// Create a new `FnNormalizer` from a function pointer.
    #[must_use]
    pub fn new(f: fn(&str) -> Cow<'_, str>) -> Self {
        Self { f }
    }
}

impl Normalizer for FnNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        (self.f)(input)
    }
}

impl fmt::Debug for FnNormalizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FnNormalizer(<fn>)")
    }
}

/// Wraps a closure as a [`Prefilter`].
///
/// Enables inline prefilters without defining a new type:
/// ```
/// use hayai::*;
///
/// let p = FnPrefilter(|s: &str| s.starts_with("safe_"));
/// assert!(p.is_safe("safe_command"));
/// assert!(!p.is_safe("dangerous"));
/// ```
pub struct FnPrefilter<F>(pub F);

impl<F> Prefilter for FnPrefilter<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    fn is_safe(&self, input: &str) -> bool {
        (self.0)(input)
    }
}

impl<F> fmt::Debug for FnPrefilter<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FnPrefilter(<closure>)")
    }
}

/// Case-insensitive ASCII substring search without allocation.
/// `needle` must be uppercase ASCII bytes.
#[inline]
#[must_use]
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
            .finish_non_exhaustive()
    }
}

impl RegexMatcher {
    /// Create a matcher with default normalizer and prefilter.
    ///
    /// # Errors
    /// Returns [`HayaiError::InvalidPattern`] if any regex pattern is invalid.
    pub fn new(patterns: &[String]) -> Result<Self, HayaiError> {
        Self::with_plugins(patterns, IdentityNormalizer, NullPrefilter)
    }
}

impl<N: Normalizer, P: Prefilter> RegexMatcher<N, P> {
    /// Create a matcher with custom normalizer and prefilter.
    ///
    /// # Errors
    /// Returns [`HayaiError::InvalidPattern`] if any regex pattern is invalid.
    pub fn with_plugins(
        patterns: &[String],
        normalizer: N,
        prefilter: P,
    ) -> Result<Self, HayaiError> {
        let count = patterns.len();
        let set = RegexSetBuilder::new(patterns)
            .size_limit(100 * 1024 * 1024)
            .build()?;
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
        let matcher = RegexMatcher::new(&patterns).unwrap();
        let matches = matcher.check("rm -rf /");
        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn matcher_no_match() {
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        let matches = matcher.check("ls -la");
        assert!(matches.is_empty());
    }

    #[test]
    fn matcher_multiple_matches() {
        let patterns = vec![r"rm".to_string(), r"-rf".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        let matches = matcher.check("rm -rf /");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn matcher_with_path_normalizer() {
        let patterns = vec![r"^rm\s+-rf".to_string()];
        let matcher =
            RegexMatcher::with_plugins(&patterns, PathNormalizer, NullPrefilter).unwrap();
        let matches = matcher.check("/usr/bin/rm -rf /");
        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn matcher_with_prefilter_skips() {
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            IdentityNormalizer,
            PrefixPrefilter::new(["rm"], 3),
        )
        .unwrap();
        let matches = matcher.check("ls -la");
        assert!(matches.is_empty());
        let matches = matcher.check("rm -rf /");
        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn matcher_invalid_regex_rejected() {
        let patterns = vec!["[invalid".to_string()];
        assert!(RegexMatcher::new(&patterns).is_err());
    }

    #[test]
    fn matcher_pattern_count() {
        let patterns = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.pattern_count(), 3);
    }

    #[test]
    fn matcher_empty_input() {
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert!(matcher.check("").is_empty());
    }

    #[test]
    fn matcher_debug_impl() {
        let patterns = vec!["test".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        let debug = format!("{matcher:?}");
        assert!(debug.contains("RegexMatcher"));
        assert!(debug.contains("pattern_count"));
    }

    // ── CompositePrefilter tests ─────────────────────────────────

    #[test]
    fn composite_prefilter_both_safe() {
        let prefix = PrefixPrefilter::new(["rm", "git"], 3);
        let keyword = KeywordPrefilter::new([b"DROP ".to_vec()]);
        let composite = CompositePrefilter {
            first: prefix,
            second: keyword,
        };
        // "ls -la" is safe for both prefix (no "rm"/"git") and keyword (no "DROP")
        assert!(composite.is_safe("ls -la"));
    }

    #[test]
    fn composite_prefilter_first_unsafe() {
        let prefix = PrefixPrefilter::new(["rm", "git"], 3);
        let keyword = KeywordPrefilter::new([b"DROP ".to_vec()]);
        let composite = CompositePrefilter {
            first: prefix,
            second: keyword,
        };
        // "rm -rf /" — prefix says unsafe, so composite says unsafe
        assert!(!composite.is_safe("rm -rf /"));
    }

    #[test]
    fn composite_prefilter_second_unsafe() {
        let prefix = PrefixPrefilter::new(["rm", "git"], 3);
        let keyword = KeywordPrefilter::new([b"DROP ".to_vec()]);
        let composite = CompositePrefilter {
            first: prefix,
            second: keyword,
        };
        // "psql DROP TABLE" — prefix says safe, but keyword says unsafe
        assert!(!composite.is_safe("psql DROP TABLE users"));
    }

    #[test]
    fn composite_prefilter_both_unsafe() {
        let prefix = PrefixPrefilter::new(["rm"], 3);
        let keyword = KeywordPrefilter::new([b"DROP ".to_vec()]);
        let composite = CompositePrefilter {
            first: prefix,
            second: keyword,
        };
        // Short-circuits on first
        assert!(!composite.is_safe("rm DROP TABLE"));
    }

    // ── FnNormalizer tests ───────────────────────────────────────

    #[test]
    fn fn_normalizer_uppercase() {
        let n = FnNormalizer::new(|s| Cow::Owned(s.to_uppercase()));
        assert_eq!(&*n.normalize("hello"), "HELLO");
    }

    #[test]
    fn fn_normalizer_identity() {
        let n = FnNormalizer::new(|s| Cow::Borrowed(s));
        let result = n.normalize("unchanged");
        assert!(matches!(result, Cow::Borrowed("unchanged")));
    }

    #[test]
    fn fn_normalizer_debug() {
        let n = FnNormalizer::new(|s| Cow::Borrowed(s));
        let debug = format!("{n:?}");
        assert!(debug.contains("FnNormalizer"));
    }

    #[test]
    fn fn_normalizer_in_matcher() {
        let patterns = vec![r"HELLO".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            FnNormalizer::new(|s| Cow::Owned(s.to_uppercase())),
            NullPrefilter,
        )
        .unwrap();
        assert_eq!(matcher.check("hello world"), vec![0]);
    }

    // ── FnPrefilter tests ────────────────────────────────────────

    #[test]
    fn fn_prefilter_custom() {
        let p = FnPrefilter(|s: &str| s.starts_with("safe_"));
        assert!(p.is_safe("safe_command"));
        assert!(!p.is_safe("dangerous"));
    }

    #[test]
    fn fn_prefilter_debug() {
        let p = FnPrefilter(|_: &str| true);
        let debug = format!("{p:?}");
        assert!(debug.contains("FnPrefilter"));
    }

    #[test]
    fn fn_prefilter_in_matcher() {
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            IdentityNormalizer,
            FnPrefilter(|s: &str| !s.starts_with("rm")),
        )
        .unwrap();
        assert!(matcher.check("ls -la").is_empty());
        assert_eq!(matcher.check("rm -rf /"), vec![0]);
    }

    // ── MatchResult tests ────────────────────────────────────────

    #[test]
    fn match_result_empty() {
        let r = MatchResult::from(vec![]);
        assert!(r.is_empty());
        assert!(!r.matched_any());
        assert_eq!(r.first(), None);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn match_result_with_matches() {
        let r = MatchResult::from(vec![0, 3, 7]);
        assert!(!r.is_empty());
        assert!(r.matched_any());
        assert_eq!(r.first(), Some(0));
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn match_result_from_vec() {
        let indices = vec![1, 2, 3];
        let r: MatchResult = indices.into();
        assert_eq!(r.indices, vec![1, 2, 3]);
    }

    #[test]
    fn match_result_equality() {
        let a = MatchResult::from(vec![0, 1]);
        let b = MatchResult::from(vec![0, 1]);
        let c = MatchResult::from(vec![0, 2]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn match_result_debug() {
        let r = MatchResult::from(vec![0]);
        let debug = format!("{r:?}");
        assert!(debug.contains("MatchResult"));
        assert!(debug.contains("indices"));
    }

    // ── Performance edge cases ───────────────────────────────────

    #[test]
    fn matcher_large_pattern_set() {
        let patterns: Vec<String> = (0..1000).map(|i| format!(r"\bpattern_{i}\b")).collect();
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.pattern_count(), 1000);
        assert!(matcher.check("no match here").is_empty());
        assert_eq!(matcher.check("pattern_500"), vec![500]);
    }

    #[test]
    fn matcher_unicode_input() {
        let patterns = vec!["caf\u{00e9}".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.check("I love caf\u{00e9}"), vec![0]);
    }

    #[test]
    fn matcher_unicode_no_match() {
        let patterns = vec!["caf\u{00e9}".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert!(matcher.check("I love cafe").is_empty());
    }

    #[test]
    fn path_normalizer_preserves_args() {
        let n = PathNormalizer;
        let result = n.normalize("/nix/store/abc123-pkg-1.0/bin/cmd --flag value");
        assert_eq!(&*result, "cmd --flag value");
    }

    #[test]
    fn chained_normalizer_three_deep() {
        let n = ChainedNormalizer {
            first: PathNormalizer,
            second: ChainedNormalizer {
                first: IdentityNormalizer,
                second: IdentityNormalizer,
            },
        };
        assert_eq!(&*n.normalize("/usr/bin/ls -la"), "ls -la");
    }

    #[test]
    fn chained_normalizer_fn_plus_path() {
        let n = ChainedNormalizer {
            first: PathNormalizer,
            second: FnNormalizer::new(|s| Cow::Owned(s.to_lowercase())),
        };
        assert_eq!(&*n.normalize("/usr/bin/LS -LA"), "ls -la");
    }

    // ── Composite in matcher ─────────────────────────────────────

    #[test]
    fn matcher_with_composite_prefilter() {
        let patterns = vec![r"rm\s+-rf".to_string(), r"DROP\s+TABLE".to_string()];
        let composite = CompositePrefilter {
            first: PrefixPrefilter::new(["rm", "psql"], 3),
            second: KeywordPrefilter::new([b"DROP ".to_vec()]),
        };
        let matcher =
            RegexMatcher::with_plugins(&patterns, IdentityNormalizer, composite).unwrap();
        assert!(matcher.check("ls -la").is_empty());
        assert_eq!(matcher.check("rm -rf /"), vec![0]);
        assert_eq!(matcher.check("echo DROP TABLE users"), vec![1]);
    }
}
