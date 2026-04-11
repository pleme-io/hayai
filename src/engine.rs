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
    /// Transform the input, returning `Cow::Borrowed` when no change is needed.
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
    /// Returns `true` if the input can be safely skipped (no DFA needed).
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
    #[must_use]
    fn check(&self, input: &str) -> Vec<usize>;
    /// Number of patterns in the engine.
    #[must_use]
    fn pattern_count(&self) -> usize;
}

// ═══════════════════════════════════════════════════════════════════
// MatchResult — richer match info
// ═══════════════════════════════════════════════════════════════════

/// Result of a pattern match with additional context.
///
/// Wraps the raw `Vec<usize>` of matched pattern indices with
/// convenience methods for common checks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

    /// Returns `true` if the given pattern index is among the matches.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        self.indices.contains(&index)
    }

    /// Returns an iterator over the matched pattern indices.
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.indices.iter()
    }
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.indices.is_empty() {
            f.write_str("no matches")
        } else {
            write!(f, "{} match(es): {:?}", self.indices.len(), self.indices)
        }
    }
}

impl From<Vec<usize>> for MatchResult {
    fn from(indices: Vec<usize>) -> Self {
        Self { indices }
    }
}

impl IntoIterator for MatchResult {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.indices.into_iter()
    }
}

impl<'a> IntoIterator for &'a MatchResult {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.indices.iter()
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

// TODO(scope): `?` is unavailable inside `LazyLock`; this regex is a
// compile-time constant and cannot fail, but a `regex_static!` macro or
// `OnceLock` with a fallible init would let us remove the `unwrap`.
#[allow(clippy::unwrap_used)]
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
    /// First normalizer applied to the input.
    pub first: A,
    /// Second normalizer applied to the result of `first`.
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
        !input
            .split_whitespace()
            .take(self.max_words)
            .any(|word| {
                self.prefixes.contains(word)
                    || self.prefixes.iter().any(|p| word.starts_with(p.as_str()))
            })
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
    /// First prefilter to check.
    pub first: A,
    /// Second prefilter to check (only if first reports safe).
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
pub(crate) fn contains_ascii_ci(haystack: &[u8], needle: &[u8]) -> bool {
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
    pub fn new<I, S>(patterns: I) -> Result<Self, HayaiError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::with_plugins(patterns, IdentityNormalizer, NullPrefilter)
    }
}

impl<N: Normalizer, P: Prefilter> RegexMatcher<N, P> {
    /// Create a matcher with custom normalizer and prefilter.
    ///
    /// # Errors
    /// Returns [`HayaiError::InvalidPattern`] if any regex pattern is invalid.
    pub fn with_plugins<I, S>(
        patterns: I,
        normalizer: N,
        prefilter: P,
    ) -> Result<Self, HayaiError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let patterns: Vec<String> = patterns
            .into_iter()
            .map(|s| s.as_ref().to_owned())
            .collect();
        let count = patterns.len();
        let set = RegexSetBuilder::new(&patterns)
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

// ═══════════════════════════════════════════════════════════════════
// Test mocks — available under `#[cfg(test)]`
// ═══════════════════════════════════════════════════════════════════

/// A configurable mock for [`MatchEngine`], useful for testing code
/// that consumes matchers without building a real regex set.
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct MockMatchEngine {
    pub results: Vec<usize>,
    pub count: usize,
}

#[cfg(test)]
impl MockMatchEngine {
    pub fn new(results: Vec<usize>, count: usize) -> Self {
        Self { results, count }
    }

    pub fn empty(count: usize) -> Self {
        Self {
            results: Vec::new(),
            count,
        }
    }
}

#[cfg(test)]
impl MatchEngine for MockMatchEngine {
    fn check(&self, _input: &str) -> Vec<usize> {
        self.results.clone()
    }

    fn pattern_count(&self) -> usize {
        self.count
    }
}

/// A mock normalizer that applies an optional transform.
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct MockNormalizer {
    pub transform: Option<fn(&str) -> String>,
}

#[cfg(test)]
impl MockNormalizer {
    pub fn identity() -> Self {
        Self { transform: None }
    }

    pub fn with_transform(f: fn(&str) -> String) -> Self {
        Self { transform: Some(f) }
    }
}

#[cfg(test)]
impl Normalizer for MockNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match self.transform {
            Some(f) => Cow::Owned(f(input)),
            None => Cow::Borrowed(input),
        }
    }
}

/// A mock prefilter that returns a fixed value.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct MockPrefilter {
    pub always_safe: bool,
}

#[cfg(test)]
impl Prefilter for MockPrefilter {
    fn is_safe(&self, _input: &str) -> bool {
        self.always_safe
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

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

    #[test]
    fn match_result_contains() {
        let r = MatchResult::from(vec![0, 3, 7]);
        assert!(r.contains(0));
        assert!(r.contains(3));
        assert!(!r.contains(1));
    }

    #[test]
    fn match_result_iter() {
        let r = MatchResult::from(vec![1, 2, 3]);
        let collected: Vec<_> = r.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn match_result_into_iter() {
        let r = MatchResult::from(vec![4, 5]);
        let collected: Vec<usize> = r.into_iter().collect();
        assert_eq!(collected, vec![4, 5]);
    }

    #[test]
    fn match_result_ref_into_iter() {
        let r = MatchResult::from(vec![6, 7]);
        let collected: Vec<_> = (&r).into_iter().copied().collect();
        assert_eq!(collected, vec![6, 7]);
    }

    #[test]
    fn match_result_default() {
        let r = MatchResult::default();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn match_result_display() {
        let empty = MatchResult::default();
        assert_eq!(format!("{empty}"), "no matches");

        let matched = MatchResult::from(vec![0, 3]);
        let display = format!("{matched}");
        assert!(display.contains("2 match(es)"));
        assert!(display.contains("[0, 3]"));
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

    // ── Error type tests ──────────────────────────────────────────

    #[test]
    fn matcher_invalid_regex_returns_invalid_pattern() {
        let patterns = vec!["[invalid".to_string()];
        let err = RegexMatcher::new(&patterns).unwrap_err();
        assert_matches!(err, HayaiError::InvalidPattern { .. });
        assert!(err.to_string().contains("invalid regex"));
    }

    #[test]
    fn matcher_accepts_str_slices() {
        let matcher = RegexMatcher::new(["rm", "ls"]).unwrap();
        assert_eq!(matcher.pattern_count(), 2);
    }

    #[test]
    fn matcher_empty_pattern_set() {
        let patterns: Vec<String> = vec![];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.pattern_count(), 0);
        assert!(matcher.check("anything").is_empty());
    }

    // ── PrefixPrefilter edge cases ────────────────────────────────

    #[test]
    fn prefix_prefilter_empty_input() {
        let p = PrefixPrefilter::new(["rm"], 3);
        assert!(p.is_safe(""));
    }

    #[test]
    fn prefix_prefilter_max_words_zero() {
        let p = PrefixPrefilter::new(["rm"], 0);
        assert!(p.is_safe("rm -rf /"));
    }

    #[test]
    fn prefix_prefilter_partial_prefix_match() {
        let p = PrefixPrefilter::new(["git"], 3);
        assert!(!p.is_safe("git-lfs pull"));
    }

    #[test]
    fn prefix_prefilter_word_beyond_max_words() {
        let p = PrefixPrefilter::new(["dangerous"], 2);
        assert!(p.is_safe("safe safe dangerous"));
    }

    #[test]
    fn prefix_prefilter_empty_set() {
        let p = PrefixPrefilter::new(Vec::<String>::new(), 3);
        assert!(p.is_safe("rm -rf /"));
    }

    #[test]
    fn prefix_prefilter_prefix_set_accessor() {
        let p = PrefixPrefilter::new(["rm", "git"], 3);
        assert!(p.prefix_set().contains("rm"));
        assert!(p.prefix_set().contains("git"));
        assert_eq!(p.prefix_set().len(), 2);
    }

    // ── KeywordPrefilter edge cases ───────────────────────────────

    #[test]
    fn keyword_prefilter_empty_input() {
        let p = KeywordPrefilter::new([b"DROP ".to_vec()]);
        assert!(p.is_safe(""));
    }

    #[test]
    fn keyword_prefilter_empty_keywords() {
        let p = KeywordPrefilter::new(Vec::<Vec<u8>>::new());
        assert!(p.is_safe("DROP TABLE users"));
    }

    #[test]
    fn keyword_prefilter_mixed_case_match() {
        let p = KeywordPrefilter::new([b"DELETE".to_vec()]);
        assert!(!p.is_safe("please delete this"));
        assert!(!p.is_safe("please DELETE this"));
        assert!(!p.is_safe("please DeLeTe this"));
    }

    // ── ChainedNormalizer edge cases ──────────────────────────────

    #[test]
    fn chained_normalizer_both_transform() {
        let n = ChainedNormalizer {
            first: PathNormalizer,
            second: FnNormalizer::new(|s| Cow::Owned(s.to_uppercase())),
        };
        let result = n.normalize("/usr/bin/hello world");
        assert_eq!(&*result, "HELLO WORLD");
    }

    #[test]
    fn chained_normalizer_second_only_transforms() {
        let n = ChainedNormalizer {
            first: IdentityNormalizer,
            second: FnNormalizer::new(|s| Cow::Owned(s.to_uppercase())),
        };
        let result = n.normalize("hello");
        assert_eq!(&*result, "HELLO");
    }

    // ── MatchResult clone ─────────────────────────────────────────

    #[test]
    fn match_result_clone() {
        let r = MatchResult::from(vec![1, 2, 3]);
        let cloned = r.clone();
        assert_eq!(r, cloned);
    }

    // ── Mock tests ───────────────────────────────────────────────

    #[test]
    fn mock_match_engine_returns_configured() {
        let mock = MockMatchEngine::new(vec![0, 2], 5);
        assert_eq!(mock.check("anything"), vec![0, 2]);
        assert_eq!(mock.pattern_count(), 5);
    }

    #[test]
    fn mock_match_engine_empty() {
        let mock = MockMatchEngine::empty(10);
        assert!(mock.check("anything").is_empty());
        assert_eq!(mock.pattern_count(), 10);
    }

    #[test]
    fn mock_normalizer_identity() {
        let n = MockNormalizer::identity();
        let result = n.normalize("hello");
        assert!(matches!(result, Cow::Borrowed("hello")));
    }

    #[test]
    fn mock_normalizer_transform() {
        let n = MockNormalizer::with_transform(|s| s.to_uppercase());
        assert_eq!(&*n.normalize("hello"), "HELLO");
    }

    #[test]
    fn mock_prefilter_always_safe() {
        let p = MockPrefilter { always_safe: true };
        assert!(p.is_safe("anything"));
    }

    #[test]
    fn mock_prefilter_never_safe() {
        let p = MockPrefilter { always_safe: false };
        assert!(!p.is_safe("anything"));
    }

    #[test]
    fn mock_prefilter_in_regex_matcher() {
        let matcher = RegexMatcher::with_plugins(
            ["rm"],
            IdentityNormalizer,
            MockPrefilter { always_safe: true },
        )
        .unwrap();
        assert!(matcher.check("rm -rf /").is_empty());
    }

    // ── Normalizer edge cases ────────────────────────────────────

    #[test]
    fn identity_normalizer_empty_string() {
        let n = IdentityNormalizer;
        let result = n.normalize("");
        assert!(matches!(result, Cow::Borrowed("")));
        assert_eq!(&*result, "");
    }

    #[test]
    fn identity_normalizer_unicode() {
        let n = IdentityNormalizer;
        let input = "日本語テスト 🚀 café";
        let result = n.normalize(input);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, input);
    }

    #[test]
    fn identity_normalizer_very_long_string() {
        let n = IdentityNormalizer;
        let input = "a".repeat(100_000);
        let result = n.normalize(&input);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.len(), 100_000);
    }

    #[test]
    fn path_normalizer_empty_string() {
        let n = PathNormalizer;
        let result = n.normalize("");
        assert!(matches!(result, Cow::Borrowed("")));
    }

    #[test]
    fn path_normalizer_unicode_no_path() {
        let n = PathNormalizer;
        let input = "café résumé naïve";
        let result = n.normalize(input);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, input);
    }

    #[test]
    fn path_normalizer_only_path_prefix() {
        let n = PathNormalizer;
        let result = n.normalize("/usr/bin/");
        assert_eq!(&*result, "");
    }

    #[test]
    fn path_normalizer_bin_prefix() {
        let n = PathNormalizer;
        assert_eq!(&*n.normalize("/bin/ls"), "ls");
    }

    // ── ChainedNormalizer edge cases ─────────────────────────────

    #[test]
    fn chained_normalizer_empty_input() {
        let n = ChainedNormalizer {
            first: PathNormalizer,
            second: FnNormalizer::new(|s| Cow::Owned(s.to_uppercase())),
        };
        let result = n.normalize("");
        assert_eq!(&*result, "");
    }

    #[test]
    fn chained_normalizer_second_returns_borrowed_after_first_owned() {
        // First transforms (Owned), second returns Borrowed (no-op).
        // The chained normalizer should still return Owned from first.
        let n = ChainedNormalizer {
            first: PathNormalizer,
            second: IdentityNormalizer,
        };
        let result = n.normalize("/usr/bin/cmd");
        assert_eq!(&*result, "cmd");
        // PathNormalizer produces Owned, IdentityNormalizer borrows from it,
        // but ChainedNormalizer must return the Owned string.
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn chained_normalizer_fn_trim_then_uppercase() {
        let n = ChainedNormalizer {
            first: FnNormalizer::new(|s| {
                let trimmed = s.trim();
                if trimmed.len() == s.len() {
                    Cow::Borrowed(s)
                } else {
                    Cow::Owned(trimmed.to_string())
                }
            }),
            second: FnNormalizer::new(|s| Cow::Owned(s.to_uppercase())),
        };
        assert_eq!(&*n.normalize("  hello  "), "HELLO");
    }

    // ── FnNormalizer edge cases ──────────────────────────────────

    #[test]
    fn fn_normalizer_trim_whitespace() {
        let n = FnNormalizer::new(|s| {
            let trimmed = s.trim();
            if trimmed.len() == s.len() {
                Cow::Borrowed(s)
            } else {
                Cow::Owned(trimmed.to_string())
            }
        });
        assert_eq!(&*n.normalize("  hello  "), "hello");
        // No whitespace — should borrow
        let result = n.normalize("hello");
        assert!(matches!(result, Cow::Borrowed("hello")));
    }

    #[test]
    fn fn_normalizer_replace_pattern() {
        let n = FnNormalizer::new(|s| {
            if s.contains("sudo") {
                Cow::Owned(s.replace("sudo ", ""))
            } else {
                Cow::Borrowed(s)
            }
        });
        assert_eq!(&*n.normalize("sudo rm -rf /"), "rm -rf /");
        assert_eq!(&*n.normalize("ls -la"), "ls -la");
    }

    // ── FnPrefilter edge cases ───────────────────────────────────

    #[test]
    fn fn_prefilter_length_based() {
        // Safe if input is short (< 5 chars)
        let p = FnPrefilter(|s: &str| s.len() < 5);
        assert!(p.is_safe("hi"));
        assert!(p.is_safe(""));
        assert!(!p.is_safe("hello world"));
    }

    #[test]
    fn fn_prefilter_empty_input() {
        let p = FnPrefilter(|s: &str| s.is_empty());
        assert!(p.is_safe(""));
        assert!(!p.is_safe("x"));
    }

    #[test]
    fn fn_prefilter_in_composite() {
        let composite = CompositePrefilter {
            first: FnPrefilter(|s: &str| s.starts_with("safe_")),
            second: FnPrefilter(|s: &str| s.len() < 100),
        };
        assert!(composite.is_safe("safe_short"));
        assert!(!composite.is_safe("dangerous_short"));
        assert!(!composite.is_safe(&format!("safe_{}", "x".repeat(100))));
    }

    // ── KeywordPrefilter edge cases ──────────────────────────────

    #[test]
    fn keyword_prefilter_single_byte_keyword() {
        let p = KeywordPrefilter::new([b"X".to_vec()]);
        assert!(!p.is_safe("exec"));
        assert!(!p.is_safe("EXEC"));
        assert!(p.is_safe("hello"));
    }

    #[test]
    fn keyword_prefilter_keyword_at_start() {
        let p = KeywordPrefilter::new([b"DROP".to_vec()]);
        assert!(!p.is_safe("DROP TABLE users"));
    }

    #[test]
    fn keyword_prefilter_keyword_at_end() {
        let p = KeywordPrefilter::new([b"DROP".to_vec()]);
        assert!(!p.is_safe("please DROP"));
    }

    #[test]
    fn keyword_prefilter_multiple_keywords() {
        let p = KeywordPrefilter::new([
            b"DROP".to_vec(),
            b"DELETE".to_vec(),
            b"TRUNCATE".to_vec(),
        ]);
        assert!(!p.is_safe("delete from users"));
        assert!(!p.is_safe("truncate table"));
        assert!(p.is_safe("select * from users"));
    }

    // ── PrefixPrefilter edge cases ───────────────────────────────

    #[test]
    fn prefix_prefilter_whitespace_only_input() {
        let p = PrefixPrefilter::new(["rm"], 3);
        assert!(p.is_safe("   "));
    }

    #[test]
    fn prefix_prefilter_single_word_input() {
        let p = PrefixPrefilter::new(["rm"], 3);
        assert!(!p.is_safe("rm"));
        assert!(p.is_safe("ls"));
    }

    #[test]
    fn prefix_prefilter_max_words_one() {
        let p = PrefixPrefilter::new(["git"], 1);
        assert!(!p.is_safe("git push"));
        assert!(p.is_safe("cargo git push")); // "git" is second word
    }

    // ── RegexMatcher combined normalizer + prefilter ─────────────

    #[test]
    fn matcher_normalizer_plus_prefilter_pipeline() {
        let patterns = vec![r"^rm\s+-rf".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            PathNormalizer,
            PrefixPrefilter::new(["rm"], 3),
        )
        .unwrap();
        // Path normalized then prefilter checks — "rm" is dangerous prefix
        // so prefilter does NOT skip, and regex matches
        // BUT: the prefilter sees the RAW normalized output. Let's check:
        // Input: "/usr/bin/rm -rf /" → normalized to "rm -rf /"
        // Prefilter sees "rm -rf /" → prefix "rm" matches → NOT safe → DFA runs → match
        assert_eq!(matcher.check("/usr/bin/rm -rf /"), vec![0]);

        // Input: "ls -la" → normalized to "ls -la"
        // Prefilter sees "ls -la" → no prefix match → safe → skip DFA
        assert!(matcher.check("ls -la").is_empty());
    }

    #[test]
    fn matcher_fn_normalizer_plus_fn_prefilter() {
        let patterns = vec![r"DANGER".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            FnNormalizer::new(|s| Cow::Owned(s.to_uppercase())),
            FnPrefilter(|s: &str| !s.contains("DANGER")),
        )
        .unwrap();
        // "this is danger" → normalized to "THIS IS DANGER"
        // Prefilter: "THIS IS DANGER" contains "DANGER" → not safe → DFA → match
        assert_eq!(matcher.check("this is danger"), vec![0]);
        // "hello" → "HELLO" → prefilter: no DANGER → safe → skip
        assert!(matcher.check("hello").is_empty());
    }

    #[test]
    fn matcher_case_insensitive_pattern() {
        let patterns = vec![r"(?i)drop\s+table".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.check("DROP TABLE users"), vec![0]);
        assert_eq!(matcher.check("drop table users"), vec![0]);
        assert_eq!(matcher.check("Drop Table users"), vec![0]);
    }

    #[test]
    fn matcher_overlapping_patterns() {
        let patterns = vec![
            r"rm".to_string(),
            r"rm\s+-rf".to_string(),
            r"rm\s+-rf\s+/".to_string(),
        ];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        let matches = matcher.check("rm -rf /");
        assert_eq!(matches, vec![0, 1, 2]);
    }

    #[test]
    fn matcher_single_char_patterns() {
        let patterns = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.check("a"), vec![0]);
        assert_eq!(matcher.check("abc"), vec![0, 1, 2]);
        assert!(matcher.check("xyz").is_empty());
    }

    #[test]
    fn matcher_anchored_patterns_no_false_positives() {
        let patterns = vec![r"^rm\b".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.check("rm -rf /"), vec![0]);
        assert!(matcher.check("cargo rm -rf /").is_empty()); // rm not at start
    }

    #[test]
    fn matcher_very_long_input() {
        let patterns = vec![r"needle".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        let long_input = format!("{}needle{}", "a".repeat(50_000), "b".repeat(50_000));
        assert_eq!(matcher.check(&long_input), vec![0]);
    }

    #[test]
    fn matcher_all_patterns_match() {
        let patterns = vec![r"a".to_string(), r"b".to_string(), r"c".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        assert_eq!(matcher.check("abc"), vec![0, 1, 2]);
    }

    #[test]
    fn matcher_prefilter_skips_all() {
        // Always-safe prefilter means DFA never runs
        let patterns = vec![r"rm\s+-rf".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            IdentityNormalizer,
            FnPrefilter(|_: &str| true), // always safe
        )
        .unwrap();
        // Even though "rm -rf /" would match the regex, prefilter skips it
        assert!(matcher.check("rm -rf /").is_empty());
    }

    #[test]
    fn matcher_with_mock_normalizer() {
        let patterns = vec![r"HELLO".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            MockNormalizer::with_transform(|s| s.to_uppercase()),
            NullPrefilter,
        )
        .unwrap();
        assert_eq!(matcher.check("hello"), vec![0]);
    }

    #[test]
    fn matcher_with_mock_prefilter_safe() {
        let patterns = vec![r"rm".to_string()];
        let matcher = RegexMatcher::with_plugins(
            &patterns,
            IdentityNormalizer,
            MockPrefilter { always_safe: true },
        )
        .unwrap();
        assert!(matcher.check("rm -rf /").is_empty());
    }

    // ── MatchResult edge cases ───────────────────────────────────

    #[test]
    fn match_result_single_match() {
        let r = MatchResult::from(vec![42]);
        assert!(!r.is_empty());
        assert!(r.matched_any());
        assert_eq!(r.first(), Some(42));
        assert_eq!(r.len(), 1);
        assert!(r.contains(42));
        assert!(!r.contains(0));
    }

    #[test]
    fn match_result_contains_all_indices() {
        let r = MatchResult::from(vec![0, 5, 10, 15, 20]);
        for &i in &[0, 5, 10, 15, 20] {
            assert!(r.contains(i));
        }
        for &i in &[1, 2, 3, 4, 6, 11, 21] {
            assert!(!r.contains(i));
        }
    }

    #[test]
    fn match_result_display_single() {
        let r = MatchResult::from(vec![0]);
        let display = format!("{r}");
        assert!(display.contains("1 match(es)"));
    }

    #[test]
    fn match_result_iter_empty() {
        let r = MatchResult::default();
        assert_eq!(r.iter().count(), 0);
    }

    #[test]
    fn match_result_into_iter_ref_preserves_original() {
        let r = MatchResult::from(vec![1, 2, 3]);
        let _ = (&r).into_iter().count();
        // r is still usable
        assert_eq!(r.len(), 3);
    }

    // ── MockMatchEngine edge cases ───────────────────────────────

    #[test]
    fn mock_match_engine_clone() {
        let mock = MockMatchEngine::new(vec![1, 2], 5);
        let cloned = mock.clone();
        assert_eq!(cloned.check("x"), vec![1, 2]);
        assert_eq!(cloned.pattern_count(), 5);
    }

    #[test]
    fn mock_match_engine_zero_patterns() {
        let mock = MockMatchEngine::empty(0);
        assert!(mock.check("anything").is_empty());
        assert_eq!(mock.pattern_count(), 0);
    }

    // ── CompositePrefilter with FnPrefilter ──────────────────────

    #[test]
    fn composite_fn_prefilters() {
        let composite = CompositePrefilter {
            first: FnPrefilter(|s: &str| !s.contains("rm")),
            second: FnPrefilter(|s: &str| !s.contains("sudo")),
        };
        assert!(composite.is_safe("ls -la"));
        assert!(!composite.is_safe("rm -rf /"));
        assert!(!composite.is_safe("sudo ls"));
        assert!(!composite.is_safe("sudo rm -rf /"));
    }

    // ── MatchEngine trait object tests ───────────────────────────

    #[test]
    fn match_engine_trait_object_usage() {
        let patterns = vec![r"test".to_string()];
        let matcher = RegexMatcher::new(&patterns).unwrap();
        // Use as dyn MatchEngine
        let engine: &dyn MatchEngine = &matcher;
        assert_eq!(engine.check("this is a test"), vec![0]);
        assert_eq!(engine.pattern_count(), 1);
    }

    // ── contains_ascii_ci additional tests ───────────────────────

    #[test]
    fn contains_ascii_ci_exact_match() {
        assert!(contains_ascii_ci(b"DROP", b"DROP"));
    }

    #[test]
    fn contains_ascii_ci_needle_longer_than_haystack() {
        assert!(!contains_ascii_ci(b"DR", b"DROP"));
    }

    #[test]
    fn contains_ascii_ci_single_char() {
        assert!(contains_ascii_ci(b"hello", b"H"));
        assert!(contains_ascii_ci(b"hello", b"E"));
        assert!(!contains_ascii_ci(b"hello", b"X"));
    }

    // ── proptest ──────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn contains_ascii_ci_never_panics(
                haystack in ".*",
                needle in "[A-Z ]{0,10}"
            ) {
                let _ = contains_ascii_ci(haystack.as_bytes(), needle.as_bytes());
            }

            #[test]
            fn contains_ascii_ci_agrees_with_naive(
                haystack in "[a-zA-Z0-9 ]{0,50}",
                needle in "[A-Z]{1,5}"
            ) {
                let expected = haystack.to_uppercase().contains(&needle);
                let result = contains_ascii_ci(haystack.as_bytes(), needle.as_bytes());
                prop_assert_eq!(result, expected);
            }

            #[test]
            fn identity_normalizer_always_borrows(input in ".*") {
                let n = IdentityNormalizer;
                let result = n.normalize(&input);
                prop_assert!(matches!(result, Cow::Borrowed(_)));
                prop_assert_eq!(&*result, &input);
            }

            #[test]
            fn path_normalizer_idempotent(input in ".*") {
                let n = PathNormalizer;
                let once = n.normalize(&input).into_owned();
                let twice = n.normalize(&once).into_owned();
                prop_assert_eq!(once, twice);
            }

            #[test]
            fn null_prefilter_always_unsafe(input in ".*") {
                let p = NullPrefilter;
                prop_assert!(!p.is_safe(&input));
            }

            #[test]
            fn matcher_never_panics_on_arbitrary_input(input in "\\PC{0,200}") {
                let patterns = vec![r"rm\s+-rf".to_string(), r"DROP\s+TABLE".to_string()];
                let matcher = RegexMatcher::new(&patterns).unwrap();
                let _ = matcher.check(&input);
            }

            #[test]
            fn match_result_len_equals_indices_len(indices in collection::vec(0..100usize, 0..20)) {
                let r = MatchResult::from(indices.clone());
                prop_assert_eq!(r.len(), indices.len());
                prop_assert_eq!(r.is_empty(), indices.is_empty());
                prop_assert_eq!(r.matched_any(), !indices.is_empty());
                prop_assert_eq!(r.first(), indices.first().copied());
            }

            #[test]
            fn prefix_prefilter_safe_implies_no_prefix_match(
                prefix in "[a-z]{1,5}",
                input in "[a-z]{6,20}"
            ) {
                let p = PrefixPrefilter::new([prefix.clone()], 1);
                if p.is_safe(&input) {
                    let first_word = input.split_whitespace().next().unwrap_or("");
                    prop_assert!(!first_word.starts_with(&prefix));
                }
            }
        }
    }
}
