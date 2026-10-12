use super::*;
use crate::cache::MemCache;
use crate::engine::RegexMatcher;
use proptest::prelude::*;

const PATTERNS: &[&str] = &[
    r"\brm\s+-rf\s+/",
    r"(?i)drop\s+table",
    r"(?i)\bkubectl\s+delete\s+(ns|namespace)\b",
    r"git\s+push\s+.*--force",
    r"^\s*sudo\b",
    r"(?i)terraform\s+(destroy|apply\s+-auto-approve)",
    r"[a-z]+\.prod\.example",
    r".*",
    r"(?i)ſecret",
    r"\x{212A}ill",
    r"aws\s+s3\s+rm\s+.*--recursive",
    r"(?i-u)by[t]e",
    r"(?i)élan",
    r"a{3}b+",
    r"(foo|bar)baz",
];

fn eager(input: &str) -> Vec<usize> {
    RegexMatcher::new(PATTERNS).unwrap().check(input)
}

fn lazy(input: &str) -> Vec<usize> {
    LazyRegexMatcher::new(PATTERNS).unwrap().check(input)
}

#[test]
fn lazy_agrees_with_eager_on_witnesses() {
    for input in [
        "rm -rf /",
        "RM -RF /",
        "echo DROP   TABLE users",
        "kubectl delete NS kube-system",
        "KUBECTL DELETE namespace x",
        "git push origin main --force",
        "  sudo ls",
        "TERRAFORM destroy",
        "api.prod.example",
        "SECRET",
        "\u{17F}ecret",
        "\u{212A}ill",
        "kill",
        "aws s3 rm s3://b --recursive",
        "ÉLAN",
        "aaab",
        "barbaz",
        "",
        "ls -la",
    ] {
        assert_eq!(lazy(input), eager(input), "{input:?}");
    }
}

#[test]
fn kelvin_and_long_s_fold_onto_ascii_needles() {
    let m = LazyRegexMatcher::new([r"(?i)kill\s+-9", r"(?i)ssh\b"]).unwrap();
    assert_eq!(m.check("\u{212A}ILL -9 1"), vec![0]);
    assert_eq!(m.check("\u{17F}sh host"), vec![1]);
    assert!(
        m.index()
            .entries
            .iter()
            .all(|e| e.as_ref().is_some_and(|n| !n.folded))
    );
}

#[test]
fn unindexable_patterns_are_always_candidates() {
    let index = LiteralIndex::build(&[r".*", r"\d+", r"(a|.)b?"]).unwrap();
    assert_eq!(index.unindexed(), 3);
}

#[test]
fn needles_are_required_substrings() {
    let index = LiteralIndex::build(&[r"\baws\s+s3api\s+delete-bucket\b"]).unwrap();
    let needles = index.entries[0].clone().unwrap();
    assert_eq!(needles.any_of, vec!["delete-bucket".to_string()]);
}

#[test]
fn invalid_pattern_is_rejected() {
    assert!(matches!(
        LiteralIndex::build(&["[unclosed"]),
        Err(HayaiError::InvalidPattern { .. })
    ));
    assert!(LazyRegexMatcher::new(["[unclosed"]).is_err());
}

#[test]
fn stale_index_is_rebuilt() {
    let stale = LiteralIndex::build(&["foo"]).unwrap();
    let m = LazyRegexMatcher::with_index([r"rm\s+-rf"], stale, IdentityNormalizer, NullPrefilter)
        .unwrap();
    assert_eq!(m.check("rm -rf /"), vec![0]);
    assert!(m.index().fits(&[r"rm\s+-rf"]));
}

#[test]
fn index_with_wrong_length_does_not_fit() {
    let mut index = LiteralIndex::build(&["a", "b"]).unwrap();
    index.entries.pop();
    assert!(!index.fits(&["a", "b"]));
}

#[test]
fn digest_tracks_pattern_boundaries() {
    assert_ne!(
        LiteralIndex::digest_of(&["ab", "c"]),
        LiteralIndex::digest_of(&["a", "bc"])
    );
}

#[test]
fn index_round_trips_through_json() {
    let index = LiteralIndex::build(PATTERNS).unwrap();
    let back: LiteralIndex = serde_json::from_slice(&serde_json::to_vec(&index).unwrap()).unwrap();
    assert_eq!(back, index);
}

#[test]
fn long_haystacks_scan_through_the_automaton() {
    let mut input = "x".repeat(SCAN_LIMIT + 1);
    input.push_str(" DROP TABLE t");
    assert_eq!(lazy(&input), eager(&input));
    assert!(lazy(&input).contains(&1));
}

#[test]
fn through_serves_a_fitting_index_from_the_store() {
    let store: MemCache<LiteralIndex> = MemCache::empty();
    let built = LiteralIndex::through(&store, PATTERNS).unwrap();
    assert_eq!(store.load().unwrap().0, LiteralIndex::digest_of(PATTERNS));
    assert_eq!(LiteralIndex::through(&store, PATTERNS).unwrap(), built);
}

#[test]
fn through_rebuilds_an_index_that_does_not_fit() {
    let store: MemCache<LiteralIndex> = MemCache::empty();
    let mut wrong = LiteralIndex::build(&["unrelated"]).unwrap();
    wrong.digest = LiteralIndex::digest_of(PATTERNS);
    store
        .save(LiteralIndex::digest_of(PATTERNS), &wrong)
        .unwrap();
    let index = LiteralIndex::through(&store, PATTERNS).unwrap();
    assert!(index.fits(PATTERNS));
    assert!(store.load().unwrap().1.fits(PATTERNS));
}

#[test]
fn through_works_when_the_store_cannot_save() {
    let store = crate::cache::FsCache::new("/dev/null/literal-index.json");
    assert!(
        LiteralIndex::through(&store, PATTERNS)
            .unwrap()
            .fits(PATTERNS)
    );
}

proptest! {
    #[test]
    fn lazy_agrees_with_eager(
        input in "(?s)(rm|RM|-rf|/| |drop|TABLE|kubectl|delete|ns|git|push|--force|sudo|\u{17F}|\u{212A}|élan|ÉLAN|aaa|b|foo|bar|baz|aws|s3|--recursive|terraform|destroy|prod\\.example|x|\n){0,12}"
    ) {
        prop_assert_eq!(lazy(&input), eager(&input));
    }
}
