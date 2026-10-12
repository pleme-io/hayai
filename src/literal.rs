use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use regex::RegexSetBuilder;
use regex_syntax::hir::{Class, Hir, HirKind};
use serde::{Deserialize, Serialize};

use crate::cache::CacheStore;
use crate::engine::{IdentityNormalizer, MatchEngine, Normalizer, NullPrefilter, Prefilter};
use crate::error::HayaiError;

const INDEX_VERSION: u64 = 3;
const MAX_ALTERNATES: usize = 64;
const MAX_CLASS_CHARS: u32 = 8;
const SCAN_LIMIT: usize = 16 * 1024;
const SIZE_LIMIT: usize = 100 * 1024 * 1024;
const FOLDS: [(char, u8); 2] = [('\u{212A}', b'k'), ('\u{17F}', b's')];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Needles {
    pub any_of: Vec<String>,
    pub folded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiteralIndex {
    pub digest: u64,
    pub entries: Vec<Option<Needles>>,
}

impl LiteralIndex {
    pub fn build<S: AsRef<str>>(patterns: &[S]) -> Result<Self, HayaiError> {
        let mut entries = Vec::with_capacity(patterns.len());
        for pattern in patterns {
            let pattern = pattern.as_ref();
            let Ok(hir) = regex_syntax::parse(pattern) else {
                regex::Regex::new(pattern)?;
                entries.push(None);
                continue;
            };
            entries.push(needles(&hir));
        }
        Ok(Self {
            digest: Self::digest_of(patterns),
            entries,
        })
    }

    #[must_use]
    pub fn digest_of<S: AsRef<str>>(patterns: &[S]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325 ^ INDEX_VERSION;
        let mut feed = |byte: u8| {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        };
        for pattern in patterns {
            pattern.as_ref().bytes().for_each(&mut feed);
            feed(0xff);
        }
        hash
    }

    #[must_use]
    pub fn fits<S: AsRef<str>>(&self, patterns: &[S]) -> bool {
        self.entries.len() == patterns.len() && self.digest == Self::digest_of(patterns)
    }

    pub fn through<S: AsRef<str>>(
        store: &dyn CacheStore<Self>,
        patterns: &[S],
    ) -> Result<Self, HayaiError> {
        let digest = Self::digest_of(patterns);
        if let Some((fingerprint, index)) = store.load()
            && fingerprint == digest
            && index.fits(patterns)
        {
            return Ok(index);
        }
        let index = Self::build(patterns)?;
        let _ = store.save(digest, &index);
        Ok(index)
    }

    #[must_use]
    pub fn unindexed(&self) -> usize {
        self.entries.iter().filter(|e| e.is_none()).count()
    }
}

fn needles(hir: &Hir) -> Option<Needles> {
    let mut folded = false;
    let any_of = required(hir, &mut folded)?;
    if any_of.iter().any(Vec::is_empty) {
        return None;
    }
    let any_of: BTreeSet<String> = any_of
        .into_iter()
        .map(|bytes| String::from_utf8(bytes).map(|lit| canonical(&lit)))
        .collect::<Result<_, _>>()
        .ok()?;
    Some(Needles {
        any_of: any_of.into_iter().collect(),
        folded,
    })
}

fn canonical(text: &str) -> String {
    text.chars()
        .map(|c| {
            FOLDS
                .iter()
                .find(|(wide, _)| *wide == c)
                .map_or(c.to_ascii_lowercase(), |(_, ascii)| char::from(*ascii))
        })
        .collect()
}

fn atom(hir: &Hir, folded: &mut bool) -> Option<Vec<u8>> {
    match hir.kind() {
        HirKind::Literal(lit) => Some(lit.0.to_ascii_lowercase()),
        HirKind::Class(Class::Unicode(class)) => {
            let total: u32 = class
                .ranges()
                .iter()
                .map(|r| u32::from(r.end()) - u32::from(r.start()) + 1)
                .sum();
            if total > MAX_CLASS_CHARS {
                return None;
            }
            let chars: Vec<char> = class
                .ranges()
                .iter()
                .flat_map(|r| r.start()..=r.end())
                .collect();
            let ascii: BTreeSet<u8> = chars
                .iter()
                .filter(|c| c.is_ascii())
                .map(|c| (*c as u8).to_ascii_lowercase())
                .collect();
            let wide: Vec<char> = chars.iter().copied().filter(|c| !c.is_ascii()).collect();
            match (ascii.len(), wide.len()) {
                (1, 0) => ascii.first().map(|b| vec![*b]),
                (1, _) => {
                    let letter = ascii.first().copied()?;
                    if !wide.iter().all(|w| FOLDS.contains(&(*w, letter))) {
                        *folded = true;
                    }
                    Some(vec![letter])
                }
                (0, 1) => chars.first().map(|c| c.to_string().into_bytes()),
                _ => None,
            }
        }
        HirKind::Class(Class::Bytes(class)) => {
            let bytes: BTreeSet<u8> = class
                .ranges()
                .iter()
                .flat_map(|r| r.start()..=r.end())
                .map(|b| b.to_ascii_lowercase())
                .collect();
            if bytes.len() == 1 {
                bytes.first().map(|b| vec![*b])
            } else {
                None
            }
        }
        _ => None,
    }
}

fn score(set: &[Vec<u8>]) -> (usize, std::cmp::Reverse<usize>) {
    (
        set.iter().map(Vec::len).min().unwrap_or(0),
        std::cmp::Reverse(set.len()),
    )
}

fn better(best: Option<Vec<Vec<u8>>>, next: Vec<Vec<u8>>) -> Option<Vec<Vec<u8>>> {
    match best {
        Some(b) if score(&b) >= score(&next) => Some(b),
        _ => Some(next),
    }
}

fn required(hir: &Hir, folded: &mut bool) -> Option<Vec<Vec<u8>>> {
    match hir.kind() {
        HirKind::Literal(_) | HirKind::Class(_) => atom(hir, folded).map(|a| vec![a]),
        HirKind::Capture(cap) => required(&cap.sub, folded),
        HirKind::Repetition(rep) if rep.min >= 1 => required(&rep.sub, folded),
        HirKind::Alternation(branches) => {
            let mut set = BTreeSet::new();
            for branch in branches {
                set.extend(required(branch, folded)?);
                if set.len() > MAX_ALTERNATES {
                    return None;
                }
            }
            Some(set.into_iter().collect())
        }
        HirKind::Concat(children) => {
            let mut best = None;
            let mut run: Vec<u8> = Vec::new();
            for child in children {
                if matches!(child.kind(), HirKind::Look(_)) {
                    continue;
                }
                if let Some(bytes) = atom(child, folded) {
                    run.extend(bytes);
                    continue;
                }
                if !run.is_empty() {
                    best = better(best, vec![std::mem::take(&mut run)]);
                }
                if let Some(set) = required(child, folded) {
                    best = better(best, set);
                }
            }
            if !run.is_empty() {
                best = better(best, vec![run]);
            }
            best
        }
        _ => None,
    }
}

pub struct LazyRegexMatcher<N: Normalizer = IdentityNormalizer, P: Prefilter = NullPrefilter> {
    patterns: Vec<String>,
    index: LiteralIndex,
    literals: Vec<String>,
    automaton: OnceLock<Option<AhoCorasick>>,
    owners: Vec<Vec<usize>>,
    always: Vec<usize>,
    folded: Vec<usize>,
    normalizer: N,
    prefilter: P,
}

impl<N: Normalizer + fmt::Debug, P: Prefilter + fmt::Debug> fmt::Debug for LazyRegexMatcher<N, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyRegexMatcher")
            .field("pattern_count", &self.patterns.len())
            .field("literal_count", &self.literals.len())
            .field("unindexed", &self.always.len())
            .field("normalizer", &self.normalizer)
            .field("prefilter", &self.prefilter)
            .finish_non_exhaustive()
    }
}

impl LazyRegexMatcher {
    pub fn new<I, S>(patterns: I) -> Result<Self, HayaiError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::with_plugins(patterns, IdentityNormalizer, NullPrefilter)
    }
}

impl<N: Normalizer, P: Prefilter> LazyRegexMatcher<N, P> {
    pub fn with_plugins<I, S>(patterns: I, normalizer: N, prefilter: P) -> Result<Self, HayaiError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let patterns: Vec<String> = patterns
            .into_iter()
            .map(|s| s.as_ref().to_owned())
            .collect();
        let index = LiteralIndex::build(&patterns)?;
        Ok(Self::assemble(patterns, index, normalizer, prefilter))
    }

    pub fn with_index<I, S>(
        patterns: I,
        index: LiteralIndex,
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
        let index = if index.fits(&patterns) {
            index
        } else {
            LiteralIndex::build(&patterns)?
        };
        Ok(Self::assemble(patterns, index, normalizer, prefilter))
    }

    fn assemble(patterns: Vec<String>, index: LiteralIndex, normalizer: N, prefilter: P) -> Self {
        let mut literals: Vec<String> = Vec::new();
        let mut owners: Vec<Vec<usize>> = Vec::new();
        let mut slot: HashMap<&str, usize> = HashMap::new();
        let mut always = Vec::new();
        let mut folded = Vec::new();
        for (i, entry) in index.entries.iter().enumerate() {
            let Some(needles) = entry else {
                always.push(i);
                continue;
            };
            if needles.folded {
                folded.push(i);
            }
            for lit in &needles.any_of {
                let id = *slot.entry(lit.as_str()).or_insert_with(|| {
                    literals.push(lit.clone());
                    owners.push(Vec::new());
                    owners.len() - 1
                });
                owners[id].push(i);
            }
        }
        Self {
            patterns,
            index,
            literals,
            automaton: OnceLock::new(),
            owners,
            always,
            folded,
            normalizer,
            prefilter,
        }
    }

    #[must_use]
    pub fn index(&self) -> &LiteralIndex {
        &self.index
    }

    fn automaton(&self) -> Option<&AhoCorasick> {
        self.automaton
            .get_or_init(|| {
                AhoCorasick::builder()
                    .kind(Some(AhoCorasickKind::NoncontiguousNFA))
                    .match_kind(MatchKind::Standard)
                    .build(&self.literals)
                    .ok()
            })
            .as_ref()
    }

    #[must_use]
    pub fn candidates(&self, haystack: &str) -> Vec<usize> {
        let mut hit = vec![false; self.patterns.len()];
        for &i in &self.always {
            hit[i] = true;
        }
        if !haystack.is_ascii() {
            for &i in &self.folded {
                hit[i] = true;
            }
        }
        let scanned = canonical(haystack);
        let mut mark = |id: usize| {
            for &i in &self.owners[id] {
                hit[i] = true;
            }
        };
        match (scanned.len() > SCAN_LIMIT)
            .then(|| self.automaton())
            .flatten()
        {
            Some(ac) => ac
                .find_overlapping_iter(scanned.as_str())
                .for_each(|m| mark(m.pattern().as_usize())),
            None => self
                .literals
                .iter()
                .enumerate()
                .filter(|(_, lit)| scanned.contains(lit.as_str()))
                .for_each(|(id, _)| mark(id)),
        }
        hit.iter()
            .enumerate()
            .filter_map(|(i, h)| h.then_some(i))
            .collect()
    }

    fn matches(&self, haystack: &str, candidates: &[usize]) -> Vec<usize> {
        let built = RegexSetBuilder::new(candidates.iter().map(|&i| &self.patterns[i]))
            .size_limit(SIZE_LIMIT)
            .build();
        match built {
            Ok(set) => set
                .matches(haystack)
                .iter()
                .map(|j| candidates[j])
                .collect(),
            Err(_) => candidates
                .iter()
                .copied()
                .filter(|&i| {
                    regex::RegexBuilder::new(&self.patterns[i])
                        .size_limit(SIZE_LIMIT)
                        .build()
                        .map_or(true, |re| re.is_match(haystack))
                })
                .collect(),
        }
    }
}

impl<N: Normalizer, P: Prefilter> MatchEngine for LazyRegexMatcher<N, P> {
    fn check(&self, input: &str) -> Vec<usize> {
        let normalized = self.normalizer.normalize(input);
        if self.prefilter.is_safe(&normalized) {
            return Vec::new();
        }
        let candidates = self.candidates(&normalized);
        if candidates.is_empty() {
            return Vec::new();
        }
        self.matches(&normalized, &candidates)
    }

    fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests;
