use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::{env, fs};

use serde::{de::DeserializeOwned, Serialize};

/// Trait for cache storage — abstracts filesystem for testability.
///
/// Generic over the cached data type.
pub trait CacheStore<T: Serialize + DeserializeOwned> {
    fn load(&self) -> Option<(u64, T)>;
    fn save(&self, fingerprint: u64, data: &T) -> anyhow::Result<()>;
}

/// Trait for fingerprinting — abstracts input to a hash.
pub trait Fingerprinter {
    fn fingerprint(&self) -> u64;
}

// ═══════════════════════════════════════════════════════════════════
// Filesystem implementations
// ═══════════════════════════════════════════════════════════════════

/// JSON entry stored in the cache file.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CacheEntry<T> {
    fingerprint: u64,
    data: T,
}

/// Filesystem-backed cache at a configurable path.
pub struct FsCache {
    pub path: PathBuf,
}

impl FsCache {
    /// Create a cache at `~/.cache/{app_name}/compiled.json`.
    #[must_use]
    pub fn for_app(app_name: &str) -> Self {
        Self {
            path: env::var("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from(env::var("HOME").unwrap_or_default()).join(".cache")
                })
                .join(app_name)
                .join("compiled.json"),
        }
    }
}

impl<T: Serialize + DeserializeOwned> CacheStore<T> for FsCache {
    fn load(&self) -> Option<(u64, T)> {
        let content = fs::read(&self.path).ok()?;
        let entry: CacheEntry<T> = serde_json::from_slice(&content).ok()?;
        Some((entry.fingerprint, entry.data))
    }

    fn save(&self, fingerprint: u64, data: &T) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let entry = CacheEntry { fingerprint, data };
        fs::write(&self.path, serde_json::to_vec(&entry)?)?;
        Ok(())
    }
}

/// Fingerprint based on file modification times.
pub struct FsFingerprinter {
    pub paths: Vec<PathBuf>,
}

impl Fingerprinter for FsFingerprinter {
    fn fingerprint(&self) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        for path in &self.paths {
            if let Ok(meta) = fs::metadata(path) {
                if let Ok(mtime) = meta.modified() {
                    mtime_nanos(mtime).hash(&mut hasher);
                }
            }
            // Also scan directories
            if path.is_dir() {
                if let Ok(entries) = fs::read_dir(path) {
                    for entry in entries.flatten() {
                        if let Ok(meta) = entry.metadata() {
                            if let Ok(mtime) = meta.modified() {
                                mtime_nanos(mtime).hash(&mut hasher);
                            }
                        }
                    }
                }
            }
        }
        hasher.finish()
    }
}

fn mtime_nanos(t: std::time::SystemTime) -> u64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════
// In-memory implementations (for testing)
// ═══════════════════════════════════════════════════════════════════

/// In-memory cache for testing.
pub struct MemCache<T> {
    data: std::cell::RefCell<Option<(u64, T)>>,
}

impl<T> MemCache<T> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: std::cell::RefCell::new(None),
        }
    }
}

impl<T: Serialize + DeserializeOwned + Clone> CacheStore<T> for MemCache<T> {
    fn load(&self) -> Option<(u64, T)> {
        self.data.borrow().clone()
    }

    fn save(&self, fingerprint: u64, data: &T) -> anyhow::Result<()> {
        *self.data.borrow_mut() = Some((fingerprint, data.clone()));
        Ok(())
    }
}

/// Fixed fingerprint for testing.
pub struct FixedFingerprinter(pub u64);

impl Fingerprinter for FixedFingerprinter {
    fn fingerprint(&self) -> u64 {
        self.0
    }
}

// ═══════════════════════════════════════════════════════════════════
// Resolver: cache-aware resolution
// ═══════════════════════════════════════════════════════════════════

/// Resolve data with caching. Try cache first, fall back to resolver
/// function, auto-populate cache on miss.
///
/// # Errors
/// Returns an error if the resolve function fails.
pub fn resolve_cached<T: Serialize + DeserializeOwned>(
    cache: &dyn CacheStore<T>,
    fp: &dyn Fingerprinter,
    resolve_fn: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let current_fp = fp.fingerprint();

    // Cache hit
    if let Some((cached_fp, data)) = cache.load() {
        if cached_fp == current_fp {
            return Ok(data);
        }
    }

    // Cache miss — resolve and save
    let data = resolve_fn()?;
    let _ = cache.save(current_fp, &data);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_miss_resolves_and_saves() {
        let cache = MemCache::empty();
        let fp = FixedFingerprinter(42);
        let data: Vec<String> =
            resolve_cached(&cache, &fp, || Ok(vec!["test".to_string()])).unwrap();
        assert_eq!(data.len(), 1);
        assert!(cache.load().is_some());
        assert_eq!(cache.load().unwrap().0, 42);
    }

    #[test]
    fn cache_hit_skips_resolution() {
        let cache = MemCache::empty();
        let fp = FixedFingerprinter(42);
        cache.save(42, &vec!["cached".to_string()]).unwrap();
        let data: Vec<String> = resolve_cached(&cache, &fp, || {
            panic!("should not be called on cache hit");
        })
        .unwrap();
        assert_eq!(data, vec!["cached"]);
    }

    #[test]
    fn stale_cache_resolves_fresh() {
        let cache = MemCache::empty();
        let fp = FixedFingerprinter(99);
        cache.save(42, &vec!["old".to_string()]).unwrap();
        let data: Vec<String> =
            resolve_cached(&cache, &fp, || Ok(vec!["new".to_string()])).unwrap();
        assert_eq!(data, vec!["new"]);
        assert_eq!(cache.load().unwrap().0, 99);
    }

    #[test]
    fn mem_cache_empty_returns_none() {
        let cache: MemCache<Vec<String>> = MemCache::empty();
        assert!(cache.load().is_none());
    }

    #[test]
    fn fixed_fingerprinter() {
        let fp = FixedFingerprinter(12345);
        assert_eq!(fp.fingerprint(), 12345);
    }

    #[test]
    fn fs_cache_for_app() {
        let cache = FsCache::for_app("test-app");
        assert!(cache.path.to_str().unwrap().contains("test-app"));
        assert!(cache.path.to_str().unwrap().contains("compiled.json"));
    }
}
