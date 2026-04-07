use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Mutex;
use std::{env, fs};

use serde::{de::DeserializeOwned, Serialize};

use crate::error::HayaiError;

/// Trait for cache storage — abstracts filesystem for testability.
///
/// Generic over the cached data type. `Send + Sync` required for daemon use.
pub trait CacheStore<T: Serialize + DeserializeOwned>: Send + Sync {
    /// Load the cached data along with its fingerprint, if present.
    fn load(&self) -> Option<(u64, T)>;
    /// Persist data under the given fingerprint.
    fn save(&self, fingerprint: u64, data: &T) -> Result<(), HayaiError>;
}

/// Trait for fingerprinting — abstracts input to a hash.
pub trait Fingerprinter: Send + Sync {
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
    ///
    /// Respects `XDG_CACHE_HOME` when set, otherwise falls back to `$HOME/.cache`.
    #[must_use]
    pub fn for_app(app_name: &str) -> Self {
        let base = match env::var("XDG_CACHE_HOME") {
            Ok(dir) => PathBuf::from(dir),
            Err(_) => PathBuf::from(env::var("HOME").unwrap_or_default()).join(".cache"),
        };
        Self {
            path: base.join(app_name).join("compiled.json"),
        }
    }
}

impl<T: Serialize + DeserializeOwned> CacheStore<T> for FsCache {
    fn load(&self) -> Option<(u64, T)> {
        let content = fs::read(&self.path).ok()?;
        let entry: CacheEntry<T> = serde_json::from_slice(&content).ok()?;
        Some((entry.fingerprint, entry.data))
    }

    fn save(&self, fingerprint: u64, data: &T) -> Result<(), HayaiError> {
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

impl FsFingerprinter {
    /// Convenience: create from a list of directories to scan.
    #[must_use]
    pub fn from_dirs(dirs: Vec<PathBuf>) -> Self {
        Self { paths: dirs }
    }
}

impl Fingerprinter for FsFingerprinter {
    fn fingerprint(&self) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        for path in &self.paths {
            if let Ok(meta) = fs::metadata(path)
                && let Ok(mtime) = meta.modified()
            {
                mtime_nanos(mtime).hash(&mut hasher);
            }
            if path.is_dir()
                && let Ok(entries) = fs::read_dir(path)
            {
                for entry in entries.flatten() {
                    if let Ok(meta) = entry.metadata()
                        && let Ok(mtime) = meta.modified()
                    {
                        mtime_nanos(mtime).hash(&mut hasher);
                    }
                }
            }
        }
        hasher.finish()
    }
}

/// Convert a `SystemTime` to a `u64` hash-friendly representation.
///
/// Uses nanosecond precision, truncated to `u64` which is sufficient
/// for fingerprinting purposes (wraps every ~584 years).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn mtime_nanos(t: std::time::SystemTime) -> u64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════
// In-memory implementations (for testing and daemon use)
// ═══════════════════════════════════════════════════════════════════

/// Thread-safe in-memory cache.
pub struct MemCache<T> {
    data: Mutex<Option<(u64, T)>>,
}

impl<T> MemCache<T> {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Mutex::new(None),
        }
    }
}

impl<T: Serialize + DeserializeOwned + Clone + Send> CacheStore<T> for MemCache<T> {
    fn load(&self) -> Option<(u64, T)> {
        let guard = self.data.lock().ok()?;
        guard.clone()
    }

    fn save(&self, fingerprint: u64, data: &T) -> Result<(), HayaiError> {
        let mut guard = self.data.lock().map_err(|e| HayaiError::MutexPoisoned {
            context: format!("MemCache: {e}"),
        })?;
        *guard = Some((fingerprint, data.clone()));
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
    resolve_fn: impl FnOnce() -> Result<T, HayaiError>,
) -> Result<T, HayaiError> {
    let current_fp = fp.fingerprint();

    if let Some((cached_fp, data)) = cache.load()
        && cached_fp == current_fp
    {
        return Ok(data);
    }

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

    #[test]
    fn resolve_cached_error_propagation() {
        let cache: MemCache<Vec<String>> = MemCache::empty();
        let fp = FixedFingerprinter(1);
        let result: Result<Vec<String>, HayaiError> = resolve_cached(&cache, &fp, || {
            Err(HayaiError::MutexPoisoned {
                context: "resolution failed".into(),
            })
        });
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("resolution failed")
        );
        assert!(cache.load().is_none());
    }

    #[test]
    fn resolve_cached_multiple_misses_update_fingerprint() {
        let cache: MemCache<Vec<String>> = MemCache::empty();
        let fp1 = FixedFingerprinter(1);
        let data = resolve_cached(&cache, &fp1, || Ok(vec!["v1".to_string()])).unwrap();
        assert_eq!(data, vec!["v1"]);
        assert_eq!(cache.load().unwrap().0, 1);

        let fp2 = FixedFingerprinter(2);
        let data = resolve_cached(&cache, &fp2, || Ok(vec!["v2".to_string()])).unwrap();
        assert_eq!(data, vec!["v2"]);
        assert_eq!(cache.load().unwrap().0, 2);
    }
}
