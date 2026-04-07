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
    /// Compute a `u64` fingerprint representing current state.
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
#[derive(Debug, Clone)]
pub struct FsCache {
    /// Path to the JSON cache file on disk.
    pub path: PathBuf,
}

impl FsCache {
    /// Create a cache at an explicit path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Create a cache at `~/.cache/{app_name}/compiled.json`.
    ///
    /// Respects `XDG_CACHE_HOME` when set, otherwise falls back to `$HOME/.cache`.
    #[must_use]
    pub fn for_app(app_name: impl AsRef<str>) -> Self {
        let app_name = app_name.as_ref();
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
#[derive(Debug, Clone)]
pub struct FsFingerprinter {
    /// Paths (files or directories) to include in the fingerprint.
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
pub(crate) fn mtime_nanos(t: std::time::SystemTime) -> u64 {
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

impl<T> Default for MemCache<T> {
    fn default() -> Self {
        Self {
            data: Mutex::new(None),
        }
    }
}

impl<T> MemCache<T> {
    /// Create an empty in-memory cache.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
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

// ═══════════════════════════════════════════════════════════════════
// Test mocks
// ═══════════════════════════════════════════════════════════════════

/// A mock cache store for testing consumer code that depends on
/// [`CacheStore`]. Wraps [`MemCache`] but allows injecting save failures.
#[cfg(test)]
pub struct MockCacheStore<T: Clone + Send> {
    inner: MemCache<T>,
    pub fail_saves: bool,
}

#[cfg(test)]
impl<T: Clone + Send> MockCacheStore<T> {
    pub fn new() -> Self {
        Self {
            inner: MemCache::empty(),
            fail_saves: false,
        }
    }

    pub fn failing() -> Self {
        Self {
            inner: MemCache::empty(),
            fail_saves: true,
        }
    }
}

#[cfg(test)]
impl<T: Serialize + DeserializeOwned + Clone + Send + Sync> CacheStore<T> for MockCacheStore<T> {
    fn load(&self) -> Option<(u64, T)> {
        self.inner.load()
    }

    fn save(&self, fingerprint: u64, data: &T) -> Result<(), HayaiError> {
        if self.fail_saves {
            return Err(HayaiError::MutexPoisoned {
                context: "mock save failure".into(),
            });
        }
        self.inner.save(fingerprint, data)
    }
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

    // ── FsCache with tempfile ─────────────────────────────────────

    #[test]
    fn fs_cache_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FsCache {
            path: dir.path().join("cache.json"),
        };
        let data = vec!["hello".to_string(), "world".to_string()];
        cache.save(42, &data).unwrap();

        let (fp, loaded) = CacheStore::<Vec<String>>::load(&cache).unwrap();
        assert_eq!(fp, 42);
        assert_eq!(loaded, data);
    }

    #[test]
    fn fs_cache_load_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FsCache {
            path: dir.path().join("nonexistent.json"),
        };
        assert!(CacheStore::<Vec<String>>::load(&cache).is_none());
    }

    #[test]
    fn fs_cache_load_invalid_json_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, b"not json").unwrap();
        let cache = FsCache { path };
        assert!(CacheStore::<Vec<String>>::load(&cache).is_none());
    }

    #[test]
    fn fs_cache_overwrite_updates() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FsCache {
            path: dir.path().join("cache.json"),
        };
        cache.save(1, &vec!["v1".to_string()]).unwrap();
        cache.save(2, &vec!["v2".to_string()]).unwrap();

        let (fp, data) = CacheStore::<Vec<String>>::load(&cache).unwrap();
        assert_eq!(fp, 2);
        assert_eq!(data, vec!["v2"]);
    }

    #[test]
    fn fs_cache_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FsCache {
            path: dir.path().join("a").join("b").join("c").join("cache.json"),
        };
        cache.save(1, &"data".to_string()).unwrap();
        let (fp, data) = CacheStore::<String>::load(&cache).unwrap();
        assert_eq!(fp, 1);
        assert_eq!(data, "data");
    }

    // ── FsFingerprinter tests ─────────────────────────────────────

    #[test]
    fn fs_fingerprinter_empty_paths() {
        let fp = FsFingerprinter::from_dirs(vec![]);
        let h = fp.fingerprint();
        let h2 = fp.fingerprint();
        assert_eq!(h, h2, "deterministic for same inputs");
    }

    #[test]
    fn fs_fingerprinter_file_changes_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, b"v1").unwrap();

        let fp = FsFingerprinter {
            paths: vec![file.clone()],
        };
        let h1 = fp.fingerprint();

        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&file, b"v2").unwrap();
        let h2 = fp.fingerprint();

        assert_ne!(h1, h2);
    }

    #[test]
    fn fs_fingerprinter_missing_path_still_works() {
        let fp = FsFingerprinter {
            paths: vec![PathBuf::from("/nonexistent/path/12345")],
        };
        let _ = fp.fingerprint();
    }

    #[test]
    fn fs_fingerprinter_directory_scan() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"a").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"b").unwrap();

        let fp = FsFingerprinter::from_dirs(vec![dir.path().to_path_buf()]);
        let h1 = fp.fingerprint();

        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(dir.path().join("c.txt"), b"c").unwrap();
        let h2 = fp.fingerprint();

        assert_ne!(h1, h2);
    }

    // ── mtime_nanos tests ─────────────────────────────────────────

    #[test]
    fn mtime_nanos_unix_epoch_is_zero() {
        assert_eq!(mtime_nanos(std::time::UNIX_EPOCH), 0);
    }

    #[test]
    fn mtime_nanos_positive_for_now() {
        assert!(mtime_nanos(std::time::SystemTime::now()) > 0);
    }

    // ── MemCache thread safety ────────────────────────────────────

    #[test]
    fn mem_cache_save_load_cycle() {
        let cache = MemCache::empty();
        assert!(CacheStore::<String>::load(&cache).is_none());
        cache.save(1, &"hello".to_string()).unwrap();
        let (fp, data) = CacheStore::<String>::load(&cache).unwrap();
        assert_eq!(fp, 1);
        assert_eq!(data, "hello");
    }

    // ── resolve_cached with FsCache ───────────────────────────────

    #[test]
    fn resolve_cached_with_fs_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FsCache {
            path: dir.path().join("cache.json"),
        };
        let fp = FixedFingerprinter(100);
        let data: Vec<String> =
            resolve_cached(&cache, &fp, || Ok(vec!["resolved".to_string()])).unwrap();
        assert_eq!(data, vec!["resolved"]);

        let data2: Vec<String> =
            resolve_cached(&cache, &fp, || panic!("should not resolve again")).unwrap();
        assert_eq!(data2, vec!["resolved"]);
    }

    // ── HayaiError display ────────────────────────────────────────

    #[test]
    fn hayai_error_mutex_poisoned_display() {
        let err = HayaiError::MutexPoisoned {
            context: "test mutex".into(),
        };
        assert!(err.to_string().contains("test mutex"));
    }

    #[test]
    fn hayai_error_io_display() {
        let err = HayaiError::Io {
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn hayai_error_json_display() {
        let json_err: Result<serde_json::Value, _> = serde_json::from_str("{invalid");
        let err = HayaiError::Json {
            source: json_err.unwrap_err(),
        };
        assert!(err.to_string().contains("JSON error"));
    }

    #[test]
    fn mock_cache_store_normal() {
        let store: MockCacheStore<Vec<String>> = MockCacheStore::new();
        assert!(store.load().is_none());
        store.save(1, &vec!["a".into()]).unwrap();
        let (fp, data) = store.load().unwrap();
        assert_eq!(fp, 1);
        assert_eq!(data, vec!["a"]);
    }

    #[test]
    fn mock_cache_store_failing() {
        let store: MockCacheStore<Vec<String>> = MockCacheStore::failing();
        let err = store.save(1, &vec!["a".into()]).unwrap_err();
        assert!(err.to_string().contains("mock save failure"));
    }

    #[test]
    fn resolve_cached_with_failing_save_still_returns_data() {
        let store: MockCacheStore<Vec<String>> = MockCacheStore::failing();
        let fp = FixedFingerprinter(1);
        let data = resolve_cached(&store, &fp, || Ok(vec!["ok".into()])).unwrap();
        assert_eq!(data, vec!["ok"]);
        assert!(store.load().is_none());
    }

    #[test]
    fn mem_cache_default() {
        let cache: MemCache<String> = MemCache::default();
        assert!(cache.load().is_none());
    }

    #[test]
    fn fs_cache_new() {
        let cache = FsCache::new("/tmp/test.json");
        assert_eq!(cache.path, PathBuf::from("/tmp/test.json"));
    }
}
