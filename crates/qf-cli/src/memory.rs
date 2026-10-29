//! Peak memory of this process, printed after a run for Table 4.3 and
//! NFR3. Read by the process itself because, once it exits, neither
//! `Get-Process` nor `/proc` can report it.

/// The largest resident set (Linux) or working set (Windows) this process
/// has had so far, in bytes. `None` where it cannot be read.
pub fn peak_bytes() -> Option<u64> {
    imp::peak_bytes()
}

#[cfg(target_os = "linux")]
mod imp {
    /// `VmHWM` in `/proc/self/status`, the same value as `getrusage`'s
    /// `ru_maxrss`, given in kB.
    pub fn peak_bytes() -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    pub fn peak_bytes() -> Option<u64> {
        let cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: the struct is plain data, zero is a valid value for every
        // field, and `cb` tells the call its size.
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        counters.cb = cb;
        // SAFETY: the pseudo-handle of the current process is always valid
        // and `counters` lives for the whole call.
        let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, cb) };
        (ok != 0).then_some(counters.PeakWorkingSetSize as u64)
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    pub fn peak_bytes() -> Option<u64> {
        None
    }
}

/// `peak memory 412.3 MB`, or nothing if it cannot be read.
pub fn peak_line() -> Option<String> {
    peak_bytes().map(|b| format!("peak memory {:.1} MB", b as f64 / (1024.0 * 1024.0)))
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn peak_is_read_and_grows_with_use() {
        let before = super::peak_bytes().unwrap();
        assert!(before > 0);
        let block = vec![1u8; 64 * 1024 * 1024];
        std::hint::black_box(&block);
        assert!(super::peak_bytes().unwrap() >= before + 32 * 1024 * 1024);
    }
}
