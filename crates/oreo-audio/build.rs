fn main() {
    println!("cargo:rerun-if-env-changed=OREO_VOSK_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_VOSK_STT").is_some()
        && let Some(directory) = std::env::var_os("OREO_VOSK_LIB_DIR")
    {
        println!(
            "cargo:rustc-link-search=native={}",
            std::path::Path::new(&directory).display()
        );
    }
}
