fn main() {
    cc::Build::new()
        .file("csrc/cpu_kern.c")
        .flag("-O3").flag("-march=sapphirerapids").flag("-fopenmp")
        .compile("cpu_kern");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rerun-if-changed=csrc/cpu_kern.c");
}
