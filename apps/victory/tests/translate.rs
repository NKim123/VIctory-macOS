use std::path::Path;
use std::process::Command;

#[test]
fn translate_rejects_zero_workers() {
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    for extra_args in [&[][..], &["--cache"][..]] {
        let output = Command::new(env!("CARGO_BIN_EXE_victory"))
            .arg("translate")
            .arg(&input)
            .args(["--jobs", "0"])
            .args(extra_args)
            .output()
            .expect("run victory translate");

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{extra_args:?}: {stderr}");
        assert!(stderr.contains("--jobs"), "{extra_args:?}: {stderr}");
    }
}

/// A compute shader that ends immediately, in a synthetic FSPX program.
fn end_program() -> Vec<u8> {
    let words = |values: &[u32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut fspx = words(&[u32::from_le_bytes(*b"FSPX"), 1, 3, 120]);
    fspx.extend(words(&[0, 64, 28, 0, 6, 92, 4, 0, 7, 96, 24, 0]));
    fspx.extend(words(&[0, 0, 0, 1 | (1 << 16), 1 | (16 << 16), 0, 0]));
    fspx.extend(words(&[0xbf81_0000])); // s_endpgm
    fspx.extend(words(&[0x403, 4, 1, 0, 0, 0x440]));
    let mut dxbc = Vec::from(*b"DXBC");
    dxbc.extend([0; 16]);
    dxbc.extend(words(&[1, 44 + fspx.len() as u32, 1, 36]));
    dxbc.extend(*b"ACCC");
    dxbc.extend((fspx.len() as u32).to_le_bytes());
    dxbc.extend(fspx);
    dxbc
}

#[test]
fn offline_warmup_writes_the_shared_wine_cache_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("compute.bin");
    let bytes = end_program();
    std::fs::write(&input, &bytes).unwrap();
    let from_cli = tmp.path().join("cli");
    let from_gpu = tmp.path().join("gpu");
    let mut run = Command::new(env!("CARGO_BIN_EXE_victory"));
    run.args(["translate", "--cache", "--wine-defaults"])
        .arg(&input)
        .env("VICTORY_CACHE_DIR", &from_cli)
        .env("VICTORY_DATA_DIR", tmp.path().join("data"));
    let output = run.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let cache = victory_gpu::shader_cache::ShaderCache::open(&from_gpu, true).unwrap();
    let (shader, hit) = cache.host_shader(&bytes);
    assert!(!hit);
    let shader = shader.unwrap();
    let expected_dir = cache.dir();
    let cli_dir = from_cli
        .join("shaders")
        .join(expected_dir.file_name().unwrap());
    let shard = std::fs::read_dir(expected_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let entry = std::fs::read_dir(shard.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let expected = std::fs::read(entry.path()).unwrap();
    assert!(
        expected
            .windows(shader.dxil.len())
            .any(|w| w == shader.dxil)
    );
    assert_eq!(
        std::fs::read(cli_dir.join(shard.file_name()).join(entry.file_name())).unwrap(),
        expected
    );
    let second = run.output().unwrap();
    assert!(second.status.success());
    assert!(String::from_utf8_lossy(&second.stdout).contains("1 already cached, 0 translated"));
}
