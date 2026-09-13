{
  description = "qorfinder dev toolchain (system deps only; Rust crates are managed by cargo)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          # rustup reads rust-toolchain.toml and installs the pinned
          # toolchain on first use; gcc/cmake/pkg-config/openssl are
          # build-time deps of the usearch (C++) and other native crates.
          packages = with pkgs; [
            rustup
            gcc
            cmake
            pkg-config
            openssl
          ];

          shellHook = ''
            export PATH="$HOME/.cargo/bin:$PATH"
            # Rust binaries dynamically load libstdc++.so.6 (usearch C++ crate)
            # and libssl.so.3/libcrypto.so.3 (native-tls, via ureq/fastembed) at
            # runtime. Bake those paths into built binaries via rpath (RUSTFLAGS)
            # instead of exporting LD_LIBRARY_PATH: the latter leaks into every
            # process started from this shell (git/ssh included), and nix's
            # glibc there is newer than the host's, which breaks them.
            export RUSTFLAGS="-C link-args=-Wl,-rpath,${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.openssl ]} $RUSTFLAGS"
          '';
        };
      });
}
