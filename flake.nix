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
          '';
        };
      });
}
