{
  description = "qorfinder dev shell: pinned Rust toolchain, Node.js and the Python scripts";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        # The toolchain pinned in rust-toolchain.toml, which rustup on the
        # Windows CI runner reads too.
        rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      in
      {
        devShells.default = pkgs.mkShell {
          # mkShell's stdenv provides the C compiler that rusqlite (bundled
          # SQLite) and tokenizers (onig) need. nodejs builds the browser UI.
          # Python runs the data-prep scripts.
          packages = [
            rust
            pkgs.pkg-config
            pkgs.nodejs
            pkgs.python3
          ];
        };
      });
}
