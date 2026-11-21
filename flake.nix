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
    # x86_64 Linux only: the one Nix platform CI and the release build and test.
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        # The toolchain pinned in rust-toolchain.toml, which rustup on the
        # Windows CI runner reads too.
        rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # ranx is not in nixpkgs. Its only use of ir_datasets (also not in
        # nixpkgs) is Qrels.from_ir_datasets, so that import is made lazy.
        ranx = ps: ps.buildPythonPackage rec {
          pname = "ranx";
          version = "0.3.21";
          pyproject = true;
          src = pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-ou3qlZOo8cnZtyb40oFjT3bxLylPI+5LityRPJZLjhQ=";
          };
          build-system = [ ps.setuptools ];
          dependencies = with ps; [
            cbor2
            fastparquet
            lz4
            numba
            numpy
            orjson
            pandas
            pyyaml
            requests
            rich
            scipy
            seaborn
            tabulate
            tqdm
          ];
          postPatch = ''
            substituteInPlace setup.py \
              --replace-fail '"ir_datasets",' ""
            substituteInPlace ranx/data_structures/qrels.py \
              --replace-fail $'import ir_datasets\n' "" \
              --replace-fail \
                "        qrels = Qrels.from_dict(ir_datasets" \
                $'        import ir_datasets\n\n        qrels = Qrels.from_dict(ir_datasets'
          '';
          pythonImportsCheck = [ "ranx" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          # mkShell's stdenv provides the C compiler that rusqlite (bundled
          # SQLite) and tokenizers (onig) need. Node.js 24, the major version
          # the release workflow uses, builds the browser UI.
          # Python runs the data-prep and scoring scripts.
          packages = [
            rust
            pkgs.pkg-config
            pkgs.nodejs_24
            (pkgs.python3.withPackages (ps: [ (ranx ps) ps.scipy ]))
          ];
        };

        # `nix develop .#reference` regenerates the reference embeddings
        # and fine-tunes MiniLM: PyTorch, transformers and
        # sentence-transformers, kept separate for their size.
        devShells.reference = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: [
              ps.torch
              ps.transformers
              ps.sentence-transformers
            ]))
          ];
        };
      });
}
