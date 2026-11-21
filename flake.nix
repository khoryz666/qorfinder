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

        # pandas 3.0.6, not nixpkgs' 3.0.4, which PyPI yanked: requirements*.txt
        # pin every package to the shells' versions, so they must exist on PyPI.
        # Used by the scoring shell only; the reference shell has no pandas.
        python = pkgs.python3.override {
          self = python;
          packageOverrides = pyself: pysuper: {
            pandas = pysuper.pandas.overridePythonAttrs (old: rec {
              version = "3.0.6";
              src = pkgs.fetchFromGitHub {
                owner = "pandas-dev";
                repo = "pandas";
                tag = "v${version}";
                postFetch = ''
                  sed -i 's/git_refnames = "[^"]*"/git_refnames = " (tag: v${version})"/' $out/pandas/_version.py
                '';
                hash = "sha256-hPrSPHlIEWWhFJD5Mt6HHX8aRe9KJ2TP8mo25t0CJM4=";
              };
              # 3.0.6 excludes numpy 2.5.0, so nixpkgs' "numpy>=2.0.0" patch no longer matches.
              postPatch = ''
                substituteInPlace pyproject.toml \
                  --replace-fail '"numpy>=2.0.0,!=2.5.0"' '"numpy!=2.5.0"'
              '';
              # Upstream's test suite already ran on this release; nixpkgs' runs take an hour.
              doCheck = false;
            });
          };
        };

        # regex 2026.7.10, not nixpkgs' 2026.7.11, which was never released on
        # PyPI (see pandas above). Given to transformers alone, the only package
        # that needs it: a global override would rebuild PyTorch from source.
        # Used by the reference shell only.
        referenceBase = pkgs.python3.override {
          self = referenceBase;
          packageOverrides = pyself: pysuper: {
            transformers = pysuper.transformers.override {
              regex = pysuper.regex.overridePythonAttrs (old: rec {
                version = "2026.7.10";
                src = pkgs.fetchFromGitHub {
                  owner = "mrabarnett";
                  repo = "mrab-regex";
                  tag = version;
                  hash = "sha256-JGr+3W0T/uJcIQZtK8AXariuzK0igRsauegWdCrci64=";
                };
                # nixpkgs patches 2026.7.11's version in; 2026.7.10's pyproject.toml already has its own.
                nativeBuildInputs = [ ];
              });
            };
          };
        };

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
        scoringPython = python.withPackages (ps: [ (ranx ps) ps.scipy ]);
        referencePython = referenceBase.withPackages (ps: [
          ps.torch
          ps.transformers
          ps.sentence-transformers
        ]);

        # Writes every package of a Python environment, direct and
        # transitive, at its exact version, as a pip requirements file to be
        # installed with --no-deps. argv: output file, header line.
        lockScript = pkgs.writeText "lock.py" ''
          import importlib.metadata as m, sys
          # Not on PyPI under Nix's name or version, or not on every OS.
          SKIP = {"tkinter"}  # Nix's split-out standard library module
          # torch: the CPU build, as Nix's, from PyTorch's own wheel store.
          TORCH = "https://download.pytorch.org/whl/cpu/torch-{v}%2Bcpu-cp314-cp314-{p}.whl"
          PYPI = {"torch": "\n".join(f"torch @ {TORCH.format(v='{v}', p=p)} ; sys_platform == '{o}'"
                                     for p, o in (("manylinux_2_28_x86_64", "linux"), ("win_amd64", "win32"))),
                  "triton": "triton=={v} ; sys_platform == 'linux'"}  # torch needs it only on Linux
          seen = {}
          for d in m.distributions():
              name = d.metadata["Name"].lower().replace("_", "-").replace(".", "-")
              if name not in SKIP:
                  seen[name] = PYPI.get(name, "{n}=={v}").format(n=name, v=d.version)
          with open(sys.argv[1], "w") as f:
              f.write("# Generated by `nix build .#requirements` from flake.nix; do not edit.\n")
              f.write(sys.argv[2] + "\n")
              f.write("# Install with: python -m pip install --no-deps -r <this file>\n")
              f.write("".join(seen[n] + "\n" for n in sorted(seen)))
        '';
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
            scoringPython
          ];
        };

        # `nix develop .#reference` regenerates the reference embeddings
        # and fine-tunes MiniLM: PyTorch, transformers and
        # sentence-transformers, kept separate for their size.
        devShells.reference = pkgs.mkShell {
          packages = [ referencePython ];
        };

        # requirements.txt and requirements-finetune.txt: the two shells'
        # Python packages for pip, so Windows and pip installs get the same
        # versions. CI fails if the committed files differ from these.
        packages.requirements = pkgs.runCommand "requirements" { } ''
          mkdir $out
          ${scoringPython}/bin/python3 ${lockScript} $out/requirements.txt \
            "# Python ${python.version} packages of the default shell, for scripts/score.py."
          ${referencePython}/bin/python3 ${lockScript} $out/requirements-finetune.txt \
            "# Python ${referenceBase.version} packages of the reference shell, for scripts/finetune.py and reference_embeddings.py."
        '';
      });
}
