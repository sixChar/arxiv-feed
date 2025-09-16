{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    pkgs.go
    pkgs.gcc
    pkgs.pkg-config
    pkgs.sqlite
  ];

  # CGO picks these up; the gcc wrapper also injects include/lib paths,
  # but being explicit avoids surprises.
  CGO_ENABLED = "1";
  CGO_CFLAGS  = "-I${pkgs.sqlite.dev}/include";
  CGO_LDFLAGS = "-L${pkgs.sqlite.out}/lib";

  GOFLAGS = "-tags=sqlite_fts5";
}
