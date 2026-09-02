with import <nixpkgs> {}; (
let python =
  let
  packageOverrides = self:
  super: {
    opencv4 = super.opencv4.override {
      enableGtk2 = true;
      gtk2 = pkgs.gtk2;
      enableFfmpeg = true;
    };
  };
  in
    pkgs.python313.override {inherit packageOverrides; self = python;};
in stdenv.mkDerivation {
  name = "impurePythonEnv";
  buildInputs = [
    imagemagick
    v4l-utils
    (python313.buildEnv.override {
      extraLibs = [
        pkgs.python313Packages.matplotlib
        pkgs.python313Packages.numpy
        pkgs.python313Packages.scipy
        pkgs.python313Packages.gnureadline
        pkgs.python313Packages.rasterio
        python.pkgs.opencv4
      ];
      ignoreCollisions = true;
    })
  ];
  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export LANG=en_US.UTF-8	
  '';
})
