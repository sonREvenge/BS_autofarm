# Third-party assets

Bundled assets retain their original ownership. This project does not grant a new license
for upstream code, model weights, or game artwork.

- Entity and terrain models, JSON descriptions, brawler profiles and current UI templates:
  [PylaAI](https://github.com/PylaAI/PylaAI),
  commit `9b13d0f21de06637baffdff9a5a67b8b0d26a9c2`.
  PylaAI publishes CC BY-NC 4.0; full text in `licenses/PylaAI.txt`.
  The entity ONNX metadata separately identifies Ultralytics and AGPL-3.0.
  That license is included in `licenses/Ultralytics-AGPL-3.0.txt`.
  These notices are retained; do not assume the weights have a permissive commercial license.
- Retained UI templates also originate from REvengeBSIndustry,
  commit `185236e3783f6d56045d0decc38210ff8ec67029`. The original has no license file.
  Game images belong to their respective owners.
- Android server: [Genymobile scrcpy 5.0.1](https://github.com/Genymobile/scrcpy/releases/tag/v5.0.1),
  Apache-2.0; full license in `licenses/scrcpy.txt`. The binary is unmodified.
  Its SHA-256 matches the release's `SHA256SUMS.txt`.

Asset sources and hashes are in `src/revenge/assets/manifest.json`.
