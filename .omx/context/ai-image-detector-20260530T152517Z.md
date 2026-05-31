Task statement
- Build the local MVP into a simple, easy-to-use open-source AI Image Detector project.
- Research practical SOTA/open-source AI-generated image detection approaches.
- Implement real end-to-end functionality and run benchmark testing on standard data.

Desired outcome
- A usable Python package with CLI, optional web UI/API, model loading, batch detection, and evaluation tooling.
- Reproducible benchmark command(s) and saved benchmark results.
- Documentation that explains model choice, limitations, install, usage, and benchmark reproduction.

Known facts/evidence
- Workspace root: /Volumes/wd/github_star/AIImageDetector
- Current code package: /Volumes/wd/github_star/AIImageDetector/ai-image-detector
- Existing MVP wraps a UnivFD/UniversalFakeDetect-style CLIP ViT-L/14 backbone plus linear head.
- No AGENTS.md file was found in /, /Volumes, /Volumes/wd, /Volumes/wd/github_star, the workspace root, the package root, or $HOME.
- The package root is not currently a git repository.

Constraints
- Follow system/developer/user instructions over this snapshot.
- Use current web research for SOTA claims and cite sources in final output.
- Keep implementation practical for a small open-source project.
- Avoid claiming deterministic proof: AI image detection is probabilistic.

Unknowns/open questions
- Which public benchmark can be downloaded and evaluated within the local runtime budget.
- Whether pretrained UnivFD head weights and OpenAI CLIP weights are reachable from the current network.
- Whether the current Python environment already has all ML dependencies.

Likely codebase touchpoints
- pyproject.toml
- README.md
- aidetector/model.py
- aidetector/cli.py
- aidetector/types.py
- new evaluation, dataset, API, and test modules
