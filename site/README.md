# Public field guide

The dependency-free static site lives in `docs/`, ready for GitHub Pages from `main:/docs`.

- Rebuild the six pages: `python3 site/build.py`
- Validate local links and HTML invariants: `python3 site/check.py`
- Check JavaScript syntax: `node --check docs/assets/site.js`
- Preview using any static HTTP server, for example `python3 -m http.server --directory docs 8000`.

The build does not load model weights or call any service. The browser uses only local CSS/JavaScript; there are no analytics or third-party scripts. Theme preference is saved locally when storage is available. Interactive figures illustrate arithmetic and recorded outcomes, not live model inference.

Publication QA covered all six pages at 1440×1000 and 390×844, no document overflow or JavaScript errors, light/dark themes, the alpha slider at 1 and 6, all three replay arms, mobile navigation, and Escape dismissal. Desktop and mobile screenshots were visually reviewed. Browser QA used an isolated local headless Chrome instance; screenshots and machine-specific test harnesses are not part of the public release.

The visual language follows the author's Decision Tier and Hook Chain field guides. Site copy is based on the linked experiment reports and preserves the small-sample, control, termination and public-provenance caveats.
