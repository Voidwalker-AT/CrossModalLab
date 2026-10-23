# Dashboard credential and API audit — 2026-10-23

Scope: current tracked repository and all unique blobs reachable from all Git refs at `2136f89405eefc2b767ba10513852ee779646ce8`, followed by a fresh staged-file scan for the refinement. No external OAuth file was opened, copied, uploaded, or embedded. No credential values were printed.

The reachable-history scan examined 848 Git objects, including 446 unique blobs totaling 18,564,962 bytes. Credential patterns covered Google OAuth secret prefixes and client-secret JSON values, Google API keys, GitHub tokens, OpenAI-style keys, AWS access keys and private-key headers. **No pattern matches were found.** The scan includes tracked frontend source/build inputs and historical tracked content. It does not prove the absence of every possible credential format, and it does not inspect dangling objects, untracked private files, or external files. No shared-history rewrite or credential rotation was performed because no repository credential exposure was identified.

API boundaries remain unchanged and regression-tested:

- Binding is `127.0.0.1` only; non-loopback Host headers are rejected.
- Only fixed evidence endpoints and registered development sample IDs are accepted.
- Images must resolve within the workspace and match their recorded SHA-256.
- Static paths resolve within the production frontend build directory; traversal attempts are rejected.
- POST/PUT/PATCH/DELETE return 405. No training, inference, file-editing, arbitrary command, or credential endpoint exists.
- The API returns relative dataset/artifact identities, public model revisions, measured results and Git metadata. It does not enumerate arbitrary filesystem paths or return environment variables.
- React renders raw completions and explanations as text; no raw HTML injection is used. The existing CSP and `nosniff` headers remain in place.

This is a local read-only demonstration service, not a reviewed public deployment service. Tests and scans support the stated scope; they are not a penetration-test or exhaustive security guarantee.
