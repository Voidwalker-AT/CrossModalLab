# Reviewed WSCTL v1.0 exit-code repair proposal

Reproduction on 2026-10-24: with CrossModalLab listening on 8765, the direct launcher reports the listener owner and returns 2. The exact registered WSCTL command reports the same conflict but returns 0. No process is stopped. This makes terminal automation mistake a failed launch for success.

The cause is visible in the installed dispatcher: it calls the command and then unconditionally executes `exit 0`. The outer CLI also needs to explicitly propagate its subprocess status. Proposed changes, subject to separate approval:

1. `E:/Workspace/Shared/scripts/workspace-projects.ps1`: initialize a command exit-code variable; capture `$LASTEXITCODE` immediately after the existing `cmd.exe /c` invocation; preserve the existing finally/location cleanup; use `exit $commandExitCode` instead of unconditional zero for this run branch.
2. `E:/Workspace/wsctl.ps1`: immediately after the existing `run` branch's `pwsh` dispatcher invocation, add `exit $LASTEXITCODE`.

No command registration, environment, unrelated project behavior, or other CLI branch is changed. Save byte-for-byte local backups before applying. Re-test a successful no-GPU launch, occupied-port failure, deliberate fixture exit 7 and Ctrl+C cleanup. Keep backups excluded from Git. These are global WSCTL source files, so the earlier registry-only approval does not authorize this repair.

Applied on 2026-10-24 after separate explicit approval of these two files. The dispatcher now captures the child status immediately and exits with it from the run branch, before the normal inspection branch's exit 0. The outer CLI forwards that status. Fixture success=0, fixture failure=7, missing command=1 and actual busy-port launch=2 all passed. The actual dashboard started, responded to health requests and shut down on Ctrl+C; the Windows batch shim's Y confirmation was handled. The existing ten-view WSCTL panel test passed. Backups and full before/after hashes are retained only in ignored `.cache/wsctl/`. The registry's previous 28 entries and other settings were compared unchanged. The broader existing platform doctor reported unrelated stale-path/toolchain warnings; no unrelated project was repaired.
