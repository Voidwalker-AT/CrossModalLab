param([Parameter(Mandatory=$true)][string]$WorkspaceRoot)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$dispatcher = Join-Path $WorkspaceRoot 'Shared/scripts/workspace-projects.ps1'
$cli = Join-Path $WorkspaceRoot 'wsctl.ps1'
foreach ($file in @($dispatcher,$cli)) {
    $tokens=$null; $errors=$null
    [System.Management.Automation.Language.Parser]::ParseFile($file,[ref]$tokens,[ref]$errors) > $null
    if ($errors.Count) { throw $errors }
}
$fixture = Join-Path $projectRoot ('.cache/wsctl/fixture_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixture > $null
$manifest = @{schemaVersion=2;workspace=@{name='fixture';root='.';sharedPath='Shared'};projects=@(@{
    id='dispatch-regression';name='Dispatch regression';path='.';category='active';status='active';
    languages=@('python');packageManager='none';environment='existing';
    commands=@{success=@{executable='python';args=@('-c','raise SystemExit(0)')};failure=@{executable='python';args=@('-c','raise SystemExit(7)')}}
})}
$manifest | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $fixture 'workspace.json') -Encoding utf8
foreach ($case in @(@('success',0),@('failure',7),@('missing',1))) {
    & pwsh -NoProfile -File $dispatcher -WorkspaceRoot $fixture -Name 'dispatch-regression' -Run $case[0] > $null
    if ($LASTEXITCODE -ne $case[1]) { throw "Dispatch $($case[0]): expected $($case[1]), got $LASTEXITCODE" }
    Write-Output "PASS dispatcher $($case[0]) exit $LASTEXITCODE"
}
& pwsh -NoProfile -File $cli project dl-exp1 -Json > $null
if ($LASTEXITCODE -ne 0) { throw 'Existing project inspection failed' }
& pwsh -NoProfile -File $cli run crossmodallab missing-command > $null
if ($LASTEXITCODE -ne 1) { throw 'Outer CLI did not propagate missing-command failure' }
Write-Output 'PASS PowerShell syntax, existing dl-exp1 resolution and outer CLI failure propagation'
