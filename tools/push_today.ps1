<#
.SYNOPSIS
    Verify today's lesson, commit it, and push.

.DESCRIPTION
    The daily gate. Rebuilds the notebook from its .py source, executes it
    headlessly, and only commits if every cell ran clean. Nothing broken
    reaches GitHub.

.PARAMETER Lesson
    Lesson number to ship, e.g. 4. Omit to verify and commit everything.

.PARAMETER NoPush
    Commit locally but do not push. Useful while you are still reviewing.

.EXAMPLE
    ./tools/push_today.ps1 4
    ./tools/push_today.ps1 4 -NoPush
#>
param(
    [int]$Lesson = 0,
    [switch]$NoPush
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$target = if ($Lesson -gt 0) { "$Lesson" } else { "" }

Write-Host "`n[1/5] Building notebooks from lesson sources..." -ForegroundColor Cyan
python tools/build_nb.py $target
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "`n[2/5] Executing notebooks headless..." -ForegroundColor Cyan
python tools/verify.py $target
if ($LASTEXITCODE -ne 0) { throw "verification failed -- nothing was committed" }

Write-Host "`n[3/5] Refreshing progress files..." -ForegroundColor Cyan
python tools/track.py sync
if ($LASTEXITCODE -ne 0) { throw "track sync failed" }

Write-Host "`n[4/5] Committing..." -ForegroundColor Cyan
git add -A
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "  nothing to commit -- working tree is clean." -ForegroundColor Yellow
    exit 0
}
$staged | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }

if ($Lesson -gt 0) {
    $title = python -c "import json,sys;d=json.load(open('curriculum.json'));print(next(l['title'] for l in d['lessons'] if l['id']==$Lesson))"
    $msg = "L{0:d2}: {1}" -f $Lesson, $title
} else {
    $msg = "Update lessons and site"
}
git commit -q -m $msg
Write-Host "  committed: $msg" -ForegroundColor Green

if ($NoPush) {
    Write-Host "`n[5/5] -NoPush set -- staying local. Push later with: git push" -ForegroundColor Yellow
    exit 0
}

$remote = git remote 2>$null
if (-not $remote) {
    Write-Host "`n[5/5] No git remote configured yet. To publish:" -ForegroundColor Yellow
    Write-Host "  1. Create an empty repo at https://github.com/new  (name: dl-from-scratch)"
    Write-Host "  2. git remote add origin https://github.com/Utathyaworks/dl-from-scratch.git"
    Write-Host "  3. git push -u origin main"
    Write-Host "  4. Repo Settings > Pages > Source: GitHub Actions"
    exit 0
}

Write-Host "`n[5/5] Pushing..." -ForegroundColor Cyan
git push
Write-Host "`nDone. Site redeploys automatically." -ForegroundColor Green
