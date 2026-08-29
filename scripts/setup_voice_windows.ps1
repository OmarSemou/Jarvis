[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Project virtual environment not found at $venvPython. Create .venv with Python 3.13 first."
}

function Install-VerifiedAsset {
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$Destination,
        [Parameter(Mandatory)][string]$Sha256
    )
    $expected = $Sha256.ToLowerInvariant()
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLowerInvariant()
        if ($actual -eq $expected) {
            Write-Host "Verified existing $([IO.Path]::GetFileName($Destination))"
            return
        }
        if (-not $Force) {
            throw "Existing asset has an unexpected SHA-256 and was not replaced: $Destination`nExpected: $expected`nActual:   $actual"
        }
        Write-Warning "Replacing hash-mismatched asset because -Force was supplied: $Destination"
    }
    $temporary = "$Destination.download-$([guid]::NewGuid().ToString('N'))"
    try {
        Write-Host "Downloading $([IO.Path]::GetFileName($Destination)) from pinned openWakeWord v0.5.1"
        Invoke-WebRequest -Uri $Url -OutFile $temporary
        $downloaded = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporary).Hash.ToLowerInvariant()
        if ($downloaded -ne $expected) {
            throw "Downloaded asset failed SHA-256 verification: $Destination`nExpected: $expected`nActual:   $downloaded"
        }
        Move-Item -Force -LiteralPath $temporary -Destination $Destination
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -Force -LiteralPath $temporary
        }
    }
}

Write-Host "Installing/verifying openwakeword==0.6.0 in project .venv (ONNX runtime; no PyTorch)."
& $venvPython -m pip install --disable-pip-version-check "openwakeword==0.6.0"
if ($LASTEXITCODE -ne 0) {
    throw "pip failed while installing openwakeword==0.6.0"
}

$assetRoot = Join-Path $repoRoot "data\models\wakeword"
$releaseRoot = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
Install-VerifiedAsset `
    "$releaseRoot/hey_jarvis_v0.1.onnx" `
    (Join-Path $assetRoot "hey_jarvis_v0.1.onnx") `
    "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb"
Install-VerifiedAsset `
    "$releaseRoot/melspectrogram.onnx" `
    (Join-Path $assetRoot "melspectrogram.onnx") `
    "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"
Install-VerifiedAsset `
    "$releaseRoot/embedding_model.onnx" `
    (Join-Path $assetRoot "embedding_model.onnx") `
    "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"
Install-VerifiedAsset `
    "$releaseRoot/silero_vad.onnx" `
    (Join-Path $assetRoot "silero_vad.onnx") `
    "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28"

Write-Host "Voice dependency setup complete. Normal Jarvis startup performs no downloads."
Write-Host "Wake phrase: Hey Jarvis (bare Jarvis can have a higher false-reject rate)."
Write-Host "Model licensing/provenance limitations are documented in docs/audio.md."
