[CmdletBinding()]
param(
    [ValidateSet("kokoro", "piper", "all")]
    [string[]]$Providers = @("all"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Project virtual environment not found at $venvPython. Create .venv with Python 3.13 first."
}

$selected = @($Providers | ForEach-Object { $_.ToLowerInvariant() })
if ($selected -contains "all") {
    $selected = @("kokoro", "piper")
}
$selected = @($selected | Select-Object -Unique)

function Install-PinnedPackage {
    param([Parameter(Mandatory)][string]$Requirement)
    Write-Host "Installing/verifying $Requirement in project .venv"
    & $venvPython -m pip install --disable-pip-version-check $Requirement
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed while installing $Requirement"
    }
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
            throw "Existing file has an unexpected SHA-256 and was not replaced: $Destination`nExpected: $expected`nActual:   $actual`nInspect it or rerun explicitly with -Force."
        }
        Write-Warning "-Force permits replacing the hash-mismatched file: $Destination"
    }
    $temporary = "$Destination.download-$([guid]::NewGuid().ToString('N'))"
    try {
        Write-Host "Downloading $([IO.Path]::GetFileName($Destination)) from pinned upstream release"
        Invoke-WebRequest -Uri $Url -OutFile $temporary
        $downloaded = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporary).Hash.ToLowerInvariant()
        if ($downloaded -ne $expected) {
            throw "Downloaded file failed SHA-256 verification: $Destination`nExpected: $expected`nActual:   $downloaded"
        }
        Move-Item -Force -LiteralPath $temporary -Destination $Destination
        Write-Host "Installed and verified $Destination"
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -Force -LiteralPath $temporary
        }
    }
}

$ttsRoot = Join-Path $repoRoot "data\models\tts"
if ($selected -contains "kokoro") {
    Install-PinnedPackage "kokoro-onnx==0.6.1"
    $kokoroRoot = Join-Path $ttsRoot "kokoro"
    Install-VerifiedAsset `
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" `
        (Join-Path $kokoroRoot "kokoro-v1.0.onnx") `
        "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
    Install-VerifiedAsset `
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" `
        (Join-Path $kokoroRoot "voices-v1.0.bin") `
        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
}

if ($selected -contains "piper") {
    Install-PinnedPackage "piper-tts==1.7.0"
    $piperRoot = Join-Path $ttsRoot "piper"
    $piperBase = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US"
    Install-VerifiedAsset `
        "$piperBase/joe/medium/en_US-joe-medium.onnx?download=true" `
        (Join-Path $piperRoot "en_US-joe-medium.onnx") `
        "58afce0321b8d9c46d7cdf9c16500cc55a793b4220212dba6b70fb788b3baf06"
    Install-VerifiedAsset `
        "$piperBase/joe/medium/en_US-joe-medium.onnx.json?download=true" `
        (Join-Path $piperRoot "en_US-joe-medium.onnx.json") `
        "3d6d5410b3795cb1950595247ef8f06190719e6fdbfa3a2356d8ec368e1aad33"
    Install-VerifiedAsset `
        "$piperBase/john/medium/en_US-john-medium.onnx?download=true" `
        (Join-Path $piperRoot "en_US-john-medium.onnx") `
        "789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988"
    Install-VerifiedAsset `
        "$piperBase/john/medium/en_US-john-medium.onnx.json?download=true" `
        (Join-Path $piperRoot "en_US-john-medium.onnx.json") `
        "af60f177b6b550f3d7a302720c0fb89e7f94a82b5dca464775ef63b1c69ba09a"
}

Write-Host "TTS setup complete. Normal Jarvis startup performs no downloads or installs."
