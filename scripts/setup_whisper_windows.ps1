[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$WhisperVersion = "v1.9.1"
$WhisperArchiveUrl = "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
$WhisperArchiveSha256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"

$ModelRevision = "c521a4b02f422512d734391fdf08bb08c0862f68"
$ModelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/$ModelRevision/ggml-small.bin?download=true"
$ModelSha256 = "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$DataRoot = Join-Path $RepositoryRoot "data"
$DownloadRoot = Join-Path $DataRoot "downloads"
$WhisperRoot = Join-Path $DataRoot "tools\whisper.cpp\$WhisperVersion"
$ModelRoot = Join-Path $DataRoot "models\whisper"
$ArchivePath = Join-Path $DownloadRoot "whisper-$WhisperVersion-bin-x64.zip"
$ModelPath = Join-Path $ModelRoot "ggml-small.bin"
$ExecutablePath = Join-Path $WhisperRoot "Release\whisper-cli.exe"

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $ExistingHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ExistingHash -eq $ExpectedSha256) {
            Write-Host "$Label already downloaded and verified."
            return
        }
        Write-Host "$Label hash mismatch; replacing the local file."
        Remove-Item -LiteralPath $Destination -Force
    }

    $PartialPath = "$Destination.partial"
    if (Test-Path -LiteralPath $PartialPath) {
        Remove-Item -LiteralPath $PartialPath -Force
    }

    Write-Host "Downloading $Label ..."
    try {
        Invoke-WebRequest -Uri $Uri -OutFile $PartialPath -UseBasicParsing
        $ActualHash = (Get-FileHash -LiteralPath $PartialPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $ExpectedSha256) {
            throw "$Label SHA-256 mismatch. Expected $ExpectedSha256 but received $ActualHash."
        }
        Move-Item -LiteralPath $PartialPath -Destination $Destination -Force
    }
    catch {
        if (Test-Path -LiteralPath $PartialPath) {
            Remove-Item -LiteralPath $PartialPath -Force
        }
        throw
    }
    Write-Host "$Label downloaded and SHA-256 verified."
}

New-Item -ItemType Directory -Path $DownloadRoot -Force | Out-Null
New-Item -ItemType Directory -Path $WhisperRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ModelRoot -Force | Out-Null

Get-VerifiedDownload `
    -Uri $WhisperArchiveUrl `
    -Destination $ArchivePath `
    -ExpectedSha256 $WhisperArchiveSha256 `
    -Label "whisper.cpp $WhisperVersion Windows x64 CPU build"

Write-Host "Extracting whisper.cpp $WhisperVersion ..."
try {
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $WhisperRoot -Force
}
catch {
    throw "Could not extract whisper.cpp archive: $($_.Exception.Message)"
}
if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
    throw "Extraction completed but whisper-cli.exe was not found at $ExecutablePath"
}

Get-VerifiedDownload `
    -Uri $ModelUrl `
    -Destination $ModelPath `
    -ExpectedSha256 $ModelSha256 `
    -Label "multilingual Whisper small ggml model"

Write-Host ""
Write-Host "Jarvis whisper.cpp setup complete."
Write-Host "Executable: $ExecutablePath"
Write-Host "Model:      $ModelPath"
Write-Host "Backend:    official Windows x64 CPU build (GPU disabled in Phase 2C1)"
Write-Host "No global PATH or system installation was modified."
