# download_data.ps1
# Downloads and extracts the MovieLens 100k dataset from the official
# GroupLens source into .\data\ml-100k\
#
# Usage (from this folder):
#   powershell -ExecutionPolicy Bypass -File .\download_data.ps1

$ErrorActionPreference = "Stop"

$url     = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
$dataDir = Join-Path $PSScriptRoot "data"
$zipPath = Join-Path $dataDir "ml-100k.zip"

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

Write-Host "Downloading MovieLens 100k from $url ..."
Invoke-WebRequest -Uri $url -OutFile $zipPath

Write-Host "Extracting to $dataDir ..."
Expand-Archive -Path $zipPath -DestinationPath $dataDir -Force

$infoFile = Join-Path $dataDir "ml-100k\u.info"
if (Test-Path $infoFile) {
    Write-Host "`nDataset ready. Contents of u.info:"
    Get-Content $infoFile
} else {
    Write-Warning "u.info not found - extraction may have failed."
}
