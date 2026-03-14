param(
    [string]$Version
)
if ($Version -eq "") {
  Write-Host "Please provide a version number"
  exit 1
}
$imageName = "littleorange666/vision_translate"

Write-Host "Building version $Version"
docker build . -t "${imageName}:$Version"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Pushing version $Version"
docker push "${imageName}:$Version"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Building latest"
docker build . -t "${imageName}:latest"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Pushing latest"
docker push "${imageName}:latest"
if ($LASTEXITCODE -ne 0) { exit 1 }
