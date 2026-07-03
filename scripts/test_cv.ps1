param([string]$Token)

Write-Host "=== Subiendo CV ==="
$result = curl.exe -s -X POST http://localhost:8000/api/v1/me/cv -F "file=@C:\Projects\Devalign\cv_jack.pdf" -H "Authorization: Bearer $Token" --max-time 30
Write-Host $result

$cvId = ($result | ConvertFrom-Json).cv_id
Write-Host "CV ID: $cvId"

Write-Host "=== Monitoreando estado ==="
$start = Get-Date
for ($i=0; $i -lt 60; $i++) {
    $status = curl.exe -s -X GET "http://localhost:8000/api/v1/me/cv/status" -H "Authorization: Bearer $Token"
    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    Write-Host "${elapsed}s -> $status"
    if ($status -match '"status":"completed"') {
        Write-Host "COMPLETADO en ${elapsed}s"
        exit 0
    }
    if ($status -match '"status":"failed"') {
        Write-Host "FALLADO en ${elapsed}s"
        exit 1
    }
    Start-Sleep -Seconds 3
}
Write-Host "TIMEOUT - no completo"
exit 2
