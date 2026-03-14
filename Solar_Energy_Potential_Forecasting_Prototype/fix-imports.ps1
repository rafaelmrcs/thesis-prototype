Get-ChildItem -Path "e:\thesis_code\src" -Recurse -Filter "*.tsx" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    $originalContent = $content
    
    # Remove version numbers from imports
    $content = $content -replace '"([^"]+)@\d+\.\d+(\.\d+)?"', '"$1"'
    $content = $content -replace "'([^']+)@\d+\.\d+(\.\d+)?'", "'`$1'"
    
    if ($content -ne $originalContent) {
        Set-Content -Path $_.FullName -Value $content -NoNewline
        Write-Host "Fixed: $($_.Name)"
    }
}

Write-Host "`nDone fixing all imports!"
