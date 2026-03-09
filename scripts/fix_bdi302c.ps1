$ErrorActionPreference = 'Stop'
$p = 'c:\Users\dduya\Work\project\flm_agent\data\raw\BDI302c.MD'
$head = git show 'HEAD:data/raw/BDI302c.MD' | Out-String
$n = $head -replace "`r`n", "`n" -replace "`r", "`n"
$subject = ([regex]::Match($n, '(?im)^\|\s*Subject Code:\s*\|\s*([^|\n\r]+?)\s*(?:\||$)')).Groups[1].Value.Trim()
$body = [regex]::Replace($n, '(?s)\A\s*#\s+.*?\n', '', 1)

$matHead = [regex]::Match($body, '(?m)^##\s+\d+\s+material\(s\)\s*$')
$cloLine = [regex]::Match($body, '(?m)^\|\s*CLO Name\s*\|\s*CLO Details\s*\|\s*LO Details\s*\|\s*$')
$sesLine = [regex]::Match($body, '(?m)^\|\s*Session\s*\|\s*Topic\s*\|')
$catLine = [regex]::Match($body, '(?m)^\|\s*Category\s*\|\s*Type\s*\|')

$general = $body.Substring(0, $matHead.Index).Trim("`n")
$matStart = $matHead.Index + $matHead.Length
if ($matStart -lt $body.Length -and $body[$matStart] -eq "`n") { $matStart++ }
$materials = $body.Substring($matStart, $cloLine.Index - $matStart).Trim("`n")
$learning = $body.Substring($cloLine.Index, $sesLine.Index - $cloLine.Index).Trim("`n")
$sessions = $body.Substring($sesLine.Index, $catLine.Index - $sesLine.Index).Trim("`n")
$assess = $body.Substring($catLine.Index).Trim("`n")

$new = @"
# Syllabus: $subject

## General information

$general

## Materials

$materials

## Learning outcomes

$learning

## CLOs-PLOs maps


## Sessions

$sessions

## Constructive questions


## Assignments and assessments

$assess
"@

$new = ($new.TrimEnd("`n") + "`n") -replace "`n", "`r`n"
Set-Content -Path $p -Value $new -Encoding UTF8
Write-Output 'BDI302c.MD corrected.'
