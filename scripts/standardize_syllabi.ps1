$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Get-SectionType([string]$heading) {
    $h = $heading.Trim().ToLowerInvariant()
    if ($h -match '^##\s*general information\s*$') { return 'general' }
    if ($h -match 'materials?|material\(s\)') { return 'materials' }
    if ($h -match 'lo\(s\)|learning outcomes') { return 'learning' }
    if ($h -match 'view mapping of clos to plos|clos-plos maps') { return 'maps' }
    if ($h -match '\bsessions?\b') { return 'sessions' }
    if ($h -match 'constructive question') { return 'constructive' }
    if ($h -match 'assessment\(s\)|assignments and assessments') { return 'assessments' }
    return $null
}

function Fix-CloPloTable([string]$content) {
    $lines = @($content -split "`n", 0)
    if ($lines.Count -lt 3) { return $content }

    for ($i = 0; $i -le $lines.Count - 3; $i++) {
        $r1 = $lines[$i].Trim()
        $r2 = $lines[$i + 1].Trim()
        $r3 = $lines[$i + 2].Trim()

        if (-not ($r1.StartsWith('|') -and $r2.StartsWith('|') -and $r3.StartsWith('|'))) { continue }
        if ($r1 -notmatch '(?i)\|\s*PLOs\s*\|') { continue }
        if ($r3 -notmatch '(?i)\|\s*PLO1\s*\|') { continue }

        $cells3 = @($r3.Trim('|').Split('|') | ForEach-Object { $_.Trim() })
        if ($cells3.Count -lt 2) { continue }

        $ploLabels = $cells3[1..($cells3.Count - 1)]
        $newHeader = '|  CLO  | ' + ($ploLabels -join ' | ') + ' |'
        $sepCells = @(':-----:')
        foreach ($unused in $ploLabels) { $sepCells += ':----:' }
        $newSep = '|'+($sepCells -join '|')+'|'

        $before = @(); if ($i -gt 0) { $before = $lines[0..($i - 1)] }
        $after = @(); if ($i + 3 -le $lines.Count - 1) { $after = $lines[($i + 3)..($lines.Count - 1)] }
        $lines = @($before + $newHeader + $newSep + $after)
        break
    }

    return ($lines -join "`n")
}

function Recover-OrphanSections([hashtable]$mapped) {
    # Some source files place LO/session/assessment tables directly after materials without H2 headings.
    # Recover these tables by table header signatures and move them into canonical sections.
    $materials = [string]$mapped['materials']
    if ([string]::IsNullOrWhiteSpace($mapped['learning']) -and $materials -match '(?m)^\|\s*CLO Name\s*\|\s*CLO Details\s*\|\s*LO Details\s*\|\s*$') {
        $cloMatch = [regex]::Match($materials, '(?m)^\|\s*CLO Name\s*\|\s*CLO Details\s*\|\s*LO Details\s*\|\s*$')
        $matBefore = $materials.Substring(0, $cloMatch.Index).Trim("`n")
        $rest = $materials.Substring($cloMatch.Index)

        $sesMatch = [regex]::Match($rest, '(?m)^\|\s*Session\s*\|\s*Topic\s*\|')
        $catMatch = [regex]::Match($rest, '(?m)^\|\s*Category\s*\|\s*Type\s*\|')

        if ($sesMatch.Success) {
            $mapped['learning'] = $rest.Substring(0, $sesMatch.Index).Trim("`n")
            if ($catMatch.Success -and $catMatch.Index -gt $sesMatch.Index) {
                $mapped['sessions'] = $rest.Substring($sesMatch.Index, $catMatch.Index - $sesMatch.Index).Trim("`n")
                $mapped['assessments'] = $rest.Substring($catMatch.Index).Trim("`n")
            }
            else {
                $mapped['sessions'] = $rest.Substring($sesMatch.Index).Trim("`n")
            }
        }
        elseif ($catMatch.Success) {
            $mapped['learning'] = $rest.Substring(0, $catMatch.Index).Trim("`n")
            $mapped['assessments'] = $rest.Substring($catMatch.Index).Trim("`n")
        }
        else {
            $mapped['learning'] = $rest.Trim("`n")
        }

        $mapped['materials'] = $matBefore
    }
}

$root = 'c:\Users\dduya\Work\project\flm_agent\data\raw'
$files = Get-ChildItem -Path $root -File | Where-Object { $_.Extension -match '^(?i)\.md$' }

$processed = 0
$changed = 0
$skipped = 0

foreach ($file in $files) {
    $relPath = "data/raw/$($file.Name)"
    $headContent = git show "HEAD:$relPath" 2>$null | Out-String
    if ([string]::IsNullOrWhiteSpace($headContent)) {
        $headContent = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    }

    $norm = $headContent -replace "`r`n", "`n" -replace "`r", "`n"
    $subjectMatch = [regex]::Match($norm, '(?im)^\|\s*Subject Code:\s*\|\s*([^|\n\r]+?)\s*(?:\||$)')
    if (-not $subjectMatch.Success) {
        $skipped++
        continue
    }

    $processed++
    $subjectCode = $subjectMatch.Groups[1].Value.Trim()

    $body = [regex]::Replace($norm, '(?s)\A\s*#\s+.*?\n', '', 1)
    $h2Matches = [regex]::Matches($body, '(?m)^##\s+.*$')

    $pre = ''
    $sections = @()
    if ($h2Matches.Count -eq 0) {
        $pre = $body
    }
    else {
        $pre = $body.Substring(0, $h2Matches[0].Index)
        for ($i = 0; $i -lt $h2Matches.Count; $i++) {
            $hm = $h2Matches[$i]
            $heading = $hm.Value.Trim()
            $start = $hm.Index + $hm.Length
            if ($start -lt $body.Length -and $body[$start] -eq "`n") { $start++ }
            $end = if ($i -lt $h2Matches.Count - 1) { $h2Matches[$i + 1].Index } else { $body.Length }
            $content = if ($end -gt $start) { $body.Substring($start, $end - $start) } else { '' }
            $sections += [pscustomobject]@{ Heading = $heading; Content = $content }
        }
    }

    $mapped = @{
        general = $null; materials = $null; learning = $null; maps = $null;
        sessions = $null; constructive = $null; assessments = $null
    }

    foreach ($s in $sections) {
        $type = Get-SectionType $s.Heading
        if ($null -ne $type -and $null -eq $mapped[$type]) {
            $mapped[$type] = $s.Content
        }
    }

    if ($null -eq $mapped['general']) { $mapped['general'] = $pre }
    foreach ($k in @('materials','learning','maps','sessions','constructive','assessments')) {
        if ($null -eq $mapped[$k]) { $mapped[$k] = '' }
    }

    Recover-OrphanSections $mapped
    $mapped['maps'] = Fix-CloPloTable $mapped['maps']

    $order = @(
        @{ key = 'general'; title = 'General information' },
        @{ key = 'materials'; title = 'Materials' },
        @{ key = 'learning'; title = 'Learning outcomes' },
        @{ key = 'maps'; title = 'CLOs-PLOs maps' },
        @{ key = 'sessions'; title = 'Sessions' },
        @{ key = 'constructive'; title = 'Constructive questions' },
        @{ key = 'assessments'; title = 'Assignments and assessments' }
    )

    $new = "# Syllabus: $subjectCode`n`n"
    foreach ($item in $order) {
        $content = [string]$mapped[$item.key]
        $new += "## $($item.title)`n`n"
        if (-not [string]::IsNullOrWhiteSpace($content)) {
            $new += ($content.Trim("`n")) + "`n`n"
        } else {
            $new += "`n"
        }
    }

    $new = $new.TrimEnd("`n") + "`n"
    $newOut = $new -replace "`n", "`r`n"

    $current = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    if ($newOut -ne $current) {
        Set-Content -Path $file.FullName -Value $newOut -Encoding UTF8
        $changed++
    }
}

Write-Output "Processed syllabus files: $processed"
Write-Output "Changed files after correction: $changed"
Write-Output "Skipped non-syllabus markdown files: $skipped"
