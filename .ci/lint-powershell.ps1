param([Parameter(Mandatory=$true)][string]$scripts_dir)

Invoke-ScriptAnalyzer -Path $scripts_dir -Severity Warning -Recurse -EnableExit
