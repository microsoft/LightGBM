$ErrorActionPreference = 'Stop'

$settings = @{
    Severity = @(
        'Information',
        'Warning',
        'Error'
    )
    IncludeDefaultRules = $true
    # Additional rules that are disabled by default.
    #
    # Some of the skips could be replaced with inline comments if PSScriptAnalyzer
    # supports that in the future (https://github.com/PowerShell/PSScriptAnalyzer/issues/849).
    Rules = @{
        PSAvoidExclaimOperator = @{
            Enable = $true
        }
        PSAvoidLongLines = @{
            Enable = $true
            MaximumLineLength = 120
        }
        PSAvoidSemicolonsAsLineTerminators = @{
            Enable = $true
        }
        PSAvoidUsingInvokeExpression = @{
            Enable = $false
        }
        PSPlaceCloseBrace = @{
            Enable = $true
            NoEmptyLineBefore = $true
            IgnoreOneLineBlock = $true
            NewLineAfter = $false
        }
        PSPlaceOpenBrace = @{
            Enable = $true
            OnSameLine = $true
            NewLineAfter = $true
            IgnoreOneLineBlock = $true
        }
        PSUseConsistentIndentation = @{
            Enable = $true
            IndentationSize = 4
            PipelineIndentation = 'IncreaseIndentationAfterEveryPipeline'
            Kind = 'space'
        }
        PSUseConsistentWhitespace = @{
            Enable = $true
            CheckInnerBrace = $true
            CheckOpenBrace = $true
            CheckOpenParen = $true
            CheckOperator = $true
            CheckSeparator = $true
            CheckPipe = $true
            CheckPipeForRedundantWhitespace = $true
            CheckParameter = $true
            IgnoreAssignmentOperatorInsideHashTable = $false
        }
        PSUseCorrectCasing = @{
            Enable = $true
        }
    }
}

# this pre-listing of files can be removed whenever PSScriptAnalyzer adds support for exclusions.
#
# see:
#
#  * https://github.com/PowerShell/PSScriptAnalyzer/issues/561
#  * https://github.com/PowerShell/vscode-powershell/issues/3048
#
$files = Get-ChildItem -Path ./ -Recurse -Filter '*.ps1' |
    Where-Object { $_.FullName -notmatch '[/\\]\.pixi[/\\]' } |
    Where-Object { $_.FullName -notmatch '[/\\]venv[/\\]' }

Invoke-ScriptAnalyzer -Path $files -EnableExit -Settings $settings
