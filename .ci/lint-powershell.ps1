$settings = @{

    IncludeDefaultRules = $true

    ExcludeRules        = @(
        'PSAvoidUsingWriteHost',
        'PSReviewUnusedParameter',
        'PSUseSingularNouns'
    )

    Rules               = @{
        PSPlaceOpenBrace                   = @{
            Enable             = $true
            OnSameLine         = $true
            NewLineAfter       = $true
            IgnoreOneLineBlock = $true
        }

        PSPlaceCloseBrace                  = @{
            Enable             = $true
            NewLineAfter       = $false
            IgnoreOneLineBlock = $true
            NoEmptyLineBefore  = $true
        }

        PSUseConsistentIndentation         = @{
            Enable              = $true
            Kind                = 'space'
            PipelineIndentation = 'IncreaseIndentationForFirstPipeline'
            IndentationSize     = 4
        }

        PSUseConsistentWhitespace          = @{
            Enable                                  = $true
            CheckInnerBrace                         = $true
            CheckOpenBrace                          = $true
            CheckOpenParen                          = $true
            CheckOperator                           = $false
            CheckPipe                               = $true
            CheckPipeForRedundantWhitespace         = $false
            CheckSeparator                          = $true
            CheckParameter                          = $false
            IgnoreAssignmentOperatorInsideHashTable = $true
        }

        PSAlignAssignmentStatement         = @{
            Enable         = $true
            CheckHashtable = $true
        }

        PSUseCorrectCasing                 = @{
            Enable = $true
        }

        PSAvoidSemicolonsAsLineTerminators = @{
            Enable = $true
        }
    }
}

Invoke-ScriptAnalyzer -Path "${env:BUILD_DIRECTORY}/.ci" -Recurse -EnableExit -Severity Warning,Error -Settings $settings
