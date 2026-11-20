<#
.SYNOPSIS
    Installs everything needed to develop QorFinder on native Windows 10/11,
    without Nix, and everything needed to edit the report in doc\ with AI.

.DESCRIPTION
    Installs, in order:
      * App Installer (winget) if it is missing
      * Visual Studio 2022 Build Tools with the C++ workload (MSVC linker)
      * rustup + Rust 1.96.0 (the version pinned by rust-toolchain.toml)
      * Node.js 24.22.3 (the version CI and the release workflow use)
      * Python 3.12
      * Git for Windows
      * LibreOffice, Pandoc, Zotero, Java JRE 21, draw.io Desktop
      * optionally VS Code, GitHub CLI, 7-Zip, SumatraPDF

    If winget cannot be installed, the script falls back to direct downloads
    for the compiler, Rust, Node, Python, Git and the document tools.

    Expect roughly 6 GB of downloads and 15-40 minutes on a fresh machine.
    The script asks for administrator rights once, at the start.

.PARAMETER SkipDocs
    Do not install the report-editing tools (LibreOffice, Pandoc, Zotero,
    Java, draw.io).

.PARAMETER WithOptional
    Also install VS Code, GitHub CLI, 7-Zip and SumatraPDF.

.PARAMETER WithScoring
    Create .venv and install the Python packages that scripts\score.py
    needs (ranx, scipy, ...).

.PARAMETER Build
    After installing, run "npm ci", "npm run build" in ui\ and
    "cargo build --release". Adds several minutes.

.PARAMETER DryRun
    Print the plan and exit. Installs nothing, does not ask for admin rights.

.PARAMETER GitName
.PARAMETER GitEmail
    Optional identity for "git config --global".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\setup-windows.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\setup-windows.ps1 -WithScoring -Build

.NOTES
    ASCII only on purpose: Windows PowerShell 5.1 reads this file correctly.
#>
[CmdletBinding()]
param(
    [switch]$SkipDocs,
    [switch]$WithOptional,
    [switch]$WithScoring,
    [switch]$Build,
    [switch]$DryRun,
    [switch]$Elevated,
    [string]$GitName,
    [string]$GitEmail
)

$ErrorActionPreference = 'Stop'

$script:Repo = $PSScriptRoot
if (-not (Test-Path (Join-Path $script:Repo 'Cargo.toml'))) {
    $script:Repo = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path (Join-Path $script:Repo 'Cargo.toml'))) {
    throw 'Run this script from inside the qorfinder repository (tools\setup-windows.ps1).'
}

$script:Failures = @()
$script:HasWinget = $false
$script:Python = $null
$script:ElevatedRun = $Elevated.IsPresent
$script:NodeVersion = '24.22.3'
$script:PythonVersion = '3.12.10'

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Message)
    Write-Host "    $Message"
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    OK: $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "    WARNING: $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "    FAILED: $Message" -ForegroundColor Red
    $script:Failures += $Message
}

# Rebuild PATH from the registry: installers update the registry, but the
# current process keeps its old environment until it is refreshed.
function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

function Add-UserPath {
    param([string]$Directory)

    if (-not (Test-Path -LiteralPath $Directory)) { return }
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @($current -split ';' | Where-Object { $_ })
    if ($parts -contains $Directory) { return }

    [Environment]::SetEnvironmentVariable('Path', (@($Directory) + $parts) -join ';', 'User')
    if (-not ($env:Path -split ';' -contains $Directory)) {
        $env:Path = "$Directory;$env:Path"
    }
    Write-Info "added to user PATH: $Directory"
}

function Invoke-Download {
    param([string]$Url, [string]$FileName)

    $dest = Join-Path $env:TEMP $FileName
    Write-Info "downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $dest -UseBasicParsing
    return $dest
}

function Invoke-Process {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Label,
        [switch]$AllowFail
    )

    Write-Step $Label
    Write-Info ($FilePath + ' ' + ($Arguments -join ' '))

    $code = -1
    try {
        $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments `
            -NoNewWindow -Wait -PassThru
        $code = $process.ExitCode
    }
    catch {
        Write-Warn $_.Exception.Message
    }

    if ($code -eq 0 -or $code -eq 1641) {
        Write-Ok "$Label finished"
    }
    elseif ($code -eq 3010) {
        Write-Warn "$Label finished; Windows asks for a reboot."
    }
    elseif ($AllowFail) {
        Write-Warn "$Label exited with code $code"
    }
    else {
        throw "$Label failed (exit code $code)"
    }

    return $code
}

function Get-GitHubAssetUrl {
    param([string]$Repository, [string]$Pattern)

    $headers = @{ 'User-Agent' = 'qorfinder-windows-setup' }
    $release = Invoke-RestMethod `
        -Uri "https://api.github.com/repos/$Repository/releases/latest" `
        -Headers $headers
    $asset = $release.assets |
        Where-Object { $_.name -match $Pattern } |
        Select-Object -First 1
    if (-not $asset) { return $null }
    return $asset.browser_download_url
}

# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Restart-Elevated {
    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"{0}"' -f $PSCommandPath),
        '-Elevated'
    )
    if ($SkipDocs) { $arguments += '-SkipDocs' }
    if ($WithOptional) { $arguments += '-WithOptional' }
    if ($WithScoring) { $arguments += '-WithScoring' }
    if ($Build) { $arguments += '-Build' }
    if ($GitName) { $arguments += @('-GitName', ('"{0}"' -f $GitName)) }
    if ($GitEmail) { $arguments += @('-GitEmail', ('"{0}"' -f $GitEmail)) }

    Write-Step 'Administrator rights are needed for the C++ build tools'
    Write-Info 'Accept the UAC prompt. A new window finishes the installation.'
    try {
        Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $arguments -Wait
    }
    catch {
        Write-Warn 'Administrator rights were not granted; nothing was installed.'
    }
}

# ---------------------------------------------------------------------------
# winget
# ---------------------------------------------------------------------------

function Test-WingetRuns {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & winget.exe --version > $null 2>&1
        return ($LASTEXITCODE -eq 0)
    }
    catch { return $false }
    finally { $ErrorActionPreference = $previous }
}

function Ensure-Winget {
    if ((Get-Command winget.exe -ErrorAction SilentlyContinue) -and (Test-WingetRuns)) {
        $script:HasWinget = $true
        Write-Ok 'winget is available'
        return $true
    }

    Write-Step 'winget is missing - installing App Installer'
    $tmp = Join-Path $env:TEMP ('winget-bootstrap-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null

    try {
        $vclibs = Join-Path $tmp 'Microsoft.VCLibs.x64.14.00.Desktop.appx'
        $xaml = Join-Path $tmp 'Microsoft.UI.Xaml.2.8.x64.appx'
        $bundle = Join-Path $tmp 'Microsoft.DesktopAppInstaller.msixbundle'

        Invoke-WebRequest -UseBasicParsing -OutFile $vclibs `
            -Uri 'https://aka.ms/Microsoft.VCLibs.x64.14.00.Desktop.appx'
        Invoke-WebRequest -UseBasicParsing -OutFile $xaml `
            -Uri 'https://github.com/microsoft/microsoft-ui-xaml/releases/download/v2.8.6/Microsoft.UI.Xaml.2.8.x64.appx'
        Invoke-WebRequest -UseBasicParsing -OutFile $bundle `
            -Uri 'https://github.com/microsoft/winget-cli/releases/latest/download/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle'

        Add-AppxPackage -Path $vclibs -ErrorAction SilentlyContinue
        Add-AppxPackage -Path $xaml -ErrorAction SilentlyContinue
        Add-AppxPackage -Path $bundle -ErrorAction Stop
    }
    catch {
        Write-Warn $_.Exception.Message
        Write-Warn 'Could not install App Installer automatically.'
        Write-Warn 'Open Microsoft Store > Library > Get updates, install App Installer, then rerun this script.'
        return $false
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    Refresh-Path
    Start-Sleep -Seconds 2

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        $alias = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\winget.exe'
        if (Test-Path -LiteralPath $alias) {
            $env:Path = (Split-Path -Parent $alias) + ';' + $env:Path
            $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        }
    }

    $script:HasWinget = [bool]$winget -and (Test-WingetRuns)
    if ($script:HasWinget) { Write-Ok 'winget installed' }
    return $script:HasWinget
}

function Test-WingetPackage {
    param([string]$Id)

    if (-not $script:HasWinget) { return $false }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & winget.exe list --id $Id -e --accept-source-agreements 2>&1 | Out-String
        return ($output -match [regex]::Escape($Id)) -and ($output -notmatch 'No installed package')
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Install-WingetPackage {
    param(
        [string]$Id,
        [string]$Name,
        [switch]$Required,
        [switch]$NoSilent,
        [string[]]$Extra
    )

    if (Test-WingetPackage $Id) {
        Write-Info "$Name is already installed"
        return $true
    }

    $arguments = @(
        'install',
        '--id', $Id,
        '-e',
        '--accept-package-agreements',
        '--accept-source-agreements',
        '--disable-interactivity'
    )
    if (-not $NoSilent) { $arguments += '--silent' }
    if ($Extra) { $arguments += $Extra }

    $code = Invoke-Process -FilePath 'winget.exe' -Arguments $arguments `
        -Label "Install $Name" -AllowFail:(-not $Required)
    $ok = ($code -eq 0 -or $code -eq 3010 -or $code -eq 1641)
    if (-not $ok -and -not $Required) {
        $script:Failures += "$Name (winget exited with $code)"
    }
    return $ok
}

# ---------------------------------------------------------------------------
# Development tools
# ---------------------------------------------------------------------------

function Test-VcTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { return $false }
    $path = & $vswhere -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null
    return [bool]$path
}

function Install-BuildToolsDirect {
    $exe = Invoke-Download 'https://aka.ms/vs/17/release/vs_BuildTools.exe' 'vs_BuildTools.exe'
    Invoke-Process -FilePath $exe -Arguments @(
        '--quiet', '--wait', '--norestart', '--nocache',
        '--add', 'Microsoft.VisualStudio.Workload.VCTools',
        '--includeRecommended'
    ) -Label 'Install Build Tools'
}

function Install-BuildTools {
    if (Test-VcTools) {
        Write-Ok 'MSVC C++ build tools are already installed'
        return
    }

    Write-Step 'Installing Visual Studio 2022 Build Tools (C++ workload)'
    Write-Info 'This is the largest download (about 3-5 GB).'

    $installed = $false
    if ($script:HasWinget) {
        $override = '--wait --passive --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
        $arguments = @(
            'install',
            '--id', 'Microsoft.VisualStudio.2022.BuildTools',
            '-e',
            '--accept-package-agreements',
            '--accept-source-agreements',
            '--disable-interactivity',
            '--override', ('"{0}"' -f $override)
        )
        try {
            Invoke-Process -FilePath winget.exe -Arguments $arguments -Label 'Install Build Tools'
            $installed = $true
        }
        catch {
            Write-Warn "winget failed: $($_.Exception.Message)"
            Write-Info 'Trying the official installer instead.'
        }
    }

    if (-not $installed) {
        Install-BuildToolsDirect
    }
    elseif (-not (Test-VcTools)) {
        Write-Warn 'winget finished but the C++ workload is not registered; running the official installer.'
        Install-BuildToolsDirect
    }

    if (-not (Test-VcTools)) {
        throw 'The C++ workload was not found after installing Build Tools. Open the Visual Studio Installer and add "Desktop development with C++".'
    }
    Write-Ok 'MSVC C++ build tools installed'
}

function Install-RustDirect {
    $init = Invoke-Download 'https://win.rustup.rs/x86_64' 'rustup-init.exe'
    Invoke-Process -FilePath $init -Arguments @(
        '-y',
        '--default-host', 'x86_64-pc-windows-msvc',
        '--profile', 'minimal'
    ) -Label 'Install rustup'
}

function Install-Rust {
    if (-not (Get-Command rustup.exe -ErrorAction SilentlyContinue)) {
        if ($script:HasWinget) {
            try {
                Install-WingetPackage -Id 'Rustlang.Rustup' -Name 'rustup' -Required | Out-Null
            }
            catch {
                Write-Warn "winget failed: $($_.Exception.Message)"
                Install-RustDirect
            }
        }
        else {
            Install-RustDirect
        }
        Refresh-Path
    }

    $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
    if (Test-Path -LiteralPath $cargoBin) {
        Add-UserPath $cargoBin
        Refresh-Path
    }

    if (-not (Get-Command rustup.exe -ErrorAction SilentlyContinue)) {
        throw 'rustup was not found after installation.'
    }

    Invoke-Process -FilePath rustup.exe `
        -Arguments @('set', 'default-host', 'x86_64-pc-windows-msvc') `
        -Label 'Select the MSVC host'

    # rust-toolchain.toml pins 1.96.0; install it explicitly so failures are visible.
    Invoke-Process -FilePath rustup.exe -Arguments @(
        'toolchain', 'install', '1.96.0',
        '--profile', 'minimal',
        '--component', 'rustfmt',
        '--component', 'clippy'
    ) -Label 'Install Rust 1.96.0 with rustfmt and clippy'
}

function Test-Node {
    $node = Get-Command node.exe -ErrorAction SilentlyContinue
    if (-not $node) { return $false }
    $version = (& node.exe --version) 2>$null
    if ($version -match '^v(\d+)\.') {
        $script:NodeMajor = [int]$Matches[1]
        return ($script:NodeMajor -ge 22)
    }
    return $false
}

function Install-Node {
    if (Test-Node) {
        if ($script:NodeMajor -ne 24) {
            Write-Warn "Node $($script:NodeMajor) is installed; CI and the release workflow use Node 24."
            Write-Warn 'Vite 8 needs Node 22.12 or newer, so this should still work.'
        }
        else {
            Write-Ok 'Node 24 is already installed'
        }
        return
    }

    Write-Step "Installing Node.js $($script:NodeVersion)"
    $url = "https://nodejs.org/dist/v$($script:NodeVersion)/node-v$($script:NodeVersion)-x64.msi"
    $msi = Invoke-Download $url "node-v$($script:NodeVersion)-x64.msi"
    Invoke-Process -FilePath msiexec.exe -Arguments @(
        '/i', ('"{0}"' -f $msi), '/qn', '/norestart'
    ) -Label 'Install Node.js'

    Refresh-Path
    $nodeDir = Join-Path $env:ProgramFiles 'nodejs'
    Add-UserPath $nodeDir
    Refresh-Path
}

function Find-Python {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $path = (& py.exe -3.12 -c "import sys; print(sys.executable)") 2>$null
            if ($LASTEXITCODE -eq 0 -and $path) { return $path.Trim() }
        }
        catch { }
    }

    $onPath = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source -notlike '*\WindowsApps\*') {
        try {
            $check = (& $onPath.Source -c "import sys; print('%d.%d' % sys.version_info[:2])") 2>$null
            if ($LASTEXITCODE -eq 0 -and $check -eq '3.12') { return $onPath.Source }
        }
        catch { }
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Install-PythonDirect {
    $url = "https://www.python.org/ftp/python/$($script:PythonVersion)/python-$($script:PythonVersion)-amd64.exe"
    $exe = Invoke-Download $url "python-$($script:PythonVersion)-amd64.exe"
    Invoke-Process -FilePath $exe -Arguments @(
        '/quiet',
        'InstallAllUsers=0',
        'PrependPath=1',
        'Include_launcher=1',
        'Include_test=0'
    ) -Label 'Install Python 3.12'
}

function Install-Python {
    $found = Find-Python
    if ($found) {
        Write-Ok "Python 3.12 is already installed: $found"
        return $found
    }

    Write-Step "Installing Python $($script:PythonVersion)"
    if ($script:HasWinget) {
        try {
            Install-WingetPackage -Id 'Python.Python.3.12' -Name 'Python 3.12' -Required | Out-Null
        }
        catch {
            Write-Warn "winget failed: $($_.Exception.Message)"
            Install-PythonDirect
        }
    }
    else {
        Install-PythonDirect
    }

    Refresh-Path
    $found = Find-Python
    if (-not $found) {
        $fallback = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
        if (Test-Path -LiteralPath $fallback) { $found = $fallback }
    }
    if (-not $found) {
        throw 'Python 3.12 was not found after installation.'
    }

    $pythonDir = Split-Path -Parent $found
    Add-UserPath $pythonDir
    Add-UserPath (Join-Path $pythonDir 'Scripts')
    Refresh-Path

    # Turn off the Microsoft Store python/python3 stubs when possible.
    foreach ($alias in @('python.exe', 'python3.exe')) {
        $stub = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\$alias"
        if (Test-Path -LiteralPath $stub) {
            try { Remove-Item -LiteralPath $stub -Force -ErrorAction Stop }
            catch { Write-Info "could not remove Store alias $alias (turn it off in Settings if python opens the Store)" }
        }
    }

    return $found
}

function Install-GitDirect {
    $url = Get-GitHubAssetUrl -Repository 'git-for-windows/git' -Pattern '^Git-[\d.]+-64-bit\.exe$'
    if (-not $url) { throw 'Could not find the Git for Windows installer.' }
    $exe = Invoke-Download $url 'Git-for-Windows-64-bit.exe'
    Invoke-Process -FilePath $exe -Arguments @('/VERYSILENT', '/NORESTART') -Label 'Install Git for Windows'
}

function Install-Git {
    if (Get-Command git.exe -ErrorAction SilentlyContinue) {
        Write-Ok 'Git is already installed'
        return
    }

    Write-Step 'Installing Git for Windows'
    if ($script:HasWinget) {
        try {
            Install-WingetPackage -Id 'Git.Git' -Name 'Git for Windows' -Required | Out-Null
        }
        catch {
            Write-Warn "winget failed: $($_.Exception.Message)"
            Install-GitDirect
        }
    }
    else {
        Install-GitDirect
    }

    Refresh-Path
    $gitDir = Join-Path $env:ProgramFiles 'Git\cmd'
    Add-UserPath $gitDir
    Refresh-Path
}

function Set-GitConfig {
    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        Write-Warn 'Git is not available; skipping git config.'
        return
    }

    Write-Step 'Configuring Git for this repository'
    & git.exe config --global core.autocrlf false
    & git.exe config --global core.eol lf
    & git.exe config --global core.longpaths true
    if ($GitName) { & git.exe config --global user.name $GitName }
    if ($GitEmail) { & git.exe config --global user.email $GitEmail }
    Write-Ok 'LF checkouts, no autocrlf conversion, long paths enabled'
    if (-not $GitName -or -not $GitEmail) {
        Write-Info 'Set your identity later with: git config --global user.name "..."'
        Write-Info '                              git config --global user.email "..."'
    }
}

# ---------------------------------------------------------------------------
# Report-editing tools
# ---------------------------------------------------------------------------

function Install-Zotero {
    # Zotero is not in the winget community repository, so always use its
    # official stable download endpoint.
    try {
        $exe = Invoke-Download 'https://www.zotero.org/download/client/dl?channel=release&platform=win-x64' 'Zotero-Setup.exe'
        Invoke-Process -FilePath $exe -Arguments @('/S') -Label 'Install Zotero' -AllowFail
    }
    catch { Write-Fail "Zotero: $($_.Exception.Message)" }
}

function Install-Docs {
    Write-Step 'Installing the report-editing tools'

    if ($script:HasWinget) {
        Install-WingetPackage -Id 'TheDocumentFoundation.LibreOffice' -Name 'LibreOffice' | Out-Null
        Install-WingetPackage -Id 'JohnMacFarlane.Pandoc' -Name 'Pandoc' | Out-Null
        Install-WingetPackage -Id 'EclipseAdoptium.Temurin.21.JRE' -Name 'Java JRE 21' | Out-Null
        Install-WingetPackage -Id 'JGraph.Draw' -Name 'draw.io Desktop' | Out-Null
    }
    else {
        Install-DocsDirect
    }

    Install-Zotero

    $libre = Join-Path $env:ProgramFiles 'LibreOffice\program'
    if (Test-Path -LiteralPath $libre) { Add-UserPath $libre }
    $pandoc = Join-Path $env:ProgramFiles 'Pandoc'
    if (Test-Path -LiteralPath $pandoc) { Add-UserPath $pandoc }
    Refresh-Path
}

function Install-DocsDirect {
    # LibreOffice: the stable directory lists the current version.
    try {
        $base = 'https://download.documentfoundation.org/libreoffice/stable/'
        $html = (Invoke-WebRequest -Uri $base -UseBasicParsing).Content
        $matches = [regex]::Matches($html, 'href="(\d+\.\d+\.\d+)/"')
        $versions = @($matches |
            ForEach-Object { $_.Groups[1].Value } |
            Sort-Object { [version]$_ } -Descending)
        if ($versions.Count -gt 0) {
            $version = $versions[0]
            $url = "$base$version/win/x86_64/LibreOffice_${version}_Win_x86-64.msi"
            $msi = Invoke-Download $url "LibreOffice_$version.msi"
            Invoke-Process -FilePath msiexec.exe -Arguments @(
                '/i', ('"{0}"' -f $msi), '/qn', '/norestart'
            ) -Label 'Install LibreOffice' -AllowFail
        }
    }
    catch { Write-Warn "LibreOffice: $($_.Exception.Message)" }

    try {
        $url = Get-GitHubAssetUrl -Repository 'jgm/pandoc' -Pattern 'windows-x86_64\.msi$'
        if ($url) {
            $msi = Invoke-Download $url 'pandoc-windows-x86_64.msi'
            Invoke-Process -FilePath msiexec.exe -Arguments @(
                '/i', ('"{0}"' -f $msi), '/qn', '/norestart'
            ) -Label 'Install Pandoc' -AllowFail
        }
    }
    catch { Write-Warn "Pandoc: $($_.Exception.Message)" }

    try {
        $msi = Invoke-Download 'https://api.adoptium.net/v3/installer/latest/21/ga/windows/x64/jre/hotspot/normal/eclipse' 'Temurin21-JRE.msi'
        Invoke-Process -FilePath msiexec.exe -Arguments @(
            '/i', ('"{0}"' -f $msi), '/qn', '/norestart'
        ) -Label 'Install Java JRE 21' -AllowFail
    }
    catch { Write-Warn "Java JRE 21: $($_.Exception.Message)" }

    try {
        $url = Get-GitHubAssetUrl -Repository 'jgraph/drawio-desktop' -Pattern 'windows-installer\.exe$'
        if ($url) {
            $exe = Invoke-Download $url 'drawio-windows-installer.exe'
            Invoke-Process -FilePath $exe -Arguments @('/S') -Label 'Install draw.io Desktop' -AllowFail
        }
    }
    catch { Write-Warn "draw.io: $($_.Exception.Message)" }
}

function Install-Optional {
    Write-Step 'Installing optional tools'

    if ($script:HasWinget) {
        Install-WingetPackage -Id 'Microsoft.VisualStudioCode' -Name 'VS Code' | Out-Null
        Install-WingetPackage -Id 'GitHub.cli' -Name 'GitHub CLI' | Out-Null
        Install-WingetPackage -Id '7zip.7zip' -Name '7-Zip' | Out-Null
        Install-WingetPackage -Id 'SumatraPDF.SumatraPDF' -Name 'SumatraPDF' | Out-Null
    }
    else {
        Write-Info 'No winget: skipping optional tools.'
    }
}

# ---------------------------------------------------------------------------
# Python scoring / build
# ---------------------------------------------------------------------------

function Install-Scoring {
    if (-not $script:Python) {
        $script:Python = Find-Python
    }
    if (-not $script:Python) {
        Write-Warn 'Python was not found; skipping the scoring environment.'
        return
    }

    Write-Step 'Creating the Python scoring environment (.venv)'
    $venv = Join-Path $script:Repo '.venv'
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Invoke-Process -FilePath $script:Python -Arguments @('-m', 'venv', $venv) -Label 'Create .venv'
    }

    Invoke-Process -FilePath $venvPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') -Label 'Upgrade pip'
    Invoke-Process -FilePath $venvPython -Arguments @(
        '-m', 'pip', 'install',
        'numpy==1.26.4',
        'numba==0.61.2',
        'pandas<3',
        'scipy',
        'cbor2',
        'fastparquet',
        'lz4',
        'orjson',
        'pyyaml',
        'rich',
        'seaborn',
        'tabulate',
        'tqdm'
    ) -Label 'Install ranx dependencies'
    Invoke-Process -FilePath $venvPython -Arguments @(
        '-m', 'pip', 'install', '--no-deps', 'ranx==0.3.21'
    ) -Label 'Install ranx (without ir_datasets)'

    # flake.nix makes Qrels.from_ir_datasets import ir_datasets lazily. Do the
    # same here so the Windows install does not need trec-car-tools.
    $qrels = Join-Path $venv 'Lib\site-packages\ranx\data_structures\qrels.py'
    if (Test-Path -LiteralPath $qrels) {
        try {
            $source = [IO.File]::ReadAllText($qrels)
            $source = $source -replace "(?m)^import ir_datasets\r?\n", ''
            $source = $source -replace 'qrels = Qrels\.from_dict\(ir_datasets', "import ir_datasets`r`n`r`n        qrels = Qrels.from_dict(ir_datasets"
            [IO.File]::WriteAllText($qrels, $source)
            Write-Ok 'ranx patched for Windows (lazy ir_datasets import)'
        }
        catch { Write-Warn "Could not patch ranx: $($_.Exception.Message)" }
    }

    Invoke-Process -FilePath $venvPython -Arguments @(
        (Join-Path $script:Repo 'scripts\score.py')
    ) -Label 'Run scripts\score.py' -AllowFail
}

function Build-Repo {
    Write-Step 'Building the UI and the release binary'

    Push-Location (Join-Path $script:Repo 'ui')
    try {
        Invoke-Process -FilePath 'npm.cmd' -Arguments @('ci') -Label 'npm ci'
        Invoke-Process -FilePath 'npm.cmd' -Arguments @('run', 'build') -Label 'npm run build'
    }
    finally {
        Pop-Location
    }

    Push-Location $script:Repo
    try {
        Invoke-Process -FilePath 'cargo.exe' -Arguments @('build', '--release') -Label 'cargo build --release'
    }
    finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

function Get-VersionLine {
    param([string]$FilePath, [string[]]$Arguments)

    if (-not (Test-Path -LiteralPath $FilePath) -and -not (Get-Command $FilePath -ErrorAction SilentlyContinue)) {
        return 'not found'
    }
    try {
        $output = & $FilePath @Arguments 2>$null | Select-Object -First 1
        if ($output) { return $output.Trim() }
    }
    catch { }
    return 'not found'
}

function Show-Summary {
    Write-Step 'Summary'

    $cargo = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
    $node = Join-Path $env:ProgramFiles 'nodejs\node.exe'
    $git = Join-Path $env:ProgramFiles 'Git\cmd\git.exe'
    $soffice = Join-Path $env:ProgramFiles 'LibreOffice\program\soffice.exe'

    Write-Info ('rustc / cargo : ' + (Get-VersionLine $cargo @('--version')))
    Write-Info ('node          : ' + (Get-VersionLine $node @('--version')))
    Write-Info ('git           : ' + (Get-VersionLine $git @('--version')))
    if ($script:Python) {
        Write-Info ('python        : ' + (Get-VersionLine $script:Python @('--version')))
    }
    if (Test-Path -LiteralPath $soffice) {
        Write-Info ('libreoffice   : ' + (Get-VersionLine $soffice @('--version')))
    }
    if (Test-VcTools) { Write-Info 'MSVC C++ tools: installed' }
    else { Write-Info 'MSVC C++ tools: NOT FOUND' }

    if ($script:Failures.Count -gt 0) {
        Write-Host ''
        Write-Host 'Failures to check by hand:' -ForegroundColor Yellow
        foreach ($failure in $script:Failures) { Write-Host "  - $failure" -ForegroundColor Yellow }
    }

    Write-Host ''
    Write-Host 'Next steps:' -ForegroundColor Cyan
    Write-Host '  1. Close this window and open a new PowerShell (PATH is now updated).'
    Write-Host '  2. cd ' + $script:Repo
    Write-Host '  3. cargo test --workspace'
    Write-Host '  4. cargo test --workspace -- --include-ignored'
    Write-Host '  5. .\target\release\qorfinder.exe serve'
    Write-Host ''
    Write-Host 'If Build Tools asked for a reboot, reboot before step 3.' -ForegroundColor Yellow
    Write-Host 'The report is in doc\ (not tracked by Git): back it up before AI edits.' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'Report editing with AI:' -ForegroundColor Cyan
    Write-Host '  backup : Copy-Item doc\23ACB05662_FYP1.odt doc\.backup\23ACB05662_FYP1.before-ai.odt'
    Write-Host '  read   : New-Item -ItemType Directory -Force doc\.work | Out-Null; pandoc doc\23ACB05662_FYP1.odt --extract-media=doc\.work\media -t gfm --wrap=none -o doc\.work\report.md'
    Write-Host '  apply  : edit in Writer with Track Changes on, then refresh Zotero and the TOC'
    Write-Host '  export : & "C:\Program Files\LibreOffice\program\soffice.exe" --headless --convert-to pdf --outdir doc doc\23ACB05662_FYP1.odt'
    Write-Host '  note   : never convert doc\.work\report.md back to ODT; it loses styles and citations.'
}

function Show-Plan {
    Write-Step 'Dry run: nothing will be installed'
    Write-Info 'winget (App Installer) if missing'
    Write-Info 'Visual Studio 2022 Build Tools + Desktop development with C++'
    Write-Info 'rustup + Rust 1.96.0 (rustfmt, clippy)'
    Write-Info "Node.js $($script:NodeVersion)"
    Write-Info "Python $($script:PythonVersion)"
    Write-Info 'Git for Windows'
    if (-not $SkipDocs) {
        Write-Info 'LibreOffice, Pandoc, Zotero, Java JRE 21, draw.io Desktop'
    }
    if ($WithOptional) {
        Write-Info 'VS Code, GitHub CLI, 7-Zip, SumatraPDF'
    }
    if ($WithScoring) {
        Write-Info 'Python .venv with ranx/scipy (only needed to re-score results)'
    }
    if ($Build) {
        Write-Info 'npm ci + npm run build + cargo build --release'
    }
    Write-Info ''
    Write-Info 'Run without -DryRun to install.'
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if ($DryRun) {
    Show-Plan
    exit 0
}

if (-not (Test-Admin)) {
    Restart-Elevated
    exit 0
}

$logPath = Join-Path $script:Repo 'setup-windows.log'
try { Start-Transcript -Path $logPath -Append | Out-Null } catch { }

try {
    Write-Step 'QorFinder Windows setup'
    Write-Info "repository: $script:Repo"
    Write-Info "log:        $logPath"

    Ensure-Winget | Out-Null

    Install-BuildTools
    Install-Rust
    Install-Node
    $script:Python = Install-Python
    Install-Git
    Set-GitConfig

    if (-not $SkipDocs) {
        Install-Docs
    }

    if ($WithOptional) {
        Install-Optional
    }

    if ($WithScoring) {
        Install-Scoring
    }

    if ($Build) {
        Build-Repo
    }
}
catch {
    Write-Fail $_.Exception.Message
}
finally {
    Show-Summary
    try { Stop-Transcript | Out-Null } catch { }
    if ($script:ElevatedRun) {
        Write-Host ''
        Read-Host 'Press Enter to close this window' | Out-Null
    }
}
