# Function to download and analyze JavaScript files
function Find-Endpoints {
    param (
        [Parameter(Mandatory=$true)]
        [string]$BaseUrl
    )

    # Regular expressions for finding endpoints
    $patterns = @(
        '["''`]\/[a-zA-Z0-9_?&=\/\-\#\.]*["''`]',                # /api/v1/users
        '(?:url.*?|href.*?|action.*?)[""''`](/[^""''`]+)[""''`]', # url: "/endpoint"
        'fetch\([""''`](/[^""''`]+)[""''`]\)',                    # fetch("/api/data")
        'ajax\({.*?url:\s*[""''`](/[^""''`]+)[""''`]'            # ajax({url: "/endpoint"})
    )

    try {
        # Get the main page
        $response = Invoke-WebRequest -Uri $BaseUrl -UseBasicParsing

        # Find all JavaScript file URLs
        $jsFiles = $response.Links | Where-Object {
            $_.href -like "*.js" -or
            $_.src -like "*.js"
        } | Select-Object -ExpandProperty href

        # Add any script tags with inline src
        $jsFiles += ([regex]'src="([^"]+\.js)"').Matches($response.Content) |
            ForEach-Object { $_.Groups[1].Value }

        Write-Host "Found $($jsFiles.Count) JavaScript files" -ForegroundColor Green

        # Process each JavaScript file
        foreach ($jsFile in $jsFiles) {
            # Resolve relative URLs
            if ($jsFile -notlike "http*") {
                $jsFile = New-Object -TypeName Uri -ArgumentList ([Uri]$BaseUrl, $jsFile)
            }

            Write-Host "Analyzing $jsFile" -ForegroundColor Yellow

            try {
                $jsContent = (Invoke-WebRequest -Uri $jsFile -UseBasicParsing).Content

                # Search for endpoints using our patterns
                foreach ($pattern in $patterns) {
                    $matches = [regex]::Matches($jsContent, $pattern)
                    foreach ($match in $matches) {
                        $endpoint = $match.Groups[1].Value
                        if ($endpoint) {
                            Write-Host "Found endpoint: $endpoint" -ForegroundColor Cyan
                        }
                    }
                }
            }
            catch {
                Write-Host "Error downloading $jsFile : $_" -ForegroundColor Red
            }
        }
    }
    catch {
        Write-Host "Error accessing $BaseUrl : $_" -ForegroundColor Red
    }
}

# Example usage
# Find-Endpoints -BaseUrl "http://example.com"