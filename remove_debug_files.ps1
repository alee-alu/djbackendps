# PowerShell script to remove debugging and unimportant files

# Function to safely remove a file if it exists
function Remove-FileIfExists {
    param (
        [string]$FilePath
    )
    
    if (Test-Path $FilePath) {
        Write-Host "Removing: $FilePath"
        Remove-Item -Path $FilePath -Force
    } else {
        Write-Host "File not found: $FilePath"
    }
}

# Function to recursively remove __pycache__ directories
function Remove-PycacheDirectories {
    param (
        [string]$RootPath
    )
    
    $pycacheDirs = Get-ChildItem -Path $RootPath -Directory -Recurse -Filter "__pycache__"
    
    foreach ($dir in $pycacheDirs) {
        Write-Host "Removing pycache directory: $($dir.FullName)"
        Remove-Item -Path $dir.FullName -Recurse -Force
    }
}

# Set the root directory
$rootDir = "C:\Users\ACER\Desktop\djbackendps_new"

# List of files to remove
$filesToRemove = @(
    # Root directory test files
    "$rootDir\test_heart_model.py",
    "$rootDir\test_heart_model_direct.py",
    "$rootDir\fix_indentation.py",
    "$rootDir\add_model_type_column.py",
    
    # Streamlit app debug files
    "$rootDir\djbackendps\streamlit_app\simple_heart_test.py",
    "$rootDir\djbackendps\streamlit_app\test_fixes.py",
    "$rootDir\djbackendps\streamlit_app\test_heart_model_direct.py",
    "$rootDir\djbackendps\streamlit_app\debug_dataset_loader.py",
    "$rootDir\djbackendps\streamlit_app\direct_fix.py",
    "$rootDir\djbackendps\streamlit_app\fix_dataset_loading.py",
    "$rootDir\djbackendps\streamlit_app\fix_gender_pregnancy.py",
    "$rootDir\djbackendps\streamlit_app\fix_prediction_type.py",
    "$rootDir\djbackendps\streamlit_app\fix_session_persistence.py",
    "$rootDir\djbackendps\streamlit_app\app_fixed.py",
    "$rootDir\djbackendps\streamlit_app\create_kidney_model_debug.py",
    "$rootDir\djbackendps\streamlit_app\debug_dataset_loading.py",
    
    # One-time scripts and SQL files
    "$rootDir\djbackendps\add_model_type_column.py",
    "$rootDir\djbackendps\add_username_column.py",
    "$rootDir\djbackendps\add_username_column.sql"
)

# Remove individual files
Write-Host "Starting removal of debugging files..."
foreach ($file in $filesToRemove) {
    Remove-FileIfExists -FilePath $file
}

# Remove __pycache__ directories
Write-Host "Removing __pycache__ directories..."
Remove-PycacheDirectories -RootPath $rootDir

Write-Host "Cleanup completed!"
