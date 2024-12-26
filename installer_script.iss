

[Setup]
AppName=app
AppVersion=1.0
AppVerName=ImageSegmentationApplication Version 1.0
DefaultDirName={pf}\ImageSegmentationApplication
DefaultGroupName=ImageSegmentationApplication
OutputDir=.\Output
OutputBaseFilename=ImageSegmentationApplication
LicenseFile=LICENSE.txt

[Files]
; The Flask app executable
Source: "dist\app.exe"; DestDir: "{app}"; Flags: ignoreversion

; The static folder (recursesubdirs to include subdirectories)
Source: "static\*"; DestDir: "{app}\static"; Flags: recursesubdirs;

; The templates folder (recursesubdirs to include subdirectories)
Source: "templates\*"; DestDir: "{app}\templates"; Flags: recursesubdirs;

; License file (place in app directory for better access after installation)
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion;

; Custom icon for the desktop and Start Menu shortcuts
Source: "app.ico"; DestDir: "{app}"; Flags: ignoreversion;

[Tasks]
; Allow the user to choose if they want a desktop icon
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons"; Flags: unchecked

[Icons]
; Create shortcuts in the Start Menu and on the Desktop
Name: "{{group}}\ImageSegmentationApplication"; Filename: "{app}\app.exe"
Name: "{{desktop}}\ImageSegmentationApplication"; Filename: "{app}\app.exe"; Tasks: desktopicon



















