rem Erstellt mithilfe von ChatGPT3.5 von OpenAI
rem Setze den Pfad zum Verzeichnis mit den Dateien
set "directory=Z:\04_Daten\GroundThruth\Fussverkehrausfkommen\OTC07_23-09-19_17-00-00\vary_sigmaIOU"

rem Durchlaufe alle Dateien mit der Endung ".otdet" im Verzeichnis und seinen Unterverzeichnissen
for /r "%directory%" %%f in (*.otdet) do (
    echo Bearbeite Datei: %%f
    rem Schleife durch alle Sigma_IoU Werte von 0,30 bis 0,49 in 0,01-Schritten
    for /L %%i in (30, 1, 49) do (
        set "sigma_iou=0.%%i"
        set "filename_end=%%i"
        rem Extrahiere den Dateinamen ohne Erweiterung
        set "filename=%%~nf"

        echo ***********************************************************
        echo !filename!_!filename_end!.ottrk

        rem Führe den Befehl aus, um das Skript "detect.py" aufzurufen und die Ergebnisse in eine Datei ".ottrk" zu speichern
        python track.py -p "%%f" --sigma_iou=!sigma_iou! > "!directory!\!filename!_!filename_end!.ottrk"
        
        
    )
)
