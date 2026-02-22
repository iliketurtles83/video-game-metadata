# Script to export tables from databases included with ARRM
# After installing ARRM on Windows, database files are located in 
# Users/username/AppData/Roaming/Nexouille Soft/arrm/Database
# mdbtools is a linux toolkit for messing with mdb files

mdb-export launchbox.mdb Game > launchbox.csv
mdb-export gametdb.mdb games_on_gametdb > games_on_gametdb.csv
mdb-export dat_database.mdb dos_database > dat_database_dos.csv
mdb-export dat_database.mdb mame_database > dat_database_mame.csv