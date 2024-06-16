FILE="gamelist.xml"

for folder in ./*; do
	if [ -f "$folder/$FILE" ]; then 
		mkdir -p ./lists/"$folder"
		cp "$folder/$FILE" ./lists/"$folder"
	fi
done
