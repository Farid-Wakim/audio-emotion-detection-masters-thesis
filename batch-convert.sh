# !/bin/bash
echo "START OF CONVERSION"
for i in /content/drive/MyDrive/Colab-Notebooks/data/*.mp4; do
    [ -f "$i" ] || break
    echo "$i"
    output="/content/drive/MyDrive/Colab-Notebooks/data/$(basename $i .mp4)-audio.wav"
    echo "$output"
    ffmpeg -loglevel quiet -i "$i" -ac 1 -f wav "$output"
done
echo "END OF CONVERSION"
