# Cómo utilizar esto?
0. descargar el modelo del hipercubo y colocarlo en \model -> https://drive.google.com/drive/folders/1XtLQvLao4psy3FObWyUoUtDl5-eigjBY
1. instalar las dependencias de requirements_bigdata.txt
2. ejecutar data_extraction.py. Ahí está el código básico para obtener imágenes
3. Ejecutar data_to_json.py para generar los json y dárselos al modelo
4. Ejecutar Embeddings_tiny.py. Aquí se encuentra el código para generar los embeddings en el formato de hipercubo de 16 dimensiones. 

`python embeddings_tiny.py --data_dir ..\data\landing --ckpt ..\model\bioclip2_16dim_1epochs.pth --device cpu --bs 4 --n_workers 0 --hashcoder small --bitdim 16 `

5. Ejecutar inference.py para realizar una prueba del funcionamiento del hipercubo. 

6. Para obtener otro modelo del hipercubo, debe entrenarse usando el repositorio original de wildlifehashing