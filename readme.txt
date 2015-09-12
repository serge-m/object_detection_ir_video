Object detection using machine learning

===Requirements===
Based on Python 2 
sklearn, numpy and many others modules are required

===Step 1===
Create manual markup for training set:
python ./label_img.py  -i vid/frame_00350.jpg

===Step 2===
Generate training set:
python ./generate_or_check_db.py -ti "./vid/frame_{:05d}.jpg" -tl "./vid/frame_{:05d}.jpg.labels.txt" -d "saved_data.hdf5" -s 350 -e 450

interactive generation of training set is possible as an option:
python ./generate_or_check_db.py -ti "./vid/frame_{:05d}.jpg" -tl "./vid/frame_{:05d}.jpg.labels.txt" -d "saved_data.hdf5" -s 350 -e 450 -i


===Step 3===
Train classifier
python ./train_classifier.py -i ./saved_data.hdf5 -o classifier.pickle


===Step 4===
Visualize the results:
python ./generate_or_check_db.py -ti "./vid/frame_{:05d}.jpg" -tl "./vid/frame_{:05d}.jpg.labels.txt" -c "classifier.pickle" -s 350 -e 450 -i


Sergey Matyunin, 2015