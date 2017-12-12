# Amity Card Color Recognition - Simple
This is a simple Card color recognizer developed using K-Nearest Neighbor Classifier

## Usage
Collect Dataset of:
1. ~10 cards (Orange - Day Scholars)
2. ~10 cards (Green - Hosteliers)
3. ~10 cards (White - Faculties)

Save Day Scholars card images in path ```./datasets/Orange```<br/>
Save Hosteliers card images in path ```./datasets/Green```<br/>
Save Faculty card images in path ```./datasets/White```

Run command ```python dataset_gen.py``` and enter the sub folder (Orange or Green or White) when specified 
to generate more images.

Run Command ```python predict.py``` to see the result.

## Dependencies
1. OpenCv2
2. Numpy
3. Matplotlib
