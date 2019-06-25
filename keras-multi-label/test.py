from sklearn.metrics import classification_report
import pickle
import argparse

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

print("Skuteczność klasyfikacji pojedyńczych atrybutów")
print("Skuteczność dla modelu 1")
print("Skuteczność dla modelu 2")


mlb = pickle.loads(open("models\mlb.pickle", "rb").read())
for label in mlb.classes_:
	print(label)