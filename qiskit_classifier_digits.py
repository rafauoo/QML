from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from qiskit.visualization import circuit_drawer
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.utils import algorithm_globals

# Wczytanie zbioru Digits
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Normalizacja danych
X = MinMaxScaler().fit_transform(X)

# Redukcja wymiarów z 64 do 8 przy pomocy PCA
pca = PCA(n_components=16)
X_reduced = pca.fit_transform(X)

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=algorithm_globals.random_seed)

# Zmienna optymalizatora
optimizer = COBYLA(maxiter=5)
sampler = Sampler()
objective_func_vals = []
# Mapowanie cech przy pomocy ZZFeatureMap (używamy 8 kubitów, więc zakodujemy 8 cech)
feature_map = ZZFeatureMap(feature_dimension=16, reps=2)  # Zmiana liczby kubitów na 8

# Tworzenie ansatzu
ansatz = RealAmplitudes(num_qubits=16, reps=2)


def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    print(len(objective_func_vals), obj_func_eval)
    if len(objective_func_vals) % 10 == 0:  # Co 10 iteracji rysujemy wykres
        plt.plot(range(len(objective_func_vals)), objective_func_vals)
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.title("Objective Function Progress")
        plt.show()


# Tworzenie klasyfikatora VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler,
    callback=callback_graph
)

# Trenowanie klasyfikatora VQC
print("⚛️ Trenowanie VQC...")
vqc.fit(X_train, y_train)
# circuit_drawer(vqc.circuit, output='mpl')
# plt.show()
vqc_predictions = vqc.predict(X_test)
vqc_accuracy = accuracy_score(y_test, vqc_predictions)
print(f"⚛️ Kwantowy VQC - Accuracy: {vqc_accuracy:.2f}")

# Klasyfikator klasyczny (SVM)
svm = SVC(kernel='rbf', random_state=algorithm_globals.random_seed)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"🎯 Klasyczny SVM - Accuracy: {svm_accuracy:.2f}")

# Porównanie wyników
print("\n📊 **Porównanie wyników klasyfikatorów:**")
print(f"✅ Klasyczny SVM Accuracy: {svm_accuracy:.2f}")
print(f"⚛️ Kwantowy VQC Accuracy: {vqc_accuracy:.2f}")
