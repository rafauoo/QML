import matplotlib.pyplot as plt
import numpy as np
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Ustawienie ziarna losowego
algorithm_globals.random_seed = 123123

# 📚 Wczytanie zbioru Iris
iris = load_iris()
X = iris.data
y = iris.target

# 🔄 Normalizacja danych
X = MinMaxScaler().fit_transform(X)

# 🧪 Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=algorithm_globals.random_seed)

# 📌 Klasyczny klasyfikator SVM
svm = SVC(kernel='rbf', random_state=algorithm_globals.random_seed)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"🎯 Klasyczny SVM - Accuracy: {svm_accuracy:.2f}")

# 📌 Kwantowy klasyfikator VQC
sampler = Sampler()
objective_func_vals = []

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')
ax.set_title("Objective Function Value vs Iteration (VQC)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Objective Function Value")

def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    line.set_data(range(len(objective_func_vals)), objective_func_vals)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)

vqc = VQC(
    num_qubits=4,  # Liczba kubitów odpowiada liczbie cech
    optimizer=COBYLA(maxiter=50),
    callback=callback_graph,
    sampler=sampler,
)

print("⚛️ Trenowanie VQC...")
vqc.fit(X_train, y_train)
vqc_predictions = vqc.predict(X_test)
vqc_accuracy = accuracy_score(y_test, vqc_predictions)
print(f"⚛️ Kwantowy VQC - Accuracy: {vqc_accuracy:.2f}")

# 🔄 Wyłączenie trybu interaktywnego
plt.ioff()
plt.show()

# 📊 Porównanie wyników
print("\n📊 **Porównanie wyników klasyfikatorów:**")
print(f"✅ Klasyczny SVM Accuracy: {svm_accuracy:.2f}")
print(f"⚛️ Kwantowy VQC Accuracy: {vqc_accuracy:.2f}")
