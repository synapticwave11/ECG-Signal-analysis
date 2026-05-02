print("jai guru ji")
import numpy as np

a= np.array([1,2,3])
print(a*2)

import numpy as np
import matplotlib.pyplot as plt
SHOW_FAKE_ECG= False

#create time axis
t= np.linspace(0,1,500)


#create fake ECG-like signal
signal= np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(500)

if SHOW_FAKE_ECG:
    plt.plot(t,signal)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.title("Fake ECG")

plt.show()



# -------- MULTICLASS ECG SIGNALS ---------
def generate_ecg(freq, noise_level):
    t= np.linspace(0,1,500)
    clean= np.sin(2* np.pi* freq * t)
    noise= noise_level * np.random.randn(500)
    return clean + noise

normal_ecg= generate_ecg(freq=5, noise_level=0.1)
fast_ecg= generate_ecg(freq=7, noise_level=0.1)
slow_ecg= generate_ecg(freq=2, noise_level=0.1)
noisy_ecg= generate_ecg(freq=5, noise_level=0.4)

plt.figure(figsize=(10,6))

plt.subplot(4,1,1)
plt.plot(normal_ecg)
plt.title("Normal ECG")

plt.subplot(4,1,2)
plt.plot(fast_ecg)
plt.title("Fast ECG")

plt.subplot(4,1,3)
plt.plot(slow_ecg)
plt.title("Slow ECG")

plt.subplot(4,1,4)
plt.plot(noisy_ecg)
plt.title("Noisy ECG")

plt.tight_layout()
plt.show()

np.random.seed(0)
def noise_score(signal):
    return np.std(np.diff(signal))

#Diagnostic plot
plt.hist(normal_ecg, bins=50, alpha=0.5, label="Normal")
plt.hist(noisy_ecg, bins=50, alpha=0.5, label="Noisy")
plt.legend()
plt.title("Value Distribution")
plt.show()

print("Normal:", noise_score(normal_ecg))
print("Fast:", noise_score(fast_ecg))
print("Slow:", noise_score(slow_ecg))
print("Noisy:", noise_score(noisy_ecg))

def classify(signal, threshold=0.25):
    return "Noisy" if noise_score(signal) > threshold else "Clean"

print(classify(normal_ecg))
print(classify(noisy_ecg))

noice_score = np.std(np.diff(signal))