import numpy as np
import matplotlib.pyplot as plt

# Eğitim verisi (X: koordinatlar, y: sınıflar)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],[0],[0],[1]])  # 0 = kırmızı, 1 = mavi

# Sigmoid ve türevi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# Ağ parametreleri
input_neurons = 2
hidden_neurons = 2
output_neurons = 1
lr = 0.1
epochs = 10000

# Ağırlıkları rastgele başlat
np.random.seed(42)
wh = np.random.rand(input_neurons, hidden_neurons)
bh = np.random.rand(1, hidden_neurons)
wout = np.random.rand(hidden_neurons, output_neurons)
bout = np.random.rand(1, output_neurons)

# Eğitim döngüsü
for i in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    
    output_input = np.dot(hidden_output, wout) + bout
    output = sigmoid(output_input)
    
    # Hata ve backprop
    error = y - output
    d_output = error * sigmoid_derivative(output)
    
    error_hidden = d_output.dot(wout.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # Ağırlık güncelleme
    wout += hidden_output.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

print("Eğitim tamamlandı!")
print("Son tahminler:")
print(output)

# Görselleştirme
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red')
    else:
        plt.scatter(X[i][0], X[i][1], color='blue')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('2 Katmanlı Mini Neural Network Sınıflandırması')
plt.show()
