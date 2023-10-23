{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 34,
      "metadata": {
        "id": "lIYdn1woOS1n"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()"
      ],
      "metadata": {
        "id": "g4ks-OrsK4ao"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n6in5J9tL73r",
        "outputId": "3c777954-7d26-42a1-9f50-cfc238d36c4b"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000, 28, 28)"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_train.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4JwSX8vpMOEZ",
        "outputId": "bd255d9c-e894-4a74-ae00-a42498ed7d04"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(60000,)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = X_train.reshape(X_train.shape[0], 28*28)\n",
        "X_test = X_test.reshape(X_test.shape[0], 28*28)"
      ],
      "metadata": {
        "id": "f-68Hg-NMV_X"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for i in range(9):\n",
        "  plt.subplot(331+i)\n",
        "  plt.imshow(X_train[i].reshape(28,28), cmap='gray')\n",
        "  plt.title(f'label: {y_train[i]}')\n",
        "  plt.xticks([])\n",
        "  plt.yticks([])\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 486
        },
        "id": "-M7FUgyeRobE",
        "outputId": "5166318e-07eb-403b-b2c2-e431358dad24"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 9 Axes>"
            ],
            "image/png": "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VOfPny9dlpCQoERELVy4sHRZcXGx6tSpk2rUqJEqKSlRSim1Zs0aFRISorZt26btc9myZUpE1I4dO0qXxcTEqFGjRmnrxcTEqJiYGK/+XKdPn1YioqZPn+7VdqjegnE8zJ07V4lI6a++ffuq48ePe7w9qq9gHA9KBd7Ph4D7OOPS/6EUFBRITk6O3HTTTVJUVCQZGRnaujVr1pSkpKTSHBoaKklJSZKdnS179uwREZG33npL2rZtK23atJGcnJzSX7fccouIiKSnp5dZT2ZmZnB3majSAn08DBs2TDZt2iT/+Mc/5IEHHhCRX65OACYCfTwEooD7OOPAgQMydepU2bx5s+Tn52uv5eXlaTk6Olpq166tLWvdurWI/HJyu3fvLocPH5aDBw9ediJLdna2D6sHfCvQx0NMTIzExMSIyC8NxdixY6Vfv37y9ddf85EGvBbo4yEQBVQTkZubKwkJCVK3bl15+umnJS4uTsLDw2Xv3r0yZcoUcTqdXu/T6XRKhw4dZNGiRS5fb968eXnLBipEMI6He+65R1asWCFbt26V2267rUKPheASjOMhEARUE7Flyxb58ccfZf369dK7d+/S5UePHnW5flZWlhQWFmrd5qFDh0TkvzNh4+Li5IsvvpC+ffuKw+GouOIBHwvG8fDrRxnW/zUC7gTjeAgEATUn4tfvjatL7mhXUlIiS5cudbn+hQsXZPny5dq6y5cvl4YNG0qXLl1ERGTo0KFy6tQpWbFihW37X2eOl6W8X2kDTAXyeDh9+rTL5a+88oo4HA7p3Lmz230Alwrk8RDIAupKRI8ePaRBgwYyatQomThxojgcDlmzZs1lb5MbHR0t8+bNk8zMTGndurWsW7dO9u3bJy+//LLUqlVLRERGjBghqamp8uijj0p6err07NlTLl68KBkZGZKamippaWnStWvXy9bkzVd41qxZI8eOHZOioiIREdm6davMmjWrtI5fPxsGPBHI42H27NmyY8cOuf3226VFixby008/ydtvvy27du2SCRMmSKtWrczeFFRbgTweRAL454M/vxrijquv8OzYsUN1795dRUREqOjoaJWcnKzS0tKUiKj09PTS9RISElS7du3U7t27VXx8vAoPD1cxMTFqyZIltuOUlJSoefPmqXbt2qmwsDDVoEED1aVLFzVjxgyVl5dXul55v8Lz69eKXP26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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "img = X_test[32].reshape(28,28)\n",
        "plt.imsave('t1.png', img, cmap='gray')"
      ],
      "metadata": {
        "id": "7A2t5k-MPnlu"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "clf = RandomForestClassifier(n_estimators=200)\n",
        "clf.fit(X_train, y_train)\n",
        "y_pred = clf.predict(X_test)\n",
        "accuracy_score(y_pred, y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HtxZTNSjUq85",
        "outputId": "54c660fe-d594-4466-e560-e2a5bb90c31e"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9707"
            ]
          },
          "metadata": {},
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "joblib.dump(clf, 'random_forest_mnist.pkl')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DXgYcsWyif_R",
        "outputId": "f3924f79-6b5a-4c84-d9f4-6bde89e5640b"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['random_forest_mnist.pkl']"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "a_LPWeznkQ70"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}